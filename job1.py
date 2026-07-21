from pymongo import MongoClient

from config import (
    MONGO_URI, DB_NAME, COLL_NAME,
    NGUONG_TUOI, NGUONG_HBA1C, NGUONG_GLUCOSE, NGUONG_BMI,
)

SO_SPLIT = 4    # so map task (M1..M4 trong so do Concept 1)
SO_REDUCE = 2   # so reduce task (R1, R2) -> moi map task ghi ra 2 region

BUCKETS = {
    "Tuoi":         ["<40", "40-49", "50-59", ">=60"],
    "HbA1c":        ["<5.7", "5.7-6.4", ">=6.5"],
    "Glucose":      ["<100", "100-125", ">=126"],
    "BMI":          ["<18.5", "18.5-24.9", "25-29.9", ">=30"],
    "Cao huyet ap": ["khong", "co"],
    "Benh tim":     ["khong", "co"],
    "Gioi tinh":    ["Female", "Male", "Other"],
    "Hut thuoc":    ["No Info", "current", "ever", "former", "never", "not current"],
}


# len(nguong) phai bang len(nhan) - 1 (vd 3 nguong tuoi -> 4 nhom tuoi)
def nhom(gia_tri, nguong, nhan):
    for t, n in zip(nguong, nhan):
        if gia_tri < t:
            return n
    return nhan[-1]


# InputFormat
def doc_input():
    client = MongoClient(MONGO_URI)
    docs = list(client[DB_NAME][COLL_NAME].find({}, {"_id": 0}))
    client.close()
    return docs


# split
def chia_splits(items, so_split):
    n = len(items)
    return [items[i * n // so_split:(i + 1) * n // so_split] for i in range(so_split)]


# map: moi record -> cac cap ((chi_so, nhom), (dem, nhan))
def map_fn(doc):
    y = doc["diabetes"]
    yield ("Tuoi", nhom(doc["age"], NGUONG_TUOI, BUCKETS["Tuoi"])), (1, y)
    yield ("HbA1c", nhom(doc["HbA1c_level"], NGUONG_HBA1C, BUCKETS["HbA1c"])), (1, y)
    yield ("Glucose", nhom(doc["blood_glucose_level"], NGUONG_GLUCOSE, BUCKETS["Glucose"])), (1, y)
    yield ("BMI", nhom(doc["bmi"], NGUONG_BMI, BUCKETS["BMI"])), (1, y)
    yield ("Cao huyet ap", "co" if doc["hypertension"] else "khong"), (1, y)
    yield ("Benh tim", "co" if doc["heart_disease"] else "khong"), (1, y)
    yield ("Gioi tinh", doc["gender"]), (1, y)
    yield ("Hut thuoc", doc["smoking_history"]), (1, y)


# combine: gom cuc bo ngay trong map task, (dem, nhan) -> (tong dem, tong mac).
# Gom truoc duoc vi phep cong co tinh ket hop: cong cuc bo roi cong tiep o reduce
# van ra dung mot ket qua. Dau ra combine cung dang voi dau ra map nen reduce_fn
# dung lai nguyen ven, khong can sua.
def combine(pairs):
    g = {}
    for khoa, (dem, y) in pairs:
        n, mac = g.get(khoa, (0, 0))
        g[khoa] = (n + dem, mac + y)
    return list(g.items())


# partition: khoa nay ve reduce task nao. Cung mot khoa luon ve cung mot reducer,
# neu khong thi moi reducer chi thay mot phan va tinh ty le mac sai.
# Tu bam thay vi hash() san co: hash() cua chuoi doi theo tung lan chay Python
# (PYTHONHASHSEED ngau nhien) -> khong tai lap duoc ket qua.
def partition(khoa, so_reduce):
    return sum(ord(c) for c in str(khoa)) % so_reduce


# mot map task: map() -> RAM -> combine() -> partition() -> Region1..RegionR
def map_task(split, so_reduce):
    pairs = [kv for doc in split for kv in map_fn(doc)]
    gom = combine(pairs)
    regions = [[] for _ in range(so_reduce)]
    for khoa, val in gom:
        regions[partition(khoa, so_reduce)].append((khoa, val))
    return regions, len(pairs)


# shuffle: reduce task r doc Region r cua tat ca map task, gom theo khoa
def shuffle(cac_map_out, r):
    g = {}
    for regions in cac_map_out:
        for khoa, val in regions[r]:
            g.setdefault(khoa, []).append(val)
    return g


# reduce: mot khoa -> (tong so, so mac, ty le mac %)
def reduce_fn(values):
    n = sum(c for c, _ in values)
    mac = sum(y for _, y in values)
    return n, mac, 100.0 * mac / n


# mot reduce task: read -> sort -> reduce()
def reduce_task(grouped):
    return {khoa: reduce_fn(grouped[khoa]) for khoa in sorted(grouped)}


# OutputFormat
def in_ket_qua(ket_qua):
    for chi_so, cac_nhom in BUCKETS.items():
        print(chi_so + ":")
        for ten_nhom in cac_nhom:
            r = ket_qua.get((chi_so, ten_nhom))
            if r:
                n, mac, ty_le = r
                print(f"  {ten_nhom:<12}: n={n:>6}  mac={mac:>5}  ty le mac={ty_le:>6.2f}%")
        print()


# Toan bo luong MapReduce, tach rieng de app.py dung lai (log=False cho khoi in)
def chay_pipeline(docs, so_split=SO_SPLIT, so_reduce=SO_REDUCE, log=False):
    splits = chia_splits(docs, so_split)

    map_out = []
    for i, sp in enumerate(splits, 1):
        regions, so_cap = map_task(sp, so_reduce)
        map_out.append(regions)
        if log:
            con = sum(len(r) for r in regions)
            print(f"Map task {i}: {len(sp):>6} record -> {so_cap:>6} cap -> combine con {con:>3} cap")

    ket_qua = {}
    for r in range(so_reduce):
        grouped = shuffle(map_out, r)
        if log:
            print(f"Reduce task {r + 1}: doc Region {r + 1} cua {len(map_out)} map task -> {len(grouped)} khoa")
        ket_qua.update(reduce_task(grouped))
    return ket_qua


# chay_job = JobTracker: dieu phoi map -> combine/partition -> shuffle -> reduce
def chay_job(so_split=SO_SPLIT, so_reduce=SO_REDUCE):
    docs = doc_input()
    ket_qua = chay_pipeline(docs, so_split, so_reduce, log=True)
    print()
    in_ket_qua(ket_qua)
    return ket_qua


if __name__ == "__main__":
    chay_job()
