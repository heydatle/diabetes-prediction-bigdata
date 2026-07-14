from pymongo import MongoClient

from config import (
    MONGO_URI, DB_NAME, COLL_NAME,
    NGUONG_TUOI, NGUONG_HBA1C, NGUONG_GLUCOSE, NGUONG_BMI,
)

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


# Xac dinh nhom cho mot gia tri so: nhan lay tu BUCKETS, vi tri chon theo nguong.
# len(nguong) phai bang len(nhan) - 1 (vd 3 nguong tuoi -> 4 nhom tuoi).
def nhom(gia_tri, nguong, nhan):
    for t, n in zip(nguong, nhan):
        if gia_tri < t:
            return n
    return nhan[-1]


# InputFormat: doc du lieu tu MongoDB (DFS/table cua Concept 1)
def doc_input():
    client = MongoClient(MONGO_URI)
    docs = list(client[DB_NAME][COLL_NAME].find({}, {"_id": 0}))
    client.close()
    return docs


# Chia du lieu thanh nhieu split -> gia lap phan bo cho cac task tracker
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


# shuffle: gom cac gia tri theo cung khoa
def shuffle(pairs):
    g = {}
    for key, val in pairs:
        g.setdefault(key, []).append(val)
    return g


# reduce: mot khoa -> (tong so, so mac, ty le mac %)
def reduce_fn(values):
    n = sum(c for c, _ in values)
    mac = sum(y for _, y in values)
    return n, mac, 100.0 * mac / n


def in_ket_qua(ket_qua):
    for chi_so, cac_nhom in BUCKETS.items():
        print(chi_so + ":")
        for ten_nhom in cac_nhom:
            r = ket_qua.get((chi_so, ten_nhom))
            if r:
                n, mac, ty_le = r
                print(f"  {ten_nhom:<12}: n={n:>6}  mac={mac:>5}  ty le mac={ty_le:>6.2f}%")
        print()


def chay_job(so_split=4):
    docs = doc_input()
    splits = chia_splits(docs, so_split)

    map_out = []
    for i, sp in enumerate(splits, 1):
        pairs = [kv for doc in sp for kv in map_fn(doc)]
        map_out.extend(pairs)
        print(f"Map task {i}: {len(sp):>6} record -> {len(pairs)} cap")

    grouped = shuffle(map_out)
    print(f"Shuffle: {len(grouped)} khoa\n")

    ket_qua = {k: reduce_fn(v) for k, v in grouped.items()}
    in_ket_qua(ket_qua)
    return ket_qua


if __name__ == "__main__":
    chay_job()
