from pymongo import MongoClient

from config import MONGO_URI, DB_NAME, COLL_NAME
from job1 import chia_splits

SO_SPLIT = 4

# Bo dieu kien rong: khong rang buoc gi, tra ve toan bo benh nhan.
# Moi khoa la mot rang buoc tren mot chi so; None / list rong = bo qua chi so do.
DIEU_KIEN_RONG = {
    "tuoi": None, "bmi": None, "hba1c": None, "glucose": None,
    "gioi": [], "hut": [], "nhan": None, "cao_ha": None, "benh_tim": None,
}

# ten dieu kien -> ten truong trong MongoDB, dung cho cac dieu kien dang khoang
KHOANG = {
    "tuoi": "age",
    "bmi": "bmi",
    "hba1c": "HbA1c_level",
    "glucose": "blood_glucose_level",
}


# InputFormat
def doc_input():
    client = MongoClient(MONGO_URI)
    docs = list(client[DB_NAME][COLL_NAME].find({}, {"_id": 0}))
    client.close()
    return docs


# mot record co thoa toan bo dieu kien loc khong
def thoa(doc, dk):
    for ten, truong in KHOANG.items():
        khoang = dk.get(ten)
        if khoang and not (khoang[0] <= doc[truong] <= khoang[1]):
            return False
    if dk.get("gioi") and doc["gender"] not in dk["gioi"]:
        return False
    if dk.get("hut") and doc["smoking_history"] not in dk["hut"]:
        return False
    for ten, truong in (("nhan", "diabetes"), ("cao_ha", "hypertension"), ("benh_tim", "heart_disease")):
        if dk.get(ten) is not None and doc[truong] != dk[ten]:
            return False
    return True


# map: chi phat ra record thoa dieu kien, bo qua record con lai.
# Giong Lab 2 cau 7 (loc record theo dieu kien cot), khac o cho reduce tra ve
# ca danh sach chu khong chi con dem.
def map_fn(doc, dk):
    if thoa(doc, dk):
        yield "khop", doc


# mot map task: chi co map(). Khong combine va khong partition:
#   - chi mot khoa ("khop") -> partition thanh nhieu Region la vo nghia.
#   - reduce can chinh danh sach record de hien thi, gom cuc bo khong bot duoc
#     khoi luong phai chuyen qua shuffle -> combine khong co loi gi.
def map_task(split, dk):
    return [kv for doc in split for kv in map_fn(doc, dk)]


# shuffle: gom output cac map task theo khoa
def shuffle(cac_task_pairs):
    g = {}
    for pairs in cac_task_pairs:
        for khoa, val in pairs:
            g.setdefault(khoa, []).append(val)
    return g


# reduce: mot khoa -> (so benh nhan khop, danh sach benh nhan)
def reduce_fn(values):
    return len(values), values


# chay_job = JobTracker. docs truyen tu ngoai vao de app.py doc Mongo mot lan
# roi loc nhieu lan, khong phai doc lai moi lan doi bo loc.
def chay_job(dk, docs=None, so_split=SO_SPLIT, log=False):
    if docs is None:
        docs = doc_input()
    splits = chia_splits(docs, so_split)

    map_out = []
    for i, sp in enumerate(splits, 1):
        pairs = map_task(sp, dk)
        map_out.append(pairs)
        if log:
            print(f"Map task {i}: {len(sp):>6} record -> {len(pairs):>6} khop")

    grouped = shuffle(map_out)
    if not grouped:
        return 0, []
    return reduce_fn(grouped["khop"])


if __name__ == "__main__":
    # Vi du: nam tren 60 tuoi, da mac benh
    dk = dict(DIEU_KIEN_RONG, tuoi=(60, 120), gioi=["Male"], nhan=1)
    tong, ket_qua = chay_job(dk, log=True)
    print(f"\nTim thay {tong} benh nhan. 5 dong dau:")
    for d in ket_qua[:5]:
        print(" ", d)
