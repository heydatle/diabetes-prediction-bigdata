import csv
from pymongo import MongoClient

from config import CSV_PATH, MONGO_URI, DB_NAME, COLL_NAME

SO_NGUYEN = ("hypertension", "heart_disease", "blood_glucose_level", "diabetes")
SO_THUC = ("age", "bmi", "HbA1c_level")


def doc_va_khu_trung(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = list(rows[0].keys())
    seen, sach = set(), []
    for r in rows:
        khoa = tuple(r[c] for c in cols)
        if khoa not in seen:
            seen.add(khoa)
            sach.append(r)
    return rows, sach


def chuyen_kieu(r):
    doc = dict(r)
    for c in SO_NGUYEN:
        doc[c] = int(float(doc[c]))
    for c in SO_THUC:
        doc[c] = float(doc[c])
    return doc


def main():
    goc, sach = doc_va_khu_trung(CSV_PATH)
    print(f"Doc {len(goc)} dong, khu trung con {len(sach)}")

    client = MongoClient(MONGO_URI)
    coll = client[DB_NAME][COLL_NAME]
    coll.drop()
    coll.insert_many(chuyen_kieu(r) for r in sach)
    print(f"Da nap {coll.count_documents({})} document vao {DB_NAME}.{COLL_NAME}")
    client.close()


if __name__ == "__main__":
    main()
