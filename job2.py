import json
import math
import random

from pymongo import MongoClient

from config import MONGO_URI, DB_NAME, COLL_NAME, MODEL_PATH
from job1 import chia_splits

# Sieu tham so huan luyen (giai thich trong mo_ta.txt)
SEED = 42
SO_SPLIT = 4          # so map task moi vong lap
TY_LE_TEST = 0.2
TOC_DO_HOC = 1.0
SO_VONG = 1500        # tran vong toi da; thuc te dung som khi hoi tu
KIEM = 25             # moi bao nhieu vong thi kiem tra hoi tu
TOL = 1e-5            # loss giam < TOL sau KIEM vong -> coi la hoi tu, dung
NGUONG = 0.5

# Dac trung so -> chuan hoa mean/std (tinh tu tap train)
SO = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
# Bien hang muc -> one-hot, bo mot nhom tham chieu (giong glm trong R):
#   gioi tinh bo "Female", hut thuoc bo "No Info"
GIOI = ["Male", "Other"]
HUT = ["current", "ever", "former", "never", "not current"]

# Ten tung dac trung, dung thu tu voi vector() -> de doc lai trong so w
TEN_DAC_TRUNG = (
    SO
    + ["hypertension", "heart_disease"]
    + ["gioi=" + g for g in GIOI]
    + ["hut_thuoc=" + h for h in HUT]
)


# InputFormat
# sort("_id"): Mongo khong cam ket thu tu neu khong sort; tach train/test o duoi
# phu thuoc thu tu nay nen can co dinh de tai lap duoc ket qua.
def doc_input():
    client = MongoClient(MONGO_URI)
    docs = list(client[DB_NAME][COLL_NAME].find({}, {"_id": 0}).sort("_id", 1))
    client.close()
    return docs


# xao tron tat dinh theo SEED, cat theo TY_LE_TEST
def tach_train_test(docs):
    rng = random.Random(SEED)
    docs = docs[:]
    rng.shuffle(docs)
    n_test = int(len(docs) * TY_LE_TEST)
    return docs[n_test:], docs[:n_test]


# tinh tu train, khong dung test (tranh ro ri)
def thong_ke_chuan_hoa(train):
    mean, std = {}, {}
    n = len(train)
    for c in SO:
        m = sum(d[c] for d in train) / n
        var = sum((d[c] - m) ** 2 for d in train) / n
        mean[c] = m
        std[c] = math.sqrt(var) or 1.0
    return mean, std


# record -> vector (so da chuan hoa + nhi phan + one-hot)
def vector(doc, mean, std):
    x = [(doc[c] - mean[c]) / std[c] for c in SO]
    x.append(float(doc["hypertension"]))
    x.append(float(doc["heart_disease"]))
    x += [1.0 if doc["gender"] == g else 0.0 for g in GIOI]
    x += [1.0 if doc["smoking_history"] == h else 0.0 for h in HUT]
    return x


def sigmoid(z):
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def du_doan_p(x, w, b):
    return sigmoid(b + sum(wj * xj for wj, xj in zip(w, x)))


# map: mot record -> dong gop gradient = (sigmoid(z) - y) * x  va sai so cho b
def map_fn(mau, w, b):
    x, y = mau
    err = du_doan_p(x, w, b) - y
    grad_w = [err * xj for xj in x]
    return grad_w, err


# reduce: cong cac dong gop gradient -> gradient tong (gw, gb)
# Nhan duoc ca dong gop tho tu map lan tong cuc bo tu combine: hai thu cung dang
# (vector d chieu, sai so) nen cong chung mot cach.
def reduce_fn(dong_gop, d):
    gw = [0.0] * d
    gb = 0.0
    for grad_w, err in dong_gop:
        for j in range(d):
            gw[j] += grad_w[j]
        gb += err
    return gw, gb


# mot map task: map() tung record roi combine() cong cuc bo ngay trong task.
# Combine chinh la reduce_fn: cung la phep cong, va cong co tinh ket hop nen
# cong theo split roi cong tiep o reduce van ra dung gradient toan tap.
# Khong partition: ca job chi co mot khoa (gradient tong) -> chia Region vo nghia.
def map_task(split, w, b, d):
    dong_gop = (map_fn(m, w, b) for m in split)
    return reduce_fn(dong_gop, d)


# huan_luyen = JobTracker: moi vong la mot job MapReduce (map/combine -> reduce)
# dung som khi loss gan nhu khong giam nua, khong doan cung so vong
def huan_luyen(mau_train, so_split=SO_SPLIT):
    d = len(mau_train[0][0])
    w = [0.0] * d
    b = 0.0
    n = len(mau_train)
    splits = chia_splits(mau_train, so_split)
    loss_truoc = float("inf")
    for vong in range(1, SO_VONG + 1):
        regions = [map_task(sp, w, b, d) for sp in splits]   # map + combine
        gw, gb = reduce_fn(regions, d)                        # reduce
        for j in range(d):
            w[j] -= TOC_DO_HOC * gw[j] / n
        b -= TOC_DO_HOC * gb / n
        if vong % KIEM == 0:
            l = loss(mau_train, w, b)
            print(f"  vong {vong:>4}: loss = {l:.5f}")
            if loss_truoc - l < TOL:
                print(f"  -> hoi tu (loss giam < {TOL} sau {KIEM} vong), dung.")
                break
            loss_truoc = l
    return w, b


# Binary cross-entropy, chi de theo doi hoi tu
def loss(mau, w, b):
    tong = 0.0
    for x, y in mau:
        p = du_doan_p(x, w, b)
        p = min(max(p, 1e-12), 1 - 1e-12)
        tong += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return tong / len(mau)


# Confusion matrix + Accuracy/Precision/Recall/F1 tren tap test
def danh_gia(mau_test, w, b):
    tp = fp = tn = fn = 0
    for x, y in mau_test:
        du_doan = 1 if du_doan_p(x, w, b) >= NGUONG else 0
        if du_doan == 1 and y == 1:
            tp += 1
        elif du_doan == 1 and y == 0:
            fp += 1
        elif du_doan == 0 and y == 0:
            tn += 1
        else:
            fn += 1
    acc = (tp + tn) / len(mau_test)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "acc": acc, "prec": prec, "rec": rec, "f1": f1}


# Moc so sanh: doan tat ca la "khong mac" (ZeroR)
def moc_doan_da_so(mau_test):
    khong_mac = sum(1 for _, y in mau_test if y == 0)
    return khong_mac / len(mau_test)


def in_ket_qua(kq, moc, w, b):
    print("\nCONFUSION MATRIX (tap test):")
    print("                du doan MAC   du doan KHONG")
    print(f"  thuc MAC   :  TP={kq['tp']:>5}      FN={kq['fn']:>5}")
    print(f"  thuc KHONG :  FP={kq['fp']:>5}      TN={kq['tn']:>5}")
    print(f"\n  Accuracy : {kq['acc'] * 100:6.2f}%")
    print(f"  Precision: {kq['prec'] * 100:6.2f}%")
    print(f"  Recall   : {kq['rec'] * 100:6.2f}%")
    print(f"  F1       : {kq['f1'] * 100:6.2f}%")
    print(f"\n  Moc doan-da-so (ZeroR): accuracy {moc * 100:.2f}%, recall nhom mac = 0")

    print("\nTRONG SO (giam dan theo do lon |w|):")
    xep = sorted(zip(TEN_DAC_TRUNG, w), key=lambda t: -abs(t[1]))
    for ten, wj in xep:
        print(f"  {ten:<22} {wj:+.4f}")
    print(f"  {'(bias b)':<22} {b:+.4f}")


def luu_model(w, b, mean, std):
    model = {
        "ten_dac_trung": TEN_DAC_TRUNG,
        "w": w,
        "b": b,
        "mean": mean,
        "std": std,
        "so": SO,
        "gioi": GIOI,
        "hut": HUT,
        "nguong": NGUONG,
    }
    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)
    print(f"\nDa luu model vao {MODEL_PATH}")


def chay_job():
    docs = doc_input()
    train, test = tach_train_test(docs)
    mean, std = thong_ke_chuan_hoa(train)
    mau_train = [(vector(d, mean, std), d["diabetes"]) for d in train]
    mau_test = [(vector(d, mean, std), d["diabetes"]) for d in test]
    print(f"Train {len(mau_train)} / Test {len(mau_test)}  |  {len(TEN_DAC_TRUNG)} dac trung")

    print(f"\nHuan luyen Logistic Regression (gradient descent dang MapReduce, "
          f"{SO_SPLIT} map task/vong):")
    w, b = huan_luyen(mau_train)

    kq = danh_gia(mau_test, w, b)
    moc = moc_doan_da_so(mau_test)
    in_ket_qua(kq, moc, w, b)
    luu_model(w, b, mean, std)
    return w, b, kq


if __name__ == "__main__":
    chay_job()
