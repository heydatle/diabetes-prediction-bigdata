import json

import pandas as pd
import streamlit as st

from config import MODEL_PATH
import job1
import job2
import job3

# Tap gia tri day du cua bien hang muc (gom ca nhom tham chieu ma Job 2 bo di)
GIOI_FULL = ["Female", "Male", "Other"]
HUT_FULL = ["No Info", "current", "ever", "former", "never", "not current"]

st.set_page_config(page_title="Du doan nguy co tieu duong", layout="wide")


# Doc Mongo mot lan, dung chung cho Job 1 va Job 3
@st.cache_data
def doc_patients():
    return job1.doc_input()


# cache de khong tinh lai moi lan; pandas khong dung o day
@st.cache_data
def thong_ke_job1():
    return job1.chay_pipeline(doc_patients())


@st.cache_data
def load_model():
    with open(MODEL_PATH, encoding="utf-8") as f:
        return json.load(f)


def tab_thong_ke():
    st.header("Thong ke ty le mac theo tung chi so")
    st.caption("Bang tinh bang Job 1 (MapReduce): moi record phat ra (chi so, nhom), "
               "combine gom cuc bo trong tung map task, partition chia khoa ve cac "
               "reduce task, reduce tinh ty le mac. Nguon: MongoDB diabetes.patients.")
    ket_qua = thong_ke_job1()

    for chi_so, cac_nhom in job1.BUCKETS.items():
        bang = []
        for ten_nhom in cac_nhom:
            r = ket_qua.get((chi_so, ten_nhom))
            if not r:
                continue
            n, mac, ty_le = r
            bang.append({
                "Nhom": ten_nhom,
                "So nguoi": n,
                "So mac": mac,
                "Ty le mac (%)": round(ty_le, 2),
            })
        df = pd.DataFrame(bang)
        st.subheader(chi_so)
        col1, col2 = st.columns(2)
        col1.dataframe(df, hide_index=True)
        col2.bar_chart(df.set_index("Nhom")["Ty le mac (%)"])


def tab_tim_kiem():
    st.header("Tim kiem - loc benh nhan theo tung chi so")
    st.caption("Loc bang Job 3 (MapReduce): map giu lai record thoa dieu kien, reduce "
               "gom thanh danh sach va dem. Moi dieu kien duoi day tuong ung mot rang "
               "buoc tren mot chi so. Cot Chan doan la nhan that da co san trong du "
               "lieu, khac voi tab Du doan la suy luan cho nguoi moi.")

    c1, c2, c3 = st.columns(3)
    gioi = c1.multiselect("Gioi tinh", GIOI_FULL)
    hut = c1.multiselect("Tien su hut thuoc", HUT_FULL)
    nhan = c1.selectbox("Chan doan", ["Tat ca", "Mac (1)", "Khong mac (0)"])

    tuoi = c2.slider("Tuoi", 0.0, 100.0, (0.0, 100.0))
    bmi = c2.slider("BMI", 10.0, 60.0, (10.0, 60.0))

    hba1c = c3.slider("HbA1c", 3.0, 9.0, (3.0, 9.0))
    glucose = c3.slider("Glucose", 80, 300, (80, 300))
    cao_ha = c3.selectbox("Cao huyet ap", ["Tat ca", "Co", "Khong"])
    benh_tim = c3.selectbox("Benh tim", ["Tat ca", "Co", "Khong"])

    dk = dict(
        job3.DIEU_KIEN_RONG,
        tuoi=tuoi, bmi=bmi, hba1c=hba1c, glucose=glucose,
        gioi=gioi, hut=hut,
        nhan=None if nhan == "Tat ca" else (1 if nhan.startswith("Mac") else 0),
        cao_ha=None if cao_ha == "Tat ca" else (1 if cao_ha == "Co" else 0),
        benh_tim=None if benh_tim == "Tat ca" else (1 if benh_tim == "Co" else 0),
    )

    tong, ket_qua = job3.chay_job(dk, doc_patients())
    st.write(f"Tim thay {tong} benh nhan. Hien thi toi da 200 dong dau.")
    if ket_qua:
        st.dataframe(pd.DataFrame(ket_qua[:200]), hide_index=True)


def tab_du_doan():
    st.header("Du doan nguy co tieu duong cho mot nguoi")
    st.caption("Nhap chi so -> chuan hoa -> p = sigmoid(w.x + b) tu model.json (Job 2). "
               "Nhan du doan 'mac' neu p >= nguong.")
    model = load_model()

    c1, c2 = st.columns(2)
    gioi = c1.selectbox("Gioi tinh", GIOI_FULL)
    tuoi = c1.number_input("Tuoi", 0.0, 120.0, 40.0, step=1.0)
    bmi = c1.number_input("BMI", 10.0, 60.0, 27.0, step=0.1)
    hut = c1.selectbox("Tien su hut thuoc", HUT_FULL)

    hba1c = c2.number_input("HbA1c", 3.0, 15.0, 5.5, step=0.1)
    glucose = c2.number_input("Glucose", 50, 400, 130, step=1)
    cao_ha = c2.checkbox("Co cao huyet ap")
    benh_tim = c2.checkbox("Co benh tim")

    if st.button("Du doan"):
        doc = {
            "gender": gioi,
            "age": tuoi,
            "bmi": bmi,
            "smoking_history": hut,
            "HbA1c_level": hba1c,
            "blood_glucose_level": glucose,
            "hypertension": 1 if cao_ha else 0,
            "heart_disease": 1 if benh_tim else 0,
        }
        # dung chinh ham vector() cua job2.py -> train va du doan ma hoa y het nhau
        x = job2.vector(doc, model["mean"], model["std"])
        p = job2.du_doan_p(x, model["w"], model["b"])
        nhan = "MAC" if p >= model["nguong"] else "KHONG MAC"

        st.metric("Xac suat mac", f"{p * 100:.2f}%")
        if p >= model["nguong"]:
            st.error(f"Du doan: {nhan} (nguong {model['nguong']})")
        else:
            st.success(f"Du doan: {nhan} (nguong {model['nguong']})")

        # Dong gop tung yeu to = w_j * x_j: doc duoc yeu to nao day nguy co
        df = pd.DataFrame({
            "Yeu to": model["ten_dac_trung"],
            "Dong gop (w*x)": [round(wj * xj, 3) for wj, xj in zip(model["w"], x)],
        }).sort_values("Dong gop (w*x)", ascending=False)
        st.subheader("Yeu to anh huong")
        st.caption("Duong = day nguy co len, am = keo nguy co xuong.")
        st.dataframe(df, hide_index=True)


st.title("He thong du doan nguy co tieu duong")
t1, t2, t3 = st.tabs(["Thong ke", "Tim kiem - loc", "Du doan"])
with t1:
    tab_thong_ke()
with t2:
    tab_tim_kiem()
with t3:
    tab_du_doan()
