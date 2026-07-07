from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Heart Disease Prediction | KNN",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }
    .big-title {
        font-size: 2.2rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #ff4b4b, #ff8a8a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #eee;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #ff4b4b, #ff7676);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 0;
        font-size: 1.1rem;
        transition: 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(255,75,75,0.4);
    }
    .result-card-positive {
        background: linear-gradient(135deg, #ff6b6b, #ff4757);
        color: white;
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255,71,87,0.4);
    }
    .result-card-negative {
        background: linear-gradient(135deg, #2ed573, #1abc9c);
        color: white;
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(46,213,115,0.4);
    }
    section[data-testid="stSidebar"] {
        background-color: #fff0f0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="big-title">❤️ การทำนายข้อมูลโรคหัวใจด้วยเทคนิค K-Nearest Neighbor ❤️</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Heart Disease Prediction using Machine Learning (KNN Algorithm)</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.image("./img/heart1.jpg", caption="เป็นโรคหัวใจ", use_container_width=True)
with col2:
    st.image("./img/heart2.jpg", caption="ไม่เป็นโรคหัวใจ", use_container_width=True)

st.markdown("---")

# ============================================================
# LOAD DATA
# ============================================================
dt = pd.read_csv("./data/Heart3.csv")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ เกี่ยวกับโมเดล")
    st.write("โมเดลนี้ใช้อัลกอริทึม **K-Nearest Neighbor (KNN)** ในการทำนายว่าผู้ป่วยมีความเสี่ยงเป็นโรคหัวใจหรือไม่ จากข้อมูลสุขภาพ 11 ตัวแปร")

    k_value = st.slider("เลือกค่า K (จำนวนเพื่อนบ้าน)", min_value=1, max_value=15, value=3, step=1)

    X_all = dt.drop('HeartDisease', axis=1)
    y_all = dt.HeartDisease
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    eval_model = KNeighborsClassifier(n_neighbors=k_value)
    eval_model.fit(X_train, y_train)
    acc = accuracy_score(y_test, eval_model.predict(X_test))

    st.metric("🎯 ความแม่นยำของโมเดล (Test set)", f"{acc*100:.2f}%")
    st.caption(f"จำนวนข้อมูลทั้งหมด: {len(dt)} แถว")
    st.markdown("---")
    st.caption("จัดทำเพื่อการศึกษา 📘")

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["📊 ข้อมูล & สถิติ", "📈 การวิเคราะห์ข้อมูล", "🩺 ทำนายผล"])

# ---------------- TAB 1: DATA ----------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("จำนวนแถว", len(dt))
    c2.metric("จำนวนคอลัมน์", len(dt.columns))
    c3.metric("ผู้ป่วยโรคหัวใจ", int((dt['HeartDisease'] == 1).sum()))
    c4.metric("ผู้ไม่ป่วย", int((dt['HeartDisease'] == 0).sum()))

    st.subheader("🔎 ข้อมูลส่วนแรก 10 แถว")
    st.dataframe(dt.head(10), use_container_width=True)

    st.subheader("🔎 ข้อมูลส่วนสุดท้าย 10 แถว")
    st.dataframe(dt.tail(10), use_container_width=True)

    st.subheader("📈 สถิติพื้นฐานของข้อมูล")
    st.dataframe(dt.describe(), use_container_width=True)

# ---------------- TAB 2: VISUALIZATION ----------------
with tab2:
    st.subheader("📌 เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล")
    feature = st.selectbox("เลือกฟีเจอร์", dt.columns[:-1])

    colA, colB = st.columns(2)
    with colA:
        st.write(f"### 🎯 Boxplot: {feature}")
        fig, ax = plt.subplots()
        sns.boxplot(data=dt, x='HeartDisease', y=feature, ax=ax, palette=["#2ed573", "#ff4757"])
        st.pyplot(fig)

    with colB:
        st.write(f"### 📊 Histogram: {feature}")
        fig3, ax3 = plt.subplots()
        sns.histplot(data=dt, x=feature, hue='HeartDisease', kde=True, ax=ax3, palette=["#2ed573", "#ff4757"])
        st.pyplot(fig3)

    st.markdown("---")
    if st.checkbox("🌺 แสดง Pairplot (ใช้เวลาประมวลผลเล็กน้อย)"):
        with st.spinner("กำลังประมวลผล Pairplot..."):
            fig2 = sns.pairplot(dt, hue='HeartDisease', palette=["#2ed573", "#ff4757"])
            st.pyplot(fig2)

# ---------------- TAB 3: PREDICTION ----------------
with tab3:
    st.markdown("""
    <div style="background-color:#33beff;padding:15px;border-radius:15px;border-style:solid;border-color:black">
    <center><h4>🩺 กรอกข้อมูลโรคหัวใจสำหรับทำนาย</h4></center>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    columns_list = list(dt.drop('HeartDisease', axis=1).columns)
    labels_map = {
        "Age": "อายุ (ปี)",
        "Sex": "เพศ (0 = หญิง, 1 = ชาย)",
        "ChestPainType": "ประเภทอาการเจ็บหน้าอก (0-3)",
        "RestingBP": "ความดันโลหิตขณะพัก (mmHg)",
        "Cholesterol": "ระดับคอเลสเตอรอล (mg/dl)",
        "FastingBS": "น้ำตาลในเลือดตอนอดอาหาร > 120 (0 = ไม่, 1 = ใช่)",
        "RestingECG": "ผลคลื่นไฟฟ้าหัวใจขณะพัก (0-2)",
        "MaxHR": "อัตราการเต้นหัวใจสูงสุด",
        "ExerciseAngina": "เจ็บหน้าอกขณะออกกำลังกาย (0 = ไม่, 1 = ใช่)",
        "Oldpeak": "ค่า ST depression (Oldpeak)",
        "ST_Slope": "ความชันของ ST segment (0-2)",
    }

    input_values = []
    n_cols = 3
    cols = st.columns(n_cols)
    for i, col_name in enumerate(columns_list):
        label = labels_map.get(col_name, col_name)
        with cols[i % n_cols]:
            val = st.number_input(f"{label}", key=f"input_{col_name}", value=0.0)
            input_values.append(val)

    st.write("")
    predict_clicked = st.button("🔍 ทำนายผล")

    if predict_clicked:
        with st.spinner("กำลังวิเคราะห์ข้อมูล..."):
            time.sleep(0.6)
            X = dt.drop('HeartDisease', axis=1)
            y = dt.HeartDisease

            Knn_model = KNeighborsClassifier(n_neighbors=k_value)
            Knn_model.fit(X, y)

            x_input = np.array([input_values])
            out = Knn_model.predict(x_input)
            proba = Knn_model.predict_proba(x_input)[0]

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])

        if out[0] == 1:
            with res_col1:
                st.markdown(f"""
                <div class="result-card-positive">
                <h2>⚠️ เป็นโรคหัวใจ</h2>
                <p>ความมั่นใจของโมเดล: {proba[1]*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            with res_col2:
                st.image("./img/heart1.jpg", use_container_width=True)
            st.warning("⚠️ ควรปรึกษาแพทย์เพื่อตรวจวินิจฉัยเพิ่มเติม")
        else:
            with res_col1:
                st.markdown(f"""
                <div class="result-card-negative">
                <h2>✅ ไม่เป็นโรคหัวใจ</h2>
                <p>ความมั่นใจของโมเดล: {proba[0]*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            with res_col2:
                st.image("./img/heart2.jpg", use_container_width=True)
            st.balloons()
            st.success("✅ ผลการทำนายอยู่ในเกณฑ์ปกติ")
    else:
        st.info("👆 กรอกข้อมูลด้านบนแล้วกดปุ่ม 'ทำนายผล' เพื่อดูผลลัพธ์")