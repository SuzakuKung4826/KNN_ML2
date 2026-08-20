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
    page_title="Titanic Survival Prediction | KNN",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS  (สีตัวอักษรถูกกำหนดชัดเจนทุกจุด กันปัญหา dark/light theme ชนกัน)
# ============================================================
st.markdown("""
<style>
    /* พื้นหลังหลักของแอป */
    .stApp {
        background-color: #f0f6fc !important;
    }
    .main {
        background-color: #f0f6fc !important;
    }

    /* หัวข้อใหญ่ */
    .big-title {
        font-size: 2.2rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #6c757d !important;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* ทำให้ตัวอักษรทั่วไปในหน้าอ่านง่าย (บังคับสีเข้มบนพื้นอ่อน) */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div,
    .stMarkdown, h1, h2, h3, h4, h5, h6 {
        color: #262730 !important;
    }

    /* การ์ด Metric */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #eee;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetricValue"] {
        color: #1e3a8a !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #262730 !important;
    }

    /* ปุ่ม */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        color: #ffffff !important;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 0;
        font-size: 1.1rem;
        transition: 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(30,58,138,0.4);
    }
    .stButton>button p {
        color: #ffffff !important;
    }

    /* การ์ดผลลัพธ์ - ตัวอักษรขาวชัดเจนบนพื้นสีเข้ม */
    .result-card-positive, .result-card-positive h2, .result-card-positive p {
        color: #ffffff !important;
    }
    .result-card-positive {
        background: linear-gradient(135deg, #2ed573, #1abc9c);
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(46,213,115,0.4);
    }
    .result-card-negative, .result-card-negative h2, .result-card-negative p {
        color: #ffffff !important;
    }
    .result-card-negative {
        background: linear-gradient(135deg, #576574, #2f3542);
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(47,53,66,0.4);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #eaf2fb !important;
    }
    section[data-testid="stSidebar"] * {
        color: #262730 !important;
    }

    /* กล่อง info-box (ตัวอักษรขาวบนพื้นฟ้า) */
    .info-box h4, .info-box h5 {
        color: #ffffff !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: #262730 !important;
        font-weight: 600;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #1e3a8a !important;
    }

    /* ตาราง dataframe ให้พื้นขาวชัดเจน */
    div[data-testid="stDataFrame"] {
        background-color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="big-title">🚢 การทำนายการรอดชีวิตบนเรือไททานิคด้วยเทคนิค K-Nearest Neighbor 🚢</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Titanic Survival Prediction using Machine Learning (KNN Algorithm)</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div style="background:#1abc9c;padding:40px;border-radius:15px;text-align:center;">
        <span style="font-size:3.5rem;">🛟</span>
        <h3 style="color:#ffffff !important;margin-top:10px;">รอดชีวิต</h3>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div style="background:#2f3542;padding:40px;border-radius:15px;text-align:center;">
        <span style="font-size:3.5rem;">🌊</span>
        <h3 style="color:#ffffff !important;margin-top:10px;">ไม่รอดชีวิต</h3>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# LOAD DATA
# ============================================================
dt = pd.read_csv("./data/Titanic.csv")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ เกี่ยวกับโมเดล")
    st.write("โมเดลนี้ใช้อัลกอริทึม **K-Nearest Neighbor (KNN)** ในการทำนายว่าผู้โดยสารรอดชีวิตจากเหตุการณ์เรือไททานิคจมหรือไม่ จากข้อมูลผู้โดยสาร 6 ตัวแปร")

    k_value = st.slider("เลือกค่า K (จำนวนเพื่อนบ้าน)", min_value=1, max_value=15, value=5, step=1)

    X_all = dt.drop('Survived', axis=1)
    y_all = dt.Survived
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    eval_model = KNeighborsClassifier(n_neighbors=k_value)
    eval_model.fit(X_train, y_train)
    acc = accuracy_score(y_test, eval_model.predict(X_test))

    st.metric("🎯 ความแม่นยำของโมเดล (Test set)", f"{acc*100:.2f}%")
    st.caption(f"จำนวนข้อมูลทั้งหมด: {len(dt)} แถว")
    st.markdown("---")
    st.caption("จัดทำเพื่อการศึกษา 📘 (Dataset: Titanic - Kaggle/Data Science Dojo)")

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["📊 ข้อมูล & สถิติ", "📈 การวิเคราะห์ข้อมูล", "🚢 ทำนายผล"])

# ---------------- TAB 1: DATA ----------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("จำนวนแถว", len(dt))
    c2.metric("จำนวนคอลัมน์", len(dt.columns))
    c3.metric("ผู้โดยสารที่รอดชีวิต", int((dt['Survived'] == 1).sum()))
    c4.metric("ผู้โดยสารที่ไม่รอด", int((dt['Survived'] == 0).sum()))

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
        sns.boxplot(data=dt, x='Survived', y=feature, ax=ax, palette=["#2f3542", "#1abc9c"])
        st.pyplot(fig)

    with colB:
        st.write(f"### 📊 Histogram: {feature}")
        fig3, ax3 = plt.subplots()
        sns.histplot(data=dt, x=feature, hue='Survived', kde=True, ax=ax3, palette=["#2f3542", "#1abc9c"])
        st.pyplot(fig3)

    st.markdown("---")
    if st.checkbox("🌺 แสดง Pairplot (ใช้เวลาประมวลผลเล็กน้อย)"):
        with st.spinner("กำลังประมวลผล Pairplot..."):
            fig2 = sns.pairplot(dt, hue='Survived', palette=["#2f3542", "#1abc9c"])
            st.pyplot(fig2)

# ---------------- TAB 3: PREDICTION ----------------
with tab3:
    st.markdown("""
    <div class="info-box" style="background-color:#3498db;padding:15px;border-radius:15px;border-style:solid;border-color:black">
    <center><h4>🚢 กรอกข้อมูลผู้โดยสารสำหรับทำนายการรอดชีวิต</h4></center>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    columns_list = list(dt.drop('Survived', axis=1).columns)
    labels_map = {
        "Age": "อายุ (ปี)",
        "Sex": "เพศ (0 = หญิง, 1 = ชาย)",
        "Pclass": "ชั้นโดยสาร (1 = ชั้น 1, 2 = ชั้น 2, 3 = ชั้น 3)",
        "SibSp": "จำนวนพี่น้อง/คู่สมรสที่ติดเรือมาด้วย",
        "Parch": "จำนวนพ่อแม่/ลูกที่ติดเรือมาด้วย",
        "Fare": "ค่าโดยสาร (ปอนด์)",
    }

    default_values = {
        "Age": 29.0,
        "Sex": 1.0,
        "Pclass": 3.0,
        "SibSp": 0.0,
        "Parch": 0.0,
        "Fare": 32.0,
    }

    input_values = []
    n_cols = 3
    cols = st.columns(n_cols)
    for i, col_name in enumerate(columns_list):
        label = labels_map.get(col_name, col_name)
        with cols[i % n_cols]:
            val = st.number_input(f"{label}", key=f"input_{col_name}", value=default_values.get(col_name, 0.0))
            input_values.append(val)

    st.write("")
    predict_clicked = st.button("🔍 ทำนายผล")

    if predict_clicked:
        with st.spinner("กำลังวิเคราะห์ข้อมูล..."):
            time.sleep(0.6)
            X = dt.drop('Survived', axis=1)
            y = dt.Survived

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
                <h2>🛟 รอดชีวิต</h2>
                <p>ความมั่นใจของโมเดล: {proba[1]*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            with res_col2:
                st.markdown("""
                <div style="background:#1abc9c;padding:40px;border-radius:15px;text-align:center;height:100%;">
                    <span style="font-size:3.5rem;">🛟</span>
                </div>
                """, unsafe_allow_html=True)
            st.balloons()
            st.success("✅ ผู้โดยสารรายนี้มีแนวโน้มรอดชีวิตสูง")
        else:
            with res_col1:
                st.markdown(f"""
                <div class="result-card-negative">
                <h2>🌊 ไม่รอดชีวิต</h2>
                <p>ความมั่นใจของโมเดล: {proba[0]*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            with res_col2:
                st.markdown("""
                <div style="background:#2f3542;padding:40px;border-radius:15px;text-align:center;height:100%;">
                    <span style="font-size:3.5rem;">🌊</span>
                </div>
                """, unsafe_allow_html=True)
            st.warning("⚠️ ผู้โดยสารรายนี้มีแนวโน้มไม่รอดชีวิตตามข้อมูลที่ป้อน")
    else:
        st.info("👆 กรอกข้อมูลด้านบนแล้วกดปุ่ม 'ทำนายผล' เพื่อดูผลลัพธ์")
