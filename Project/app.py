import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)


model = joblib.load("KNN_model.pkl")
scaler = joblib.load("scaler_model.pkl")
columns = joblib.load("columns.pkl")


st.markdown("""
<style>

.main{
    background-color:#f8f9fa;
}

.title{
    text-align:center;
    color:#E63946;
    font-size:40px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:#555;
    font-size:18px;
}

.stButton>button{
    width:100%;
    background-color:#E63946;
    color:white;
    border-radius:10px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background-color:#C1121F;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>❤️ Heart Disease Prediction</div>", unsafe_allow_html=True)

st.markdown(
"<div class='subtitle'>Fill in the patient's medical information below to predict the risk of heart disease.</div>",
unsafe_allow_html=True
)

st.write("")

with st.sidebar:
    st.header("ℹ️ About")
    st.info(
        """
        This application predicts the likelihood of heart disease using a trained K-Nearest Neighbors (KNN) machine learning model.

        **Model:** KNN

        **Features Used**
        - Age
        - Blood Pressure
        - Cholesterol
        - ECG
        - Heart Rate
        - Exercise Angina
        - ST Slope
        """
    )

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Details")

    age = st.slider("Age", 18, 100, 40)

    sex = st.selectbox(
        "Gender",
        ["M", "F"]
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "ASY", "TA"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        80,
        200,
        120
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        100,
        600,
        200
    )

with col2:
    st.subheader("🩺 Clinical Details")

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar >120 mg/dL",
        [0,1]
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal","ST","LVH"]
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["Y","N"]
    )

    oldpeak = st.slider(
        "Oldpeak (ST Depression)",
        0.0,
        6.0,
        1.0
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up","Flat","Down"]
    )

st.write("")


if st.button("🔍 Predict Heart Disease Risk"):

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    st.divider()

    st.subheader("📊 Prediction Result")

    if prediction == 1:

        st.error("⚠️ High Risk of Heart Disease")

        st.progress(90)

        st.markdown("""
### Recommendations

- ❤️ Consult a cardiologist.
- 🥗 Maintain a healthy diet.
- 🚶 Exercise regularly.
- 🚭 Avoid smoking.
- 🩺 Monitor blood pressure and cholesterol.
        """)

    else:

        st.success("✅ Low Risk of Heart Disease")

        st.progress(20)

        st.markdown("""
### Recommendations

- 🥗 Continue a healthy lifestyle.
- 🚶 Exercise regularly.
- ❤️ Maintain a balanced diet.
- 🩺 Schedule routine health checkups.
        """)

st.markdown("---")
st.caption("Developed using Streamlit | Machine Learning Model: KNN")