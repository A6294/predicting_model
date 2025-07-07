import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df = pd.read_excel(r"C:\Users\ariji\Desktop\student_suicide_dataset.xlsx")

df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Social_Support'] = df['Social_Support'].map({'No': 0, 'Yes': 1})

df.dropna(inplace=True)

features = ['Age', 'CGPA', 'Screen_Time_Hours', 'PHQ9_Score',
            'GAD7_Score', 'Stress_Level', 'Sleep_Hours', 'Social_Support']
X = df[features]
y = df['Predicted_Suicide_Risk']

model = RandomForestRegressor()
model.fit(X, y)

st.set_page_config(page_title="University of Burdwan", layout="centered")
st.title("🎓 University of Burdwan")
st.subheader("💡 Predict Suicide Risk for a New Student")

with st.form("new_student_form"):
    age = st.number_input("Age", min_value=15, max_value=40, value=20)
    gender = st.selectbox("Gender", ['Male', 'Female'])
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0)
    screen_time = st.number_input("Screen Time (Hours)", min_value=0.0, max_value=24.0, value=5.0)
    phq9 = st.number_input("PHQ-9 Score", min_value=0, max_value=27, value=10)
    gad7 = st.number_input("GAD-7 Score", min_value=0, max_value=21, value=8)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    sleep = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=6.0)
    support = st.selectbox("Social Support", ['Yes', 'No'])

    submit = st.form_submit_button("🔮 Predict Risk")

if submit:
    gender_num = 0 if gender == 'Male' else 1
    support_num = 1 if support == 'Yes' else 0

    input_data = pd.DataFrame([[age, cgpa, screen_time, phq9, gad7, stress, sleep, support_num]],
                              columns=features)

    predicted_risk = model.predict(input_data)[0]

    if predicted_risk >= 40:
        level = "High"
        color = "red"
    elif predicted_risk >= 20:
        level = "Moderate"
        color = "orange"
    else:
        level = "Low"
        color = "green"

    st.markdown("### 🚨 Predicted Suicide Risk")
    st.markdown(
        f"<div style='padding:20px; background-color:{color}; color:white; font-size:20px; border-radius:10px;'>"
        f"<strong>Risk Score: {predicted_risk:.2f}</strong><br>"
        f"Risk Level: {level}"
        f"</div>",
        unsafe_allow_html=True
    )

    st.info("📌 Note: This prediction is based on the model trained on sample data. It is not a clinical diagnosis.")
