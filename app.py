import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Diabetes Risk Prediction", layout="wide")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Patient Input Variables")
    
    bmi = st.text_input("BMI", placeholder="e.g., 25.5")
    age_group = st.selectbox("Age Group", ["Select age group"])
    sex = st.selectbox("Sex", ["Select sex"])
    
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        high_bp = st.toggle("High Blood Pressure")
        phys_act = st.toggle("Physical Activity", value=True)
        veg_daily = st.toggle("Vegetables Daily", value=True)
    with t_col2:
        high_chol = st.toggle("High Cholesterol")
        fruits_daily = st.toggle("Fruits Daily", value=True)
        
    gen_health = st.selectbox("General Health (1-5)", ["Select health rating"])
    income = st.selectbox("Income Level", ["Select income level"])
    education = st.selectbox("Education Level", ["Select education level"])
    
    predict_btn = st.button("Predict Risk", type="primary", use_container_width=True)

with col2:
    st.subheader("Prediction Results")
    st.info("Enter patient data and click 'Predict Risk' to see results", icon="📊")
    
    st.divider()
    
    st.subheader("Feature Importance")
    st.caption("Major factors contributing to the prediction")
    
    data = pd.DataFrame({
        "Feature": ["BMI", "HighBP", "Age", "GenHlth", "HighChol", "PhysActivity", "Income"],
        "Importance": [30, 22, 18, 15, 9, 6, 3]
    })
    
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X("Importance", title=""),
        y=alt.Y("Feature", sort="-x", title=""),
        color=alt.Color("Feature", legend=None)
    ).properties(height=300)
    
    st.altair_chart(chart, use_container_width=True)
