import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# =====================================================
# TITLE
# =====================================================

st.title("🎓 Student Performance Prediction System")

st.write(
    "Predict student final performance using Machine Learning."
)

st.divider()

# =====================================================
# CHECK DATASET
# =====================================================

DATA_FILE = "student_performance.csv"

if not os.path.exists(DATA_FILE):
    st.error(
        "❌ student_performance.csv not found. "
        "Please place the CSV file in the same folder as app.py."
    )
    st.stop()

# Load dataset
df = pd.read_csv(DATA_FILE)

st.success("✅ Dataset loaded successfully!")

# =====================================================
# CHECK MODEL
# =====================================================

MODEL_FILE = "model/student_model.pkl"
SCALER_FILE = "model/scaler.pkl"

if not os.path.exists(MODEL_FILE):
    st.error(
        "❌ student_model.pkl not found. "
        "Please run train_model.py first."
    )
    st.stop()

if not os.path.exists(SCALER_FILE):
    st.error(
        "❌ scaler.pkl not found. "
        "Please run train_model.py first."
    )
    st.stop()

# Load model
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

st.success("✅ Machine Learning model loaded!")

# =====================================================
# DATASET
# =====================================================

st.header("📊 Dataset")

st.dataframe(
    df,
    use_container_width=True
)

st.write(
    f"Number of students: **{len(df)}**"
)

# =====================================================
# SIDEBAR INPUT
# =====================================================

st.sidebar.header("👨‍🎓 Student Information")

hours_studied = st.sidebar.number_input(
    "Hours Studied Per Day",
    min_value=0.0,
    max_value=15.0,
    value=5.0
)

attendance = st.sidebar.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

previous_score = st.sidebar.number_input(
    "Previous Score",
    min_value=0.0,
    max_value=100.0,
    value=60.0
)

assignments = st.sidebar.number_input(
    "Assignment Completion (%)",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

participation = st.sidebar.number_input(
    "Class Participation (%)",
    min_value=0.0,
    max_value=100.0,
    value=60.0
)

sleep_hours = st.sidebar.number_input(
    "Sleep Hours",
    min_value=0.0,
    max_value=12.0,
    value=7.0
)

# =====================================================
# PREDICTION BUTTON
# =====================================================

st.header("🔮 Student Performance Prediction")

if st.button(
    "Predict Performance",
    type="primary",
    use_container_width=True
):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Hours_Studied": [hours_studied],
        "Attendance": [attendance],
        "Previous_Score": [previous_score],
        "Assignments": [assignments],
        "Participation": [participation],
        "Sleep_Hours": [sleep_hours]
    })

    # Convert to NumPy
    input_array = np.asarray(input_data)

    # Scale input
    input_scaled = scaler.transform(input_array)

    # Predict
    prediction = model.predict(input_scaled)

    score = float(prediction[0])

    # Keep score between 0 and 100
    score = np.clip(score, 0, 100)

    # =================================================
    # PERFORMANCE CATEGORY
    # =================================================

    if score >= 90:
        category = "Excellent 🟢"

    elif score >= 75:
        category = "Very Good 🟢"

    elif score >= 60:
        category = "Good 🟡"

    elif score >= 50:
        category = "Average 🟠"

    else:
        category = "Needs Improvement 🔴"

    # =================================================
    # DISPLAY RESULT
    # =================================================

    st.success("Prediction completed successfully!")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Final Score",
            f"{score:.2f}/100"
        )

    with col2:
        st.metric(
            "Performance",
            category
        )

    st.divider()

    # =================================================
    # STUDENT FACTORS
    # =================================================

    st.subheader("📈 Student Performance Factors")

    chart_data = pd.DataFrame({
        "Factor": [
            "Attendance",
            "Previous Score",
            "Assignments",
            "Participation"
        ],
        "Value": [
            attendance,
            previous_score,
            assignments,
            participation
        ]
    })

    fig = px.bar(
        chart_data,
        x="Factor",
        y="Value",
        color="Value",
        range_y=[0, 100],
        title="Student Performance Factors"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =================================================
    # RECOMMENDATIONS
    # =================================================

    st.subheader("💡 Student Recommendation")

    if score >= 90:

        st.success(
            "Excellent performance! Keep maintaining your "
            "study habits and continue participating actively."
        )

    elif score >= 75:

        st.info(
            "Very good performance. Increase study consistency "
            "and focus on completing all assignments."
        )

    elif score >= 60:

        st.warning(
            "Good performance, but there is room for improvement. "
            "Try increasing study hours and attendance."
        )

    elif score >= 50:

        st.warning(
            "Your performance is average. Focus on attendance, "
            "assignments and regular study."
        )

    else:

        st.error(
            "Performance needs improvement. Increase study time, "
            "attendance and assignment completion."
        )

# =====================================================
# DATA VISUALIZATION
# =====================================================

st.divider()

st.header("📊 Data Analysis")

tab1, tab2 = st.tabs([
    "Score Distribution",
    "Feature Relationship"
])

# =====================================================
# TAB 1
# =====================================================

with tab1:

    fig1 = px.histogram(
        df,
        x="Final_Score",
        nbins=10,
        title="Final Score Distribution",
        color_discrete_sequence=["#3498DB"]
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

# =====================================================
# TAB 2
# =====================================================

with tab2:

    feature = st.selectbox(
        "Select Feature",
        [
            "Hours_Studied",
            "Attendance",
            "Previous_Score",
            "Assignments",
            "Participation",
            "Sleep_Hours"
        ]
    )

    fig2 = px.scatter(
        df,
        x=feature,
        y="Final_Score",
        title=f"{feature} vs Final Score"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Student Performance Prediction System | "
    "Python | Pandas | NumPy | Scikit-learn | Plotly | Streamlit"
)
