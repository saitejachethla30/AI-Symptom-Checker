import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

# Page setup
st.set_page_config(page_title="AI Symptom Checker", layout="centered")

# Title
st.title("🩺 AI Symptom Checker")
st.write("Select your symptoms and get predicted disease")

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Training.csv")
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        return df
    except:
        return pd.DataFrame()

df = load_data()

# Check dataset
if df.empty:
    st.error("❌ Dataset not found! Please check Training.csv")
    st.stop()

# Ensure correct column
if "prognosis" not in df.columns:
    st.error("❌ 'prognosis' column missing in dataset")
    st.stop()

# Split data
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Symptoms list
symptoms = X.columns.tolist()

# UI
selected_symptoms = st.multiselect("Choose Symptoms:", symptoms)

# Input vector
input_data = [0] * len(symptoms)
for symptom in selected_symptoms:
    input_data[symptoms.index(symptom)] = 1

# 📌 Disease Info (ALL LOWERCASE KEYS)
disease_info = {
    "hepatitis a": {
        "cause": "Viral infection due to contaminated food or water",
        "precautions": [
            "Drink clean and safe water",
            "Maintain personal hygiene",
            "Avoid outside food",
            "Take proper rest"
        ]
    },
    "malaria": {
        "cause": "Mosquito-borne disease caused by parasites",
        "precautions": [
            "Use mosquito nets",
            "Apply repellents",
            "Avoid stagnant water",
            "Keep surroundings clean"
        ]
    },
    "dengue": {
        "cause": "Viral infection transmitted by mosquitoes",
        "precautions": [
            "Wear full sleeve clothes",
            "Use repellents",
            "Avoid water accumulation",
            "Keep environment clean"
        ]
    },
    "typhoid": {
        "cause": "Bacterial infection from contaminated food and water",
        "precautions": [
            "Drink boiled water",
            "Eat hygienic food",
            "Wash hands regularly",
            "Avoid street food"
        ]
    }
}

# 🔍 Prediction
if st.button("🔍 Predict Disease"):
    if len(selected_symptoms) == 0:
        st.warning("⚠️ Please select at least one symptom")
    else:
        prediction = model.predict([input_data])[0]

        st.write(prediction)

        st.success(f"✅ Possible Disease: {prediction}")

        # 🔥 NORMALIZE TEXT
        key = str(prediction).lower().strip()
        key = key.replace("_", " ").replace("-", " ")

        # 🔥 MATCH LOGIC (VERY STRONG)
        matched = None
        for disease in disease_info.keys():
            if disease in key:
                matched = disease
                break

        # Show results
        if matched:
            st.subheader("🦠 Cause")
            st.write(disease_info[matched]["cause"])

            st.subheader("💊 Precautions")
            for p in disease_info[matched]["precautions"]:
                st.write("•", p)
        else:
            st.info("No additional information available for this disease.")

        st.warning("⚠️ This is only a basic prediction. Please consult a doctor.")

# Footer
st.markdown("---")
st.caption("Mini Project | SDG 3: Good Health & Well-being")