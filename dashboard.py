import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(page_title="Dynamic ML Dashboard", layout="centered")
st.title("🤖 Dynamic Retraining ML Dashboard")

# -------------------------------------------------
# File Paths
# -------------------------------------------------
DATA_FILE = os.path.join("..", "data", "smart_logistics_dataset.csv")

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(DATA_FILE):
        st.error(f"❌ Dataset not found: {DATA_FILE}")
        return None
    df = pd.read_csv(DATA_FILE)
    return df

df = load_data()
if df is None:
    st.stop()

st.success(f"✅ Dataset Loaded Successfully! Shape: {df.shape}")
st.write(df.head())

# -------------------------------------------------
# User Feature Selection
# -------------------------------------------------
st.subheader("🧩 Select Features for Retraining")
all_columns = df.columns.tolist()
target_col = st.selectbox("🎯 Select Target Column", all_columns, index=all_columns.index("Logistics_Delay") if "Logistics_Delay" in all_columns else 0)

feature_choices = [col for col in all_columns if col != target_col]
selected_features = st.multiselect("🧠 Select up to 4 features to train model:", feature_choices, default=feature_choices[:4])

if len(selected_features) == 0:
    st.warning("⚠️ Please select at least one feature to train the model.")
    st.stop()

if len(selected_features) > 4:
    st.warning("⚠️ Please select up to 4 features only.")
    st.stop()

# -------------------------------------------------
# Dynamic Retraining
# -------------------------------------------------
st.subheader("⚙️ Retraining Model on Selected Features...")

X = df[selected_features].copy()
y = df[target_col].copy()

# Encode categorical columns automatically
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

if y.dtype == 'object':
    le_y = LabelEncoder()
    y = le_y.fit_transform(y.astype(str))

# Split & Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
acc = accuracy_score(y_test, rf_model.predict(X_test))
st.success(f"✅ Model retrained successfully! Accuracy: {acc*100:.2f}%")

# -------------------------------------------------
# User Inputs for Prediction
# -------------------------------------------------
st.subheader("🔮 Predict Using Your Trained Model")

user_inputs = {}
for col in selected_features:
    if df[col].dtype == 'object':
        options = list(df[col].unique())
        user_inputs[col] = st.selectbox(f"{col}", options)
    else:
        user_inputs[col] = st.number_input(f"{col}", value=float(df[col].mean()))

# Prepare user input for prediction
input_df = pd.DataFrame([user_inputs])

# Encode input just like training data
for col in input_df.columns:
    if input_df[col].dtype == 'object':
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        input_df[col] = le.transform(input_df[col].astype(str))

# Predict
if st.button("🚀 Predict"):
    prediction = rf_model.predict(input_df)[0]
    st.success(f"🎯 Predicted Output: {prediction}")

# -------------------------------------------------
# Insights Tab
# -------------------------------------------------
st.subheader("📊 Insights")

# Feature Importance
try:
    importances = rf_model.feature_importances_
    imp_df = pd.DataFrame({"Feature": selected_features, "Importance": importances})
    fig, ax = plt.subplots()
    sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax, palette="viridis")
    ax.set_title("Feature Importances (Dynamic Model)")
    st.pyplot(fig)
except Exception as e:
    st.error(f"Could not plot feature importance: {e}")
