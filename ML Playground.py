import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# -------------------- UI --------------------
st.set_page_config(page_title="ML Playground", layout="wide")
st.title("🧠 Interactive Machine Learning Playground")
st.write("Experiment with datasets, models, and hyperparameters in real time.")

# -------------------- Sidebar --------------------
st.sidebar.header("⚙️ Configuration")

dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ("Iris", "Breast Cancer")
)

model_name = st.sidebar.selectbox(
    "Select Model",
    ("Logistic Regression", "Random Forest", "SVM")
)

test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)

# -------------------- Load Dataset --------------------
def load_dataset(name):
    if name == "Iris":
        data = load_iris()
    else:
        data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y

X, y = load_dataset(dataset_name)

st.subheader("📊 Dataset Preview")
st.dataframe(X.head())

# -------------------- Visualization --------------------
st.subheader("📈 Feature Visualization")
feature_x = st.selectbox("X-axis Feature", X.columns)
feature_y = st.selectbox("Y-axis Feature", X.columns)

fig, ax = plt.subplots()
sns.scatterplot(
    x=X[feature_x],
    y=X[feature_y],
    hue=y,
    palette="deep",
    ax=ax
)
ax.set_title("Feature Relationship")
st.pyplot(fig)

# -------------------- Train/Test Split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------- Model Selection --------------------
def get_model(name):
    if name == "Logistic Regression":
        C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0)
        model = LogisticRegression(C=C, max_iter=1000)

    elif name == "Random Forest":
        n_estimators = st.sidebar.slider("Number of Trees", 50, 300, 100)
        max_depth = st.sidebar.slider("Max Depth", 2, 20, 5)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

    else:
        C = st.sidebar.slider("C", 0.1, 10.0, 1.0)
        kernel = st.sidebar.selectbox("Kernel", ("linear", "rbf"))
        model = SVC(C=C, kernel=kernel)

    return model

model = get_model(model_name)

# -------------------- Training --------------------
model.fit(X_train, y_train)
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

# -------------------- Results --------------------
st.subheader("✅ Model Performance")
st.metric("Accuracy", f"{accuracy * 100:.2f}%")

st.success("Model trained successfully!")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("🚀 Built for learning & experimentation | Classic Machine Learning")

print("Streamlit ML Playground is running.")