import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load and clean dataset
df = pd.read_csv("heart_failure.csv")
df.loc[df['Cholesterol'] == 0, 'Cholesterol'] = np.nan
df.loc[df['RestingBP'] == 0, 'RestingBP'] = np.nan
df['Cholesterol'] = df['Cholesterol'].fillna(df['Cholesterol'].median())
df['RestingBP'] = df['RestingBP'].fillna(df['RestingBP'].median())

# Feature Engineering
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 40, 60, 100], labels=['Young', 'Middle', 'Senior'])
df['BP_Zero'] = (df['RestingBP'] == 0).astype(int)
df['Chol_Zero'] = (df['Cholesterol'] == 0).astype(int)
df = pd.get_dummies(df, drop_first=True)

# Split and scale
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

st.title("Heart Disease Prediction App")

selected_model = st.selectbox("Select Model", list(models.keys()))
model = models[selected_model]
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

st.subheader(f"{selected_model} Performance")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
st.write(f"Precision: {precision_score(y_test, y_pred):.2f}")
st.write(f"Recall: {recall_score(y_test, y_pred):.2f}")
st.write(f"F1 Score: {f1_score(y_test, y_pred):.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)
