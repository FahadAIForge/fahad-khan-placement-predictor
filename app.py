"""Campus Placement Predictor — interactive Streamlit demo.

Enter a student's CGPA and IQ score to predict placement outcome
with the trained logistic regression model.
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="centered")

with open("model.pkl", "rb") as f:
    clf = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

df = pd.read_csv("placement_data.csv")

st.title("🎓 Campus Placement Predictor")
st.write(
    "An end-to-end ML model (logistic regression, 85% test accuracy) that predicts "
    "whether a student gets placed, based on CGPA and IQ score."
)

st.subheader("Try it")
cgpa = st.slider("CGPA", 4.0, 10.0, 7.0, 0.1)
iq = st.slider("IQ score", 80, 160, 120, 1)

X_new = scaler.transform(np.array([[cgpa, iq]]))
proba = clf.predict_proba(X_new)[0][1]
pred = clf.predict(X_new)[0]

st.progress(float(proba), text=f"Placement probability: {proba:.0%}")
if pred == 1:
    st.success("✅ Likely to be PLACED")
else:
    st.error("❌ Likely NOT placed")

with st.expander("How the model sees the data"):
    fig, ax = plt.subplots(figsize=(7, 5))
    placed = df[df.placement == 1]
    not_placed = df[df.placement == 0]
    ax.scatter(not_placed.cgpa, not_placed.iq, c="#ef4444", label="Not placed", alpha=0.7)
    ax.scatter(placed.cgpa, placed.iq, c="#16a34a", label="Placed", alpha=0.7)
    # decision boundary
    xx, yy = np.meshgrid(np.linspace(4, 10, 200), np.linspace(80, 160, 200))
    grid = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
    zz = clf.predict(grid).reshape(xx.shape)
    ax.contour(xx, yy, zz, levels=[0.5], colors="#0f172a", linewidths=2)
    ax.scatter([cgpa], [iq], c="#0f172a", s=160, marker="*", label="Your input",
               edgecolors="white", linewidths=1.5, zorder=5)
    ax.set_xlabel("CGPA")
    ax.set_ylabel("IQ score")
    ax.legend()
    st.pyplot(fig)

st.caption("Model: Logistic Regression · Features scaled with StandardScaler · Trained on 100 student records")
