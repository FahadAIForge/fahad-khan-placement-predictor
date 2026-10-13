# Fahad Khan

## Fahad Khan

An **end-to-end machine learning project** that predicts whether a student will get
placed during campus recruitment, based on **CGPA** and **IQ score**.

## What it does

- Interactive **Streamlit web app** — move the sliders, get an instant placement
  prediction with probability and a visual decision boundary
- Full ML pipeline: data cleaning → EDA → feature scaling → train/test split →
  logistic regression → evaluation → serialized model

## Model performance

| Metric | Score |
|---|---|
| Test accuracy (20% holdout) | **85%** |
| 5-fold cross-validation | **91%** |

- Algorithm: Logistic Regression (`sklearn`)
- Features: CGPA, IQ (standardized with `StandardScaler`)
- Dataset: 100 student records (`placement_data.csv`)

## Run it locally

```bash
pip install -r requirements.txt
python train.py        # retrain the model
streamlit run app.py   # launch the web app
```

## Project structure

```
├── app.py              # Streamlit demo app
├── train.py            # training pipeline
├── placement_data.csv  # cleaned dataset
├── model.pkl           # trained model
├── scaler.pkl          # fitted scaler
└── requirements.txt
```

Built by [Fahad Khan](https://www.linkedin.com/in/fahad-khan-ai) — AI Automation Specialist.
