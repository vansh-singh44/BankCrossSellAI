🏦 BankCrossSellAI

VITBAIH Community Project 2026 – Track C Submission

An end-to-end AI-powered banking recommendation system that predicts whether a customer is likely to subscribe to a term deposit product. The project combines Machine Learning, FastAPI, and Streamlit to simulate a real-world banking cross-sell recommendation workflow.

---

🚀 Features

- 📊 Exploratory Data Analysis (EDA)
- 🤖 CatBoost-based Subscription Prediction
- 📈 Logistic Regression Baseline Comparison
- ⚡ FastAPI Prediction Service
- 🎨 Interactive Streamlit Dashboard
- 📝 Customer-level Explanations
- 💾 Saved Models and Prediction Artifacts
- 📚 Jupyter Notebook for Reproducibility

---

📂 Project Structure

BankCrossSellAI/

├── api/
│   └── app.py

├── dashboard/
│   └── streamlit_app.py

├── data/
│   └── bank-additional-full.csv

├── models/
│   ├── model.pkl
│   ├── lr_model.pkl
│   └── features.pkl

├── notebooks/
│   └── eda.ipynb

├── outputs/
│   └── predictions.csv

├── plots/

├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py

├── README.md
├── EXPLANATION.md
├── requirements.txt
└── .gitignore

---

📊 Dataset

Dataset Used:

UCI Bank Marketing Dataset

Target Variable:

y

yes → 1
no → 0

Records:

~41,188 customers

Features:

Demographics

Financial indicators

Campaign information

Economic indicators

---

🧠 Machine Learning Models

Baseline Model

- Logistic Regression

Main Model

- CatBoost Classifier

Evaluation Metrics:

Accuracy

Precision

Recall

F1 Score

ROC AUC

---

⚡ FastAPI Endpoints

Health Check

GET /health

Response

{
"status":"ok",
"model":"CatBoost"
}

---

Prediction

POST /predict

Sample Input

{
"age":45,
"job":"management",
"housing":"no",
"loan":"no"
}

Sample Response

{
"will_subscribe":true,
"probability":0.87,
"top_factors":[
"Long Call Duration",
"Previous Campaign Success"
]
}

---

🎨 Streamlit Dashboard

Features

Interactive filters

Probability cards

Customer explorer

Visual analytics

Prediction explanations

Run locally

streamlit run dashboard/streamlit_app.py

---

🚀 Running the Project

Create Environment

python -m venv .venv

Activate

Windows

.\.venv\Scripts\activate

Install Dependencies

pip install -r requirements.txt

Run API

uvicorn api.app:app --reload

Run Dashboard

streamlit run dashboard/streamlit_app.py

---

📈 Outputs

Predictions

plots/

Model artifacts

API

Dashboard

Notebook

Explanation file

---

👨‍💻 Author

Vansh Singh

VIT Bhopal University

Computer Science and Engineering

2025 Batch

---

📌 Submission

VITBAIH Community Project 2026

Track C – System Builder

CatBoost + FastAPI + Streamlit
