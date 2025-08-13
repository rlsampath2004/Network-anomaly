# 🛰️ Network Anomaly Detection using Machine Learning (KDD Cup 1999)

A **machine learning-based cybersecurity system** designed to detect anomalous network traffic patterns using the **KDD Cup 1999 dataset**.  
The goal is to improve **network intrusion detection** by classifying traffic into normal or malicious categories with high accuracy.

---

## 📌 Overview
Network security is a critical challenge in the digital era, with cyberattacks becoming increasingly sophisticated.  
This project leverages **supervised machine learning** techniques to identify various network threats, such as:
- Denial-of-Service (DoS)
- Distributed Denial-of-Service (DDoS)
- Probing attacks
- Data exfiltration

By preprocessing the dataset, selecting the most relevant features, and training multiple models, this project aims to deliver a robust anomaly detection pipeline.

---

## 📂 Dataset
- **Name:** KDD Cup 2019 Dataset  
- **Source:** [KDD Cup Official Website](https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data)  
- **Description:** Labeled network traffic records containing both normal and attack samples.  
- **Features:** Includes protocol type, service, connection duration, packet statistics, and more.

---

## ✨ Key Features
- Comprehensive **data preprocessing** and cleaning
- **Feature engineering** for improved detection accuracy
- Implementation of multiple ML models:
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost
- **Evaluation metrics:** Accuracy, Precision, Recall, F1-Score
- **Visualization:** Confusion matrix, feature importance, and accuracy plots

---

## 🛠 Tech Stack
- **Programming Language:** Python 3.x  
- **Libraries & Frameworks:**
  - Pandas, NumPy — Data manipulation
  - Scikit-learn — Machine learning models & preprocessing
  - Matplotlib, Seaborn — Data visualization
  - XGBoost — Gradient boosting model
- **Tools:** Jupyter Notebook, Git

---

## 🔄 Workflow
```mermaid
graph TD
A[Data Collection] --> B[Data Preprocessing]
B --> C[Feature Engineering]
C --> D[Model Training]
D --> E[Model Evaluation]
E --> F[Anomaly Detection]
