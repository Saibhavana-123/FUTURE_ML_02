# 🎫 Support Ticket Classification & Prioritization System

## 📌 Overview

This project is a Machine Learning + NLP based system that automatically classifies customer support tickets into categories and assigns priority levels.

It helps organizations reduce manual effort, improve response time, and manage support operations efficiently.

---

## 🎯 Objective

* Classify support tickets into categories (e.g., Technical Issue, Billing, etc.)
* Assign priority levels (High / Medium / Low)
* Automate support workflow using Machine Learning

---

## 📂 Dataset

Dataset used: **Customer Support Tickets Dataset (Kaggle)**

### 🔑 Important Features Used:

* `Ticket Description` → Input text
* `Ticket Type` → Category (Target variable)
* `Ticket Priority` → Priority reference

---

## 🛠️ Technologies Used

* Python
* Pandas
* NLTK (Natural Language Processing)
* Scikit-learn

---

## ⚙️ Methodology

### 🔹 1. Data Preprocessing

* Converted text to lowercase
* Removed punctuation
* Removed stopwords using NLTK

### 🔹 2. Feature Extraction

* Used **TF-IDF Vectorizer**
* Included **n-grams (1,2)** for better context understanding

### 🔹 3. Model Training

* Used **Logistic Regression**
* Train-test split: 80% training, 20% testing

### 🔹 4. Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score

---

## 📊 Results

* **Accuracy:** ~20%
* Model successfully classifies tickets into categories

### 🔍 Example:

Input:
"My payment failed and app is not working"

Output:

* Category: Technical Issue
* Priority: High
<img width="1023" height="691" alt="Screenshot 2026-04-09 123655" src="https://github.com/user-attachments/assets/428ee754-8931-4cea-b929-8be68fa2394a" />






---

## 🚀 Features

✔ Text preprocessing using NLP
✔ Ticket classification using ML
✔ Priority prediction using rule-based logic
✔ Model evaluation with metrics

---

## ⚠️ Limitations

* Low accuracy due to dataset complexity
* Basic preprocessing techniques used
* No advanced models or deep learning

---

## 🔮 Future Improvements

* Use advanced models (SVM, Random Forest, Deep Learning)
* Apply lemmatization and stemming
* Improve dataset quality and balance
* Build a web-based interface (Streamlit/Flask)

---



## 💡 Project Use Case

This system can be used by:

* SaaS companies
* Customer support teams
* IT service management systems

to automate ticket handling and improve efficiency.

---

## 📌 Conclusion

This project demonstrates how Machine Learning and NLP can be applied to real-world business problems like support ticket classification and prioritization.

---

