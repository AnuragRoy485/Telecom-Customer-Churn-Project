
---

```markdown
# 📞 Telecom Customer Churn Prediction

Predicting whether telecom customers will churn using data analysis and a Support Vector Classifier (SVC) machine learning model.

This repository contains:
- A complete **Jupyter Notebook** (`churn (1).ipynb`) that performs EDA, preprocessing, model training, and evaluation.
- A **trained machine learning model** stored as `svc_model.pkl` for direct use in predictions.

---

## 📌 Project Overview

The objective of this project is to analyze telecom customer data and predict whether a customer is likely to churn. The notebook walks through:

- Data loading and inspection  
- Exploratory Data Analysis (EDA)  
- Data cleaning and preprocessing  
- Feature encoding and scaling  
- Training a Support Vector Classifier (SVC)  
- Model evaluation  
- Saving the trained model as `svc_model.pkl`  

---

## 📂 Repository Structure

```

.
├── churn (1).ipynb          # Main notebook for EDA, preprocessing, and model training
├── svc_model.pkl            # Trained Support Vector Classifier model
└── README.md                # Project documentation

````

---

## 🧠 Machine Learning Model

The model trained for this project is:

### ✔ Support Vector Classifier (SVC)
- Works well for binary classification  
- Trained on encoded & scaled features  
- Achieves competitive accuracy and generalization  
- Saved as `svc_model.pkl`

### 🔧 Load the Model for Predictions

```python
import pickle

with open("svc_model.pkl", "rb") as f:
    model = pickle.load(f)

# Example prediction (ensure feature preprocessing before prediction)
model.predict([[...processed_feature_array...]])
````

---

## 🧪 Workflow Summary (from the Notebook)

### 1️⃣ Data Loading

* Read the telecom churn dataset
* Inspected missing values & anomalies

### 2️⃣ Exploratory Data Analysis (EDA)

* Churn distribution
* Pattern analysis by demographics & service usage
* Correlation heatmap
* Visualizations using Matplotlib & Seaborn

### 3️⃣ Preprocessing

* Converted non-numeric fields
* Encoded categorical data
* Scaled numerical features
* Train-test split

### 4️⃣ Model Training (SVC)

* Trained Support Vector Classifier
* Evaluated using accuracy, precision, recall, F1-score
* Saved final model as `svc_model.pkl`

---

## 🚀 How to Run This Project

### **1. Clone the Repository**

```bash
git clone https://github.com/AnuragRoy485/Telecom-Customer-Churn-Project.git
cd Telecom-Customer-Churn-Project
```

### **2. Install Dependencies**

Create a virtual environment (optional):

```bash
pip install -r requirements.txt
```

If you don’t have `requirements.txt`, install manually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### **3. Run the Jupyter Notebook**

```bash
jupyter notebook "churn (1).ipynb"
```

### **4. (Optional) Load the Saved Model**

Use `svc_model.pkl` for inference without retraining.

---

## 📊 Model Performance (Update with your values)

| Metric    | Score |
| --------- | ----- |
| Accuracy  | ___   |
| Precision | ___   |
| Recall    | ___   |
| F1-score  | ___   |

---

## 🛠 Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Pickle

---

## 👨‍💻 Author

**Anurag Roy**
GitHub: [AnuragRoy485](https://github.com/AnuragRoy485)

---
