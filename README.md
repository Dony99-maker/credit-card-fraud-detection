# Credit Card Fraud Detection Project

This project focuses on detecting fraudulent credit card transactions using Machine Learning (Logistic Regression) on a highly imbalanced dataset. The dataset contains 284,807 transactions, out of which only 492 are fraud cases.

---

## 📌 Dataset
The dataset is too large to upload to GitHub.  
Download it from Kaggle:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## 🚀 Project Workflow

### 1️⃣ Data Loading & Inspection
- Loaded the dataset using Pandas  
- Checked for missing values  
- Viewed class distribution (fraud vs non-fraud)

### 2️⃣ Exploratory Data Analysis (EDA)
- Visualized class imbalance  
- Reviewed summary statistics  

### 3️⃣ Data Preprocessing
- Scaled the `Time` and `Amount` columns  
- Separated features and target  
- Applied **SMOTE** to handle severe class imbalance  

### 4️⃣ Model Training
- Trained **Logistic Regression**  
- Performed train-test split  
- Used model to predict fraud transactions  

### 5️⃣ Model Evaluation
- Confusion Matrix  
- Classification Report  
- ROC-AUC Score  
- ROC Curve  

All evaluation plots are saved in the `/images` folder.

### 6️⃣ Saving Artifacts
- Trained model saved as `fraud_model.pkl`  
- Scaler saved as `scaler.pkl`  

Both are stored in the `/model` folder.

---

## 📊 Results
- Improved fraud detection after applying SMOTE  
- High ROC-AUC score  
- Clear performance insights using visualizations  

---

## 🛠 Technologies Used
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-Learn  
- Imbalanced-Learn  
- Joblib  

---

## 🔮 Future Improvements
- Try Random Forest, XGBoost, SVM  
- Deploy model using Streamlit or Flask  
- Add real-time fraud detection dashboard  

---

## 🙌 Acknowledgements
Dataset provided by the Machine Learning Group – Université Libre de Bruxelles.
Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud



