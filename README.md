# ❤️ Heart Disease Prediction System using SVM

## 🚀 Business Scenario

Predicting the likelihood of heart disease can significantly improve health outcomes by enabling early intervention. This project focuses on developing a **Heart Disease Prediction System** using a **Support Vector Machine (SVM)** model trained on the **Heart Failure Prediction Dataset** from Kaggle. The system is designed to predict whether an individual is at risk of heart disease based on several clinical features, with a user-friendly interface deployed using **Streamlit** and hosted on **Heroku**.

---

## 🧠 Business Problem

Heart disease is one of the leading causes of death worldwide. Early detection and prediction can save lives by enabling doctors and healthcare professionals to intervene early. The goal of this project is to build a **predictive model** that can accurately classify whether a patient is at risk of heart disease, based on medical data such as age, blood pressure, cholesterol levels, and more.

---

## 🛠️ Solution Approach

This project follows a **machine learning workflow** with the following steps:

### 1️⃣ **Data Collection and Preprocessing**

- **Dataset**:
  - Used the **Heart Failure Prediction Dataset** from Kaggle ([Dataset Link](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)).
  - The dataset includes clinical information such as age, gender, blood pressure, cholesterol levels, heart rate, and other features that are indicative of heart disease risk.
- **Data Preprocessing**:
  - **Feature Scaling**: Scaled the numerical features to ensure uniformity (using **StandardScaler**).
  - Split the data into training and testing sets (80% training, 20% testing).

### 2️⃣ **Model Building (SVM Classifier)**

- **Support Vector Machine (SVM)**: A powerful classifier for binary classification problems like heart disease prediction.
  - Chose **Radial Basis Function (RBF)** kernel due to its ability to model complex, non-linear decision boundaries.
  - **Hyperparameter Tuning**: Used **GridSearchCV** for finding optimal values for C and gamma.
  - The model was trained to classify patients into two classes: 'Risk of Heart Disease' or 'No Risk of Heart Disease'.

### 3️⃣ **Model Training and Evaluation**

- **Evaluation Metrics**:
  - **Accuracy**: Measures the proportion of correct predictions.
  - **Precision, Recall, F1-Score**: Provided additional insights into the model's performance, particularly on imbalanced classes.
  - **Confusion Matrix**: Visualized to assess false positives and false negatives.

---

## 📊 Model Performance

| Metric    | Value |
| --------- | ----- |
| Precision | 0.58  |
| Recall    | 1.00  |
| F1-Score  | 0.74  |

---

### 📌 Why Prioritize Recall in Heart Disease Prediction?

In heart disease prediction, prioritizing **recall** is crucial because it ensures that individuals at risk of heart disease are identified, minimizing the chances of missing high-risk patients who may need urgent medical attention. False negatives can have severe consequences, such as delayed treatment, making it more acceptable to have false positives, which can be addressed with further testing. Since heart disease prediction often involves imbalanced data, emphasizing recall helps detect at-risk patients, leading to timely healthcare interventions and better outcomes. Ultimately, ensuring that those at risk are flagged is more important than minimizing false positives.

---

## ⚠️ Limitations

1️⃣ **Dataset Constraints**

- The dataset may not cover all possible demographic factors or rare medical conditions, which could limit the model's generalization.
- Performance may decrease when applied to patients with significantly different clinical profiles than those seen in the dataset.

2️⃣ **Model Interpretability**

- While SVM models can be highly effective, they can be challenging to interpret, particularly when compared to decision tree-based models.

3️⃣ **Real-World Deployment**

- For deployment in real-world healthcare systems, additional steps like model validation with clinical data, ethical considerations, and compliance with healthcare regulations would be needed.

---

## 🧠 Key Skills Demonstrated

✅ **Machine Learning with SVM**  
✅ **Data Preprocessing and Feature Scaling**  
✅ **Model Evaluation & Performance Metrics**  
✅ **Hyperparameter Tuning using GridSearchCV**  
✅ **Data Handling using Pandas & NumPy**  
✅ **Visualization with Matplotlib & Seaborn**  
✅ **Streamlit for Frontend Development**  
✅ **Heroku Hosting for Deployment**

---

## 🛠️ Technical Tools & Libraries

- **Python**: Main programming language.
- **scikit-learn**: For machine learning model development, including SVM.
- **Pandas**: For data handling and analysis.
- **NumPy**: For numerical computations.
- **Matplotlib**: For data visualization.
- **Seaborn**: For statistical data visualization.
- **Streamlit**: For creating an interactive web frontend to visualize the model’s predictions.
- **Heroku**: For hosting the application online, making it accessible to users.
