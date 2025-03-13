## Credit Card Fraud Detection

### Overview
Our objective is to create a classifier for credit card fraud detection. To achieve this, we will compare classification models from different methods:

- Logistic Regression
- Support Vector Machine (SVM)
- Bagging (Random Forest)
- Boosting (XGBoost)
- Neural Network (TensorFlow/Keras)

### Dataset
**Source:** [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

The dataset contains transactions made by credit cards in September 2013 by European cardholders. It consists of transactions over two days, with 492 fraud cases out of 284,807 transactions. Since fraud cases represent only 0.172% of all transactions, the dataset is highly imbalanced. 

To address this, we apply an **undersampling strategy** to rebalance the class distribution. The dataset includes only numerical input variables obtained through **PCA transformation**, and due to confidentiality, original feature details are not provided.

### Implementation

#### Libraries Used:
```bash
NumPy, Pandas, Matplotlib, Scikit-learn, Seaborn, Plotly, TensorFlow, Keras, Xgboost, Imbalanced-learn, Flask
```

### Data Exploration
Only **0.172%** of transactions are fraudulent, making the dataset highly imbalanced. Without handling this imbalance, classifiers might predict the majority class most of the time without meaningful analysis.

To mitigate this, we use **random undersampling**, which involves selecting a subset of the majority class to balance the dataset. However, this approach may remove useful data points that contribute to the decision boundary between classes.

### Model Performance Comparison (Rounded Down)
| Model                | Accuracy | F1 Score | AUC  |
|----------------------|---------|---------|------|
| Logistic Regression | 0.95    | 0.93    | 0.97 |
| SVM                 | 0.93    | 0.91    | 0.97 |
| Random Forest       | 0.94    | 0.92    | 0.96 |
| XGBoost            | 0.95    | 0.94    | 0.97 |
| Neural Network (MLP) | 0.92    | 0.90    | 0.97 |

### Future Improvements
- Implementing **Autoencoder-based anomaly detection**
- Using **SMOTE (Synthetic Minority Over-sampling Technique)** for fraud data generation
- Deploying the model as a **real-time fraud detection API**

---

 
