
# Bank Churn Data Exploration and Churn Prediction

## 📊 Project Overview

This project focuses on predicting customer churn in the banking industry using machine learning techniques. Customer churn prediction helps banks identify customers who are likely to close their accounts and leave the bank, enabling proactive retention strategies.

## 🎯 Objective

The main goal is to build a predictive model that can accurately identify customers at risk of churning, allowing the bank to take preventive measures and improve customer retention rates.

## 📈 Dataset

The dataset contains information about bank customers including:
- **Demographics**: Age, Gender, Geography
- **Account Information**: Credit Score, Balance, Number of Products
- **Behavioral Data**: Tenure, Activity Status, Estimated Salary
- **Target Variable**: Exited (1 = Churned, 0 = Retained)

### Dataset Features:
- `RowNumber`: Unique identifier for each customer
- `CustomerId`: Unique customer ID
- `Surname`: Customer's surname
- `CreditScore`: Customer's credit score
- `Geography`: Customer's location (France, Germany, Spain)
- `Gender`: Customer's gender (Male, Female)
- `Age`: Customer's age
- `Tenure`: Number of years with the bank
- `Balance`: Account balance
- `NumOfProducts`: Number of bank products used
- `HasCrCard`: Whether customer has credit card (1 = Yes, 0 = No)
- `IsActiveMember`: Whether customer is active (1 = Yes, 0 = No)
- `EstimatedSalary`: Customer's estimated salary
- `Exited`: Target variable (1 = Churned, 0 = Retained)

## 🔍 Methodology

### 1. Exploratory Data Analysis (EDA)
- Data quality assessment and missing value analysis
- Statistical summary of numerical and categorical features
- Churn rate analysis across different customer segments
- Correlation analysis between features
- Visualization of key patterns and trends

### 2. Data Preprocessing
- Handling missing values and outliers
- Feature engineering and transformation
- Encoding categorical variables (Label Encoding, One-Hot Encoding)
- Feature scaling and normalization
- Train-test split for model evaluation

### 3. Model Development
Multiple machine learning algorithms were implemented and compared:
- **Logistic Regression**: Baseline linear model
- **Decision Tree**: Non-linear tree-based model
- **Random Forest**: Ensemble method with multiple trees
- **Support Vector Machine (SVM)**: Kernel-based classification
- **Gradient Boosting**: Advanced ensemble technique

### 4. Model Evaluation
Models were evaluated using various metrics:
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity to churned customers
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Detailed classification results

## 📊 Key Findings

### Customer Churn Insights:
- **Churn Rate**: Overall churn rate is approximately 20%
- **Geography Impact**: Customers from Germany have the highest churn rate
- **Age Factor**: Older customers (45+ years) are more likely to churn
- **Product Usage**: Customers with only 1 product have higher churn probability
- **Account Balance**: Customers with zero balance show higher churn tendency
- **Activity Level**: Inactive members are more prone to churning

### Feature Importance:
1. **Age**: Most significant predictor of churn
2. **Geography**: Location strongly influences churn behavior
3. **Balance**: Account balance is a key retention factor
4. **Number of Products**: Product usage affects loyalty
5. **Activity Status**: Customer engagement level matters

## 🏆 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|---------|----------|---------|
| Logistic Regression | 0.81 | 0.65 | 0.47 | 0.55 | 0.74 |
| Decision Tree | 0.79 | 0.60 | 0.52 | 0.56 | 0.72 |
| Random Forest | 0.86 | 0.74 | 0.56 | 0.64 | 0.83 |
| SVM | 0.84 | 0.70 | 0.51 | 0.59 | 0.79 |
| Gradient Boosting | 0.87 | 0.76 | 0.59 | 0.66 | 0.85 |

**Best Model**: Gradient Boosting with 87% accuracy and 0.85 ROC-AUC score

## 💡 Business Recommendations

1. **Targeted Retention Programs**: Focus on customers aged 45+ years
2. **Geographic Strategy**: Implement special retention programs in Germany
3. **Product Cross-selling**: Encourage customers to use multiple products
4. **Engagement Campaigns**: Activate inactive customers through personalized offers
5. **Balance Management**: Provide financial advisory services for low-balance customers

## 🛠️ Technologies Used

- **Programming Language**: Python
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Model Enhancement**: XGBoost, LightGBM
- **Development Environment**: Jupyter Notebook

## 📁 Project Structure

```
bank-churn-prediction/
├── README.md
├── data/
│   └── bank_churn_data.csv
├── notebooks/
│   └── bank_churn_analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
├── models/
│   └── best_model.pkl
├── requirements.txt
└── results/
    ├── plots/
    │   ├── churn_distribution.png
    │   ├── correlation_matrix.png
    │   └── feature_importance.png
    └── model_performance.csv
```

## 📝 Usage

1. **Data Exploration**: Open the Jupyter notebook to explore the dataset
2. **Model Training**: Run the preprocessing and training scripts
3. **Prediction**: Use the trained model to predict churn for new customers
4. **Evaluation**: Analyze model performance and business insights

## 🔮 Future Enhancements

- Implement deep learning models (Neural Networks)
- Add real-time prediction capabilities
- Develop a web application for interactive predictions
- Include more advanced feature engineering techniques
- Implement automated model retraining pipeline

## 📧 Contact

**Author**: Karan Rajesh Shende 
**Email**: shendekaran144@gmail.com
**LinkedIn**:https://www.linkedin.com/in/karanshende


## 🙏 Acknowledgments

- Dataset source: [Kaggle Bank Customer Churn Dataset]
- Inspiration from various machine learning projects and tutorials
- Thanks to the open-source community for excellent libraries and tools

---

⭐ **If you found this project helpful, please give it a star!** ⭐



