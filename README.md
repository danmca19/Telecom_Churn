# Customer Churn Prediction with Python (Telco Dataset)

This project demonstrates a complete machine learning workflow for **predicting customer churn** using a real-world telecom dataset. It includes data preprocessing, feature engineering, model training, and evaluation — along with visualization of performance metrics such as the confusion matrix.

---

## ➤ Project Structure

- **`PA_Churn_Daniel.ipynb`**  
  Main notebook containing the end-to-end machine learning pipeline. This includes:
  - Data loading and cleaning
  - Exploratory Data Analysis (EDA)
  - Feature encoding and transformation
  - Model training and evaluation (classification)

- **`ConfusionMatrix_PA.ipynb`**  
  Focused on visualizing the performance of the model, especially the **confusion matrix**, to better understand false positives/negatives and accuracy breakdown.

- **`WA_Fn-UseC_-Telco-Customer-Churn.csv`**  
  Public dataset from IBM containing anonymized customer information and churn labels. Fields include:
  - Demographics (gender, senior citizen, partner, etc.)
  - Account data (tenure, service subscriptions, contract type)
  - Payment data (monthly charges, total charges, payment method)

---

## ➤ Objective

The main goal is to **predict which customers are likely to churn**, enabling telecom companies to:
- Develop retention strategies
- Optimize marketing efforts
- Improve customer service and offerings

The project simulates a real-world use case where business teams could act based on insights from churn predictions.

---

## ➤ Technologies Used

- **Python 3.x**
- **Pandas** – data manipulation and cleaning
- **NumPy** – numerical operations
- **Matplotlib / Seaborn** – data visualization
- **Scikit-learn** – machine learning models and metrics
- **Jupyter Notebook** – interactive development environment

---

## ➤ Model Evaluation

Evaluation metrics used to measure model performance:
- Accuracy
- Precision, Recall, and F1-score
- Confusion Matrix
- ROC Curve (optional extension)
  
These metrics help determine how well the model distinguishes churned vs. retained customers.
---

## ➤ Insights & Business Recommendations

Based on the exploratory analysis and model predictions, the following actionable insights were observed:

### Insights

- **Short Tenure Increases Churn Risk**  
  Customers with less than 12 months of tenure are significantly more likely to churn. This suggests a critical period for engagement in the early lifecycle.

- **Contract Type Impacts Retention**  
  Month-to-month customers exhibit higher churn rates compared to those on one- or two-year contracts.

- **Electronic Payment Users Are More Likely to Churn**  
  A notable number of churners prefer electronic check payments — possibly indicating less commitment or lack of satisfaction.

- **High Monthly Charges Are Linked to Churn**  
  Customers paying higher monthly amounts tend to churn more, especially if combined with shorter tenure and fewer bundled services.

### Recommendations

- **Early Retention Campaigns**  
  Target new customers (less than 6 months) with welcome bonuses, discounts, or onboarding guidance to reduce early churn.

- **Promote Long-Term Contracts**  
  Offer incentives for customers to switch from monthly plans to longer-term contracts (e.g., lower rates, loyalty points).

- **Analyze Payment Method Satisfaction**  
  Investigate user experience issues with electronic payment users and provide alternative payment options or support.

- **Bundle and Price Optimization**  
  Consider redesigning service bundles for high-paying users to increase perceived value and reduce churn triggers.

