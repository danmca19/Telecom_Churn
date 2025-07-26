# 📉 Customer Churn Prediction with Python (Telco Dataset)

This project presents a complete machine learning pipeline to predict **customer churn** using real-world telecom data. It includes data preprocessing, feature engineering, model training, and performance evaluation — with a strong focus on understanding the **business cost of false predictions**, especially in retention campaigns.

---

## 🗂 Project Structure

- **`PA_Churn_Daniel.ipynb`**  
  Main notebook containing the full churn modeling process: from loading the dataset to final evaluation.

- **`ConfusionMatrix_PA.ipynb`**  
  Notebook dedicated to **interpreting the confusion matrix**, highlighting the trade-off between precision and recall in churn scenarios.

- **`WA_Fn-UseC_-Telco-Customer-Churn.csv`**  
  Public dataset containing:
  - Customer demographics and tenure
  - Subscribed services and contract type
  - Billing and payment methods
  - Churn label (`Yes` / `No`)

---

## 🎯 Business Objective

Build a model to **predict which customers are at high risk of churning**, allowing the business to:

- Proactively retain valuable customers
- Target campaigns more efficiently
- Reduce customer acquisition and churn costs

---

## 🔍 Why False Positives and False Negatives Matter

In churn modeling, understanding error types is essential:

| ❗ Error Type | Definition | Business Impact |
|--------------|------------|-----------------|
| **False Positive** | Predicted churn, but customer would stay | 🟡 *Unnecessary retention action* → Wastes resources |
| **False Negative** | Predicted no churn, but customer leaves | 🔴 *Missed opportunity* → Revenue loss, customer lost |

Thus, **high precision** avoids wasting money on customers that would stay anyway, while **high recall** ensures you capture actual churners — essential for long-term retention strategy.

---

## 💻 Technologies Used

- Python 3.x  
- `pandas`, `numpy` – data wrangling  
- `matplotlib`, `seaborn` – visual analytics  
- `scikit-learn` – ML algorithms and evaluation  
- Jupyter Notebooks for iterative exploration  

---

## 📊 Model Evaluation Metrics

The project goes beyond simple accuracy, prioritizing:

- **Precision (Churn)**: Focus on minimizing false positives
- **Recall (Churn)**: Focus on catching true churners
- **F1 Score**: Balance of precision and recall
- **Confusion Matrix**: Visualizing error types and proportions
- **ROC Curve** (optional extension)

These metrics are more aligned with **business decision-making**, where cost and impact of actions vary by customer behavior.

---

## 🔎 Key Insights from EDA

- **🕐 Short Tenure Increases Churn Risk**  
  Customers with less than 12 months of tenure show a significantly higher churn rate, indicating a critical period for engagement.

- **📄 Contract Type Impacts Retention**  
  Customers on month-to-month plans are more likely to churn than those with annual or biennial contracts.

- **💳 Electronic Check Users More Likely to Churn**  
  A high number of churners use electronic check payments, suggesting potential dissatisfaction or weaker engagement.

- **💰 High Monthly Charges Correlate with Churn**  
  Higher charges, especially when not matched by bundled services or loyalty benefits, are linked to higher churn.

- **🛎️ Low Service Usage = Higher Risk**  
  Customers using fewer value-added services (e.g., streaming, security) are more prone to churn, possibly due to lower perceived value.

---

## ✅ Strategic Recommendations

| Business Finding                         | Suggested Action                                                                 |
|------------------------------------------|----------------------------------------------------------------------------------|
| Short-tenure users are high-risk         | Launch **onboarding and loyalty campaigns** within first 6 months               |
| Month-to-month customers churn more      | Promote **long-term contracts** via discounts or bonus programs                 |
| Electronic check users show low loyalty  | Offer **incentives to switch to card/autopay** or investigate friction points   |
| High-paying users still churn            | Reassess **price-value perception** and bundle additional benefits              |
| Light service users are less committed   | Encourage **upselling or usage-based engagement** for unused services           |

---

## 📈 Sample Model Metrics (Illustrative)

| Metric             | Value         |
|--------------------|---------------|
| Accuracy           | ~82%          |
| Precision (Churn)  | ~73%          |
| Recall (Churn)     | ~78%          |
| F1 Score (Churn)   | ~75%          |
| FP Rate            | ~18%          |
| FN Rate            | ~22%          |

> Interpretation: The model performs well in identifying most actual churners (high recall), while keeping false positives at a manageable level.

---

## 🧠 Takeaways for Business Decision-Makers

- **Not all accuracy is good accuracy** — focus on **what kind of mistake costs you more**: wasting retention budget or losing a customer?
- **Even moderate recall gains can drive revenue** when integrated with CRM workflows.
- **Model usefulness improves with segmentation**: applying different strategies for high-LTV vs. low-value users can optimize ROI.
- **A/B testing should validate model ROI** by comparing churn rate with/without interventions.

---

## 🚀 Next Steps

- 📊 Test advanced models like **XGBoost** or **LightGBM**
- ⚖️ Apply **SMOTE** or **class weighting** to manage imbalance
- 🔧 Use **GridSearchCV** for hyperparameter tuning
- 🔄 Develop **weekly retraining pipeline**
- 🧩 Integrate into CRM or marketing tools for automated alerts and actions

---

## 📚 References

Dataset from IBM:  
https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/

Inspired by real-world churn mitigation strategies in telecom and subscription services.

---

