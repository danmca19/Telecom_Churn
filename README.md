# 📉 Customer Churn Prediction with Python (Telco Dataset)

This project outlines a comprehensive machine learning pipeline for **customer churn prediction** using real-world telecom data. It covers data preprocessing, feature engineering, model training, and performance evaluation, with a critical focus on understanding and quantifying the **business cost of false predictions** in retention campaigns.

---

## 🗂 Project Structure

* **`Churn_Analysis.ipynb`**
    Main notebook detailing the full churn modeling process from data loading to final evaluation.
* **`ConfusionMatrix.ipynb`**
    Notebook dedicated to **interpreting the confusion matrix** and analyzing the precision-recall trade-off in churn scenarios.
* **`WA_Fn-UseC_-Telco-Customer-Churn.csv`**
    Public dataset comprising customer demographics, subscribed services, contract types, billing information, and the churn label.

---

## 💻 Technologies Used

* Python 3.x
* `pandas`, `numpy` – for data wrangling
* `matplotlib`, `seaborn` – for visual analytics
* `scikit-learn` – for ML algorithms and evaluation
* Jupyter Notebooks – for iterative exploration

---

## 🎯 Business Objective

The primary goal is to build a robust model that **predicts customers at high risk of churning**, enabling the business to:

* Proactively retain valuable customers.
* Optimize retention campaign targeting and efficiency.
* Reduce overall customer acquisition and churn costs.

---

## 🔎 Key Insights from Exploratory Data Analysis (EDA)

* **Short Tenure Increases Churn Risk:** Customers with less than 12 months tenure exhibit significantly higher churn rates, indicating a critical engagement period.
* **Contract Type Impacts Retention:** Month-to-month plans are associated with higher churn rates compared to annual or biennial contracts.
* **Electronic Check Users Show Lower Loyalty:** A notable proportion of churners use electronic check payments, suggesting potential dissatisfaction or weaker engagement.
* **High Monthly Charges Correlate with Churn:** Elevated monthly charges, especially without corresponding value from bundled services or loyalty benefits, are linked to increased churn.
* **Low Service Usage = Higher Risk:** Customers utilizing fewer value-added services (e.g., streaming, security) are more prone to churn, possibly due to lower perceived value.
* **Class Imbalance:** Only approximately 26.5% of customers churned, indicating a significant class imbalance problem that necessitates specific handling during model training.
* **Data Type & Correlation Insights:**
    * `TotalCharges` required conversion from string to numeric, with missing values handled, highlighting a crucial preprocessing step.
    * `TotalCharges` is strongly correlated with `tenure` (r = 0.83), underscoring that customer lifetime revenue heavily depends on retention duration.
    * `MonthlyCharges` and `TotalCharges` show moderate correlation (r = 0.65), suggesting that some high-paying users churn prematurely, potentially due to dissatisfaction.
    * `SeniorCitizen` shows weak individual correlation with churn, suggesting the need for cross-variable analysis to uncover hidden patterns.

---

## 📊 Model Evaluation Metrics & Error Cost Analysis

The project prioritizes business-aligned metrics beyond simple accuracy: **Precision**, **Recall**, **F1 Score**, and the **Confusion Matrix** for detailed error type visualization.

### Understanding Error Costs: False Positives & False Negatives

In churn modeling, differentiating error types is critical as their business impact varies significantly:

| ❗ Error Type        | Definition                             | Business Impact                                      | Estimated Cost (Telco)                                                                                 |
| :------------------ | :------------------------------------- | :--------------------------------------------------- | :----------------------------------------------------------------------------------------------------- |
| **False Positive** | Predicted churn, but customer would stay | 🟡 *Unnecessary retention action* → Wasted resources | **R$ 150.00** (Cost of an unnecessary retention program/intervention)                                  |
| **False Negative** | Predicted no churn, but customer leaves  | 🔴 *Missed opportunity* → Revenue loss, customer lost | **R$ 1,680.00** (Estimated lost Customer Lifetime Value - CLV, based on ARPU of R$ 70.00 x 24 months) |

---

### Logistic Regression Performance & Cost Analysis

Based on the specific Confusion Matrix for the Logistic Regression model:

**Logistic Regression Confusion Matrix:**

|              | Predicted Churn | Predicted No Churn |
| :----------- | :-------------- | :----------------- |
| **Actual Churn** | TP = 314        | FN = 156           |
| **Actual No Churn** | FP = 247        | TN = 1396          |

**Cost-Based Insights for Logistic Regression:**

* **Total False Positive (FP) Waste:** `247 FP * R$ 150.00/FP = R$ 37,050.00`
    * *Insight:* This model generated **247 false alarms**, leading to an estimated **R$ 37,050.00** in wasted retention efforts on customers who would not have churned.
* **Total False Negative (FN) Loss:** `156 FN * R$ 1,680.00/FN = R$ 262,080.00`
    * *Insight:* The model failed to identify **156 actual churners**, resulting in a substantial estimated loss of **R$ 262,080.00** in Customer Lifetime Value.
* **Overall Model Error Cost:** `R$ 37,050.00 (FP) + R$ 262,080.00 (FN) = R$ 299,130.00`
    * *Insight:* The total estimated financial impact of misclassifications by the Logistic Regression model is nearly **R$ 300,000.00**, underscoring significant opportunities for financial optimization through model improvement.
    * *Key Observation:* The overwhelming majority of this cost (over 87%) stems from False Negatives, highlighting that **missed churners are a much larger financial drain** than wasted retention efforts for this specific model and cost structure.

---

## 📈 Comparative Model Evaluation & Strategic Recommendations

### Comparative Model Performance:

* **Top Performer: Logistic Regression (Reconfirmed)**
    * Offers an excellent balance across general metrics (F1-Score, LogLoss, Kappa, AUC), indicating strong overall predictive power.
    * Its high interpretability is a significant advantage for transparent business decision-making.
    * **Strategic Implication:** While it's the best overall, the high cost from False Negatives (missed churners) suggests that adjusting its classification threshold to prioritize **recall** could yield a better overall financial outcome if preventing churn is the primary business objective.
* **Strong Alternative: Gradient Boosting**
    * Achieved competitive performance across metrics, making it a viable option for situations prioritizing slight performance gains over interpretability.
* **High Recall Option: Naive Bayes**
    * Distinguished by an exceptionally high recall (89%), meaning it effectively detects most churn-prone customers.
    * However, its low precision (44%) results in a higher volume of false positives (unnecessary retention efforts).
    * **Strategic Fit:** This model is ideal for scenarios where the cost of losing a customer (False Negative) is *substantially* greater than the cost of an unnecessary intervention (False Positive), aligning with an "err on the side of prevention" strategy.
* **Other Models:**
    * **Random Forest** showed good general performance but did not particularly excel in the main metrics.
    * **Decision Tree** presented median performance and displayed a tendency towards overfitting.
    * **SVM** largely failed to capture the positive class (zero recall and F1-Score), likely due to a lack of configuration for class imbalance (e.g., missing `class_weight='balanced'` or threshold adjustments).

---

### Strategic Business Recommendations:

| Business Finding                 | Suggested Action                                                                                                              |
| :------------------------------- | :---------------------------------------------------------------------------------------------------------------------------- |
| Short-tenure users are high-risk | Launch **onboarding and loyalty campaigns** within the first 6-12 months to proactively engage new customers.                   |
| Month-to-month customers churn more | Promote **long-term contracts** (e.g., annual, biennial) via discounts or bonus programs to increase commitment.                  |
| Electronic check users show low loyalty | Offer **incentives to switch to card/autopay** or conduct research to investigate friction points with this payment method. |
| High-paying users still churn    | Reassess **price-value perception** for high-tier plans; bundle additional benefits or offer personalized value propositions. |
| Light service users are less committed | Encourage **upselling or usage-based engagement** for underutilized services to demonstrate perceived value.                      |
| **Cost of False Negatives is High** | **Adjust model thresholds to prioritize recall**, even if it slightly increases false positives, given the significant CLV loss. |

### Technical Recommendations for Model Improvement:

* **Address Class Imbalance:** Implement techniques like SMOTE, ADASYN, or `class_weight` parameter adjustments for models (especially SVM and Random Forest) to enhance their ability to detect the critical minority (churn) class.
* **Hyperparameter Tuning:** Conduct thorough hyperparameter optimization (using methods like GridSearchCV or Optuna) for promising models (Logistic Regression, Gradient Boosting, Naive Bayes) to further enhance their performance and optimize for cost-sensitive metrics.
* **Model Usefulness with Segmentation:** Explore segmenting customers (e.g., by high-LTV vs. low-value) and applying different prediction models or intervention strategies for each segment to optimize overall ROI.
* **A/B Testing Model ROI:** Implement A/B tests to validate the real-world ROI of churn prevention strategies by comparing churn rates in groups exposed to interventions (based on model predictions) versus control groups.

---

## 📚 References

* Dataset from IBM: [https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/](https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/)
* Inspired by real-world churn mitigation strategies in telecom and subscription services.
