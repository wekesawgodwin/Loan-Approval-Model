# Machine Learning Model for Loan Approval

## Project Overview

This project develops a cost-sensitive machine learning pipeline for loan approval using historical application data from a fintech lending context. The goal is to support more consistent, faster, and more explainable lending decisions by estimating the probability that an applicant should be approved, while accounting for the very different business costs of two types of errors:

- **False approval of a bad loan**: approximately **$50,000** loss per defaulted loan.
- **False denial of a good loan**: approximately **$8,000** in missed profit per creditworthy applicant.

Because the cost of approving a risky borrower is much higher than the cost of denying a good borrower, this project uses a **classification approach** with probability outputs and **decision-threshold tuning** rather than relying on accuracy alone.

## Business Problem

FinTech Innovations currently relies on manual loan review. That process is:

- slow,
- inconsistent,
- difficult to scale,
- and potentially biased.

A machine learning solution can help by:

- standardizing loan screening,
- reducing review time,
- supporting interpretable decision-making,
- and reducing expected financial loss.

## Dataset

The dataset contains **20,000 loan applications** and **35 columns**. It includes a mix of numerical and categorical features covering applicant demographics, income, credit profile, liabilities, assets, repayment behavior, and loan terms.

### Target Variable

- `LoanApproved` — binary label indicating whether the historical loan was approved.

### Example Feature Groups

**Numerical features**
- Age
- AnnualIncome
- CreditScore
- Experience
- LoanAmount
- LoanDuration
- MonthlyDebtPayments
- CreditCardUtilizationRate
- DebtToIncomeRatio
- SavingsAccountBalance
- CheckingAccountBalance
- TotalAssets
- TotalLiabilities
- MonthlyIncome
- NetWorth
- BaseInterestRate
- InterestRate
- MonthlyLoanPayment
- TotalDebtToIncomeRatio
- RiskScore

**Categorical features**
- EmploymentStatus
- EducationLevel
- MaritalStatus
- HomeOwnershipStatus
- BankruptcyHistory
- LoanPurpose

## Project Structure

- `financial_loan_risk.ipynb` — complete notebook with EDA, preprocessing, modeling, evaluation, and interpretation.
- `financial_loan_data(1).csv` — input dataset used for training and evaluation.

## Methodology

This project follows the **CRISP-DM** framework.

### 1. Business Understanding

The primary objective is to minimize expected financial loss while keeping the model interpretable enough for credit decision support and governance.

### 2. Data Understanding

The notebook explores:

- class imbalance,
- missing values,
- feature distributions,
- skew in financial variables,
- and potential leakage variables.

### 3. Data Preparation

The preprocessing pipeline handles different feature types appropriately:

- **Numeric features**: median imputation + standard scaling
- **Ordinal feature** (`EducationLevel`): ordinal encoding
- **Nominal categorical features**: one-hot encoding
- **Missing values**: handled inside the pipeline to avoid leakage

### Leakage Control

The following variables are excluded from the approval model because they may reflect downstream underwriting or post-decision information:

- `RiskScore`
- `InterestRate`
- `BaseInterestRate`
- `MonthlyLoanPayment`

This helps keep the model realistic for deployment and avoids leakage from features that may not be available at application time.

### 4. Feature Engineering

The notebook creates business-relevant engineered features, including:

- `IncomeDebtCoverage`
- `NetAssetPosition`
- `DebtServiceRatio`
- `CreditInquiryRate`
- `LeverageRatio`

These features are designed to make the model more informative and easier to explain to non-technical stakeholders.

### 5. Modeling

The primary model is **Logistic Regression** because it is:

- interpretable,
- stable,
- suitable for credit scoring,
- and easy to convert into a probability-based risk score.

A **Decision Tree** benchmark is also included.

### 6. Evaluation

Model evaluation goes beyond accuracy and includes:

- accuracy,
- precision,
- recall,
- F1-score,
- ROC-AUC,
- confusion matrix analysis,
- and a **custom cost function** based on business impact.

The custom expected cost is defined as:

- **False Positive (approve bad loan)** = $50,000
- **False Negative (deny good loan)** = $8,000

### Threshold Tuning

Instead of using the default 0.50 classification threshold, the notebook searches over a range of thresholds on a validation set and selects the value that minimizes expected cost.

## Results

The tuned logistic regression model achieved the following on the held-out test set:

- **Threshold used**: `0.95`
- **Accuracy**: `90.5%`
- **Precision**: `95.6%`
- **Recall**: `63.2%`
- **ROC-AUC**: `0.979`
- **Test cost**: **$4,216,000**

For comparison, the default 0.50 threshold produced a much higher expected cost. The notebook reports a reduction from **$13,986,000** to **$4,216,000**, which is a drop of about **69.9%**.

## Interpretation

The logistic regression coefficients provide transparent insight into the drivers of approval probability. In general:

- positive coefficients increase the chance of approval,
- negative coefficients decrease the chance of approval.

This supports explainability, model governance, and communication with business stakeholders.

## Key Findings

- The historical data is imbalanced, so accuracy alone is not sufficient.
- A cost-sensitive threshold is crucial because loan approval errors are asymmetric.
- Logistic regression provides a strong balance between performance and interpretability.
- Financial and credit-behavior variables such as income, net worth, credit history, debt burden, and bankruptcy history are highly relevant in the lending context.

## Business Recommendations

1. **Use the model as a decision-support system**, not as a fully autonomous black-box approval engine.
2. **Apply the tuned threshold** rather than the default 0.50 cutoff.
3. **Monitor drift regularly** in applicant characteristics, approval rates, and post-booking default rates.
4. **Recalibrate the threshold periodically** as macroeconomic conditions change.
5. **Validate fairness and compliance metrics** before any production deployment.
6. **Keep preprocessing inside the pipeline** so training and inference remain consistent.

## Limitations

- The model is trained on historical data and may inherit biases present in prior lending decisions.
- If real-world applicant behavior changes, model performance may degrade over time.
- Some variables may be correlated with business decisions made after the application stage, so leakage control is important.
- The model should be reviewed alongside regulatory, ethical, and fairness constraints before deployment.

## How to Run the Project

### Requirements

Install the following Python packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Steps

1. Open `financial_loan_risk.ipynb` in Jupyter Notebook or JupyterLab.
2. Ensure `financial_loan_data(1).csv` is in the same directory or update the file path in the notebook.
3. Run the notebook cells from top to bottom.
4. Review the EDA outputs, model evaluation, threshold tuning results, and coefficient plots.

## Suggested Extensions

If you want to improve the project further, consider:

- trying additional models such as Random Forest, Gradient Boosting, or XGBoost,
- comparing calibration curves,
- adding fairness metrics across protected groups,
- tuning the decision threshold under different business cost assumptions,
- and comparing model performance under cross-validation.

## Authoring Note

This project is designed to demonstrate both technical machine learning competence and business reasoning. The emphasis is not only on predictive performance, but also on interpretability, cost-awareness, and practical deployment thinking.
