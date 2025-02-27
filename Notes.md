# Handling Missing Values in Machine Learning

## Introduction
Missing values are common in datasets and can impact the performance of machine learning models. Understanding the types of missing data and applying appropriate imputation techniques is crucial for building robust models.

## Types of Missing Data
### 1. MCAR (Missing Completely at Random)
- **Definition:** Missing values occur randomly and are not related to any other variables.
- **Example:** A survey respondent skips a question for no specific reason.
- **Advantages:**
  - No bias in the data.
  - Simple handling methods (e.g., deletion or mean imputation) work well.
- **Disadvantages:**
  - Rare in real-world datasets.

### 2. MAR (Missing at Random)
- **Definition:** Missing values depend on observed data but not on missing values themselves.
- **Example:** Income data is missing more often for younger people but is unrelated to actual income.
- **Advantages:**
  - Can be handled with statistical imputation techniques like regression or multiple imputation.
- **Disadvantages:**
  - Requires additional modeling to properly handle missing values.

### 3. MNAR (Missing Not at Random)
- **Definition:** The probability of missing data depends on the missing values themselves.
- **Example:** Patients with severe illness may not report their symptoms.
- **Advantages:**
  - Understanding the missingness mechanism helps refine the model.
- **Disadvantages:**
  - Requires domain knowledge or assumptions to handle.
  - Often leads to biased results if not addressed properly.

## How to Identify MCAR, MAR, and MNAR
### **1. Visual Inspection**
#### **Code Snippet: Missing Data Heatmap**
```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame({'A': [1, 2, None, 4, 5], 'B': [None, 2, 3, 4, None]})
sns.heatmap(df.isnull(), cmap='viridis', cbar=False)
plt.show()
```
**Interpretation:**
- If missing values are randomly scattered, it could indicate MCAR.
- If missingness is concentrated in specific sections, it may be MAR or MNAR.

### **2. Little’s MCAR Test**
Statistically tests if data is MCAR.

#### **Code Snippet: Running Little's MCAR Test**
```python
from missingno import matrix
import pymcar.mcar

result = pymcar.mcar(df)
print(result)
```
**Possible Decision:**
- If **p-value > 0.05**, data is likely MCAR.
- If **p-value ≤ 0.05**, missingness is likely MAR or MNAR.

### **3. Correlation Analysis for MAR**
If missingness is related to observed variables, it's MAR.

#### **Code Snippet: Checking Correlation Between Missing Data and Other Features**
```python
import numpy as np

df['missing_A'] = df['A'].isnull().astype(int)
df_corr = df.corr()
print(df_corr)
```
**Interpretation:**
- If the missingness of `A` correlates with another variable (e.g., `B`), data is MAR.
- If there is **no correlation**, data may be MCAR.

### **4. MNAR Detection (Self-Selection Bias)**
Check if missingness depends on unobserved values.

#### **Code Snippet: Comparing Missing and Non-Missing Distributions**
```python
import matplotlib.pyplot as plt

df['A_missing'] = df['A'].isnull()
sns.boxplot(x=df['A_missing'], y=df['B'])
plt.show()
```
**Interpretation:**
- If missing values have a different distribution than observed values, data is likely MNAR.

---

## **Decision Based on Output**
| **Test** | **If True** | **If False** |
|----------|------------|-------------|
| Little’s MCAR Test (p > 0.05) | Data is MCAR | Data is MAR or MNAR |
| Correlation with Other Variables | Data is MAR | Data is MCAR or MNAR |
| Different Distribution of Missing Data | Data is MNAR | Data is MAR or MCAR |

## **Conclusion**
Identifying the type of missing data is crucial for applying the right imputation method. Understanding MCAR, MAR, and MNAR ensures that machine learning models are not biased due to missing values. By using visualization and statistical tests, we can make informed decisions about handling missing data.

