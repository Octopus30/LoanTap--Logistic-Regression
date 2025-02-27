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

---

## **Techniques to Handle Missing Values**

### **1. Removing Missing Values**
Useful for MCAR data when missingness is low.
```python
df_cleaned = df.dropna()  # Drop rows with missing values
df_cleaned = df.drop(columns=['column_name'])  # Drop columns with too many missing values
```
✅ **Best for:** Minimal missing data, MCAR scenarios.

### **2. Mean/Median/Mode Imputation**
For numerical data:
```python
df['column_name'].fillna(df['column_name'].mean(), inplace=True)  # Mean
```
For categorical data:
```python
df['category_column'].fillna(df['category_column'].mode()[0], inplace=True)  # Mode
```
✅ **Best for:** MAR data, univariate missingness.

### **3. K-Nearest Neighbors (KNN) Imputation**
```python
from sklearn.impute import KNNImputer
import pandas as pd

imputer = KNNImputer(n_neighbors=3)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
```
✅ **Best for:** MAR data with patterns.

### **4. Regression Imputation**
Predict missing values using regression:
```python
from sklearn.linear_model import LinearRegression

known_data = df[df['column_name'].notnull()]
missing_data = df[df['column_name'].isnull()]

model = LinearRegression()
model.fit(known_data[['other_feature']], known_data['column_name'])

df.loc[df['column_name'].isnull(), 'column_name'] = model.predict(missing_data[['other_feature']])
```
✅ **Best for:** MAR data with strong correlations.

### **5. Multiple Imputation (MICE)**
```python
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer  

imputer = IterativeImputer()
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
```
✅ **Best for:** MAR data with complex relationships.

### **6. Deep Learning-Based Imputation**
```python
from sklearn.impute import SimpleImputer
import tensorflow as tf

imputer = SimpleImputer(strategy='mean')
df_imputed = imputer.fit_transform(df)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(df.shape[1], activation='linear')
])

model.compile(optimizer='adam', loss='mse')
model.fit(df_imputed, df_imputed, epochs=10, batch_size=16)
```
✅ **Best for:** MNAR data in large datasets.

---

## **Which Method Should You Choose?**

| **Method** | **Suitable For** | **Complexity** |
|------------|----------------|---------------|
| Drop Missing Values | MCAR, Low Missingness | Low |
| Mean/Median/Mode Imputation | MAR, Univariate Data | Low |
| KNN Imputation | MAR, Pattern-Based Missingness | Medium |
| Regression Imputation | MAR, Correlated Features | Medium |
| MICE (Iterative Imputation) | MAR, Complex Relationships | High |
| Deep Learning Imputation | MNAR, Large Datasets | High |

---

## **Final Thoughts**
Handling missing values correctly is essential for building reliable machine learning models. Simple techniques like mean imputation work in some cases, while advanced methods like KNN, MICE, or deep learning improve accuracy. The right approach depends on the nature of missingness and dataset complexity.

By applying the correct technique, you can ensure that your models are both **robust** and **accurate**!

🚀 **Ready to improve your data preprocessing? Try these methods on your dataset today!**


