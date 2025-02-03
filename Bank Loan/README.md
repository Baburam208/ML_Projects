## This is a Bank loan approval project
This project can helps to approve loan in a bank. By entering some of the features of the customers', 
this machine learning model can give decision to whether give approval to the loan or not.

### 1. Dataset description
The dataset contains features: 
['ID', 'Age', 'Gender', 'Experience', 'Income', 'ZIP Code', 'Family',
    'CCAvg', 'Education', 'Mortgage', 'Home Ownership', 'Personal Loan',
    'Securities Account', 'CD Account', 'Online', 'CreditCard']


Total number of rows are 5000. 

The target column is: 'Personal Loan'

### 2. Data cleaning
#### I. Handling Missing Values
a. Drop the Row with a Blank Target Value
There is one row in the target column 'Personal Loan' where the value is blank (i.e., ' '). So, lets remove this row.
```
df = df[df['Personal Loan'] != ' ']
```

b. Impute Missing Values
For **Numerical Columns (Income, Online)**, impute with the median.
```
# Impute numerical columns
df['Income'] = df['Income'].fillna(df['Income'].median())
df['Online'] = df['Online'].fillna(df['Online'].median())
```

For **Categorical Columns (Gender, Home Ownership)**, impute with the mode (most frequent value).
```
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Home Ownership'] = df['Home Ownership'].fillna(df['Home Ownership'].mode()[0])
```

#### II. Handle Unusual Values
a. Clean the **Gender** column
Replace unusual values (`#`, `-`) with a default category ok `Unknown`.
```
df['Gender'] = df['Gender'].replace(['#', '-'], 'Unknown')
```

b. There are negative values in the **Experience** column
Negative values in the `Experience` column are likely errors. Replace them with 0.
```
df['Experience'] = df['Experience'].apply(lambda x: 0 if x < 0 else x)
```

#### III. Encode Categorical Variables
Convert categorical columns (`Gender`, `Home Ownership`) into numerical values using **one-hot encoding**.
```
df = pd.get_dummies(df, columns=['Gender', 'Home Ownership'], drop_first=True)
```

#### IV. Handle Class Imbalance
Since the dataset has highly imbalanced classes (4520 `0`s vs. 479 `1`s). To address this we have used **resampling (oversampling minority class)**
```
from imblearn.over_sampling import SMOTE

X = df.drop(columns=['Personal Loan', 'ID', 'ZIP Code'])  # Features
y = df['Personal Loan']. astype(int)  # Target

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

#### V. Feature Engineering
a. Drop Irrelevant Columns
Drop columns like `ID` and `ZIP Code` that are unlikely be useful for prediction.
```
df = df.drop(columns=['ID'])
```

#### VI. Splitting the data into train and test sets
```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
```

#### VII. Train a model
For classification task we can use ML algorithms like **Logistic Regression**, **Random Forest**, or **XGBoost**.
Here we are using `Random Forest`.
```
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
```

#### VIII. Evaluate the Model
Evaluation metrics used are:
* Precision, Recall, F1-Score
* ROC-AUC Score
* Confusion Matrix

```
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

#### IX. Result

![alt text](<accuracy metrics-1.png>)

#### X. Result Interpretation
1. Since the `f1-score` is `0.98`, which indicates good classifier.
2. **High Precision (0.98 for class 1)** indicates loan approval (class 1) is correct **98% of the time** and **2% of the approval loans are actually not eligible**.

    **Implication for Banking** high precision is desirable because it minimizes the risk of approving loans for customers who are likely to default. This reduces financial losses for the bank.

3. **High recall (0.99 for class 1)** means **99% of the actual loan approvals**, only **1% eligible customers are incorrectly denied loans**.

    **Implication for Banking** high recall is also desirable because it minimizes the risk of rejecting loans for customers who are actually eligible. This ensures that the bank does not miss out potential revenue from good customers.

4. **High Accuracy (0.98)** means **98% of all predictions are correct**.
5. **High ROC-AUC Score (0.998)** indicates that the model has an excellent ability to distinguish between loan approvals and rejections.


*Note:*

*Precison: measures the proportion of correctly predicted positive instances out of all instances predicted as positive.*

*Recall (Sensitivity): measures the proportion of actual positive instances that are correctly predicted as positive.*

### 3. Model Serving
The model is trained using `pipeline`, once the model is trained, we have saved the model using `joblib` in `.pkl` file format.

We have used `FastAPI` to serve the model, which is implemeted in the `model_serving.py` file. 

Front-end to predict is implemented in the `index.html` file, but it is not working properly, we are working on it. In fact, the model API is working in the `localhost:8000/docs`,
where we have to manually type the customer features.
