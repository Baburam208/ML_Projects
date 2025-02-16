## This is a internet service churn prediction problem

The dataset is collected from the kaggle. <https://www.kaggle.com/datasets/mehmetsabrikunt/internet-service-churn>

The datset snippet is
![alt text](<assets/dataset snippet.png>)


The dataset has shape of: `(72274, 11)`

**We got best accuracy of 95 % using `RandomForestClassifier` model.**

### Dataset Inspection
**Missing rows count in each columns are:**
```
id                                 0
is_tv_subscriber                   0
is_movie_package_subscriber        0
subscription_age                   0
bill_avg                           0
reamining_contract             21572
service_failure_count              0
download_avg                     381
upload_avg                       381
download_over_limit                0
churn                              0
```

**columns data types:**
```
#   Column                       Non-Null Count  Dtype  
---  ------                       --------------  -----  
 0   id                           72274 non-null  int64  
 1   is_tv_subscriber             72274 non-null  int64  
 2   is_movie_package_subscriber  72274 non-null  int64  
 3   subscription_age             72274 non-null  float64
 4   bill_avg                     72274 non-null  int64  
 5   reamining_contract           50702 non-null  float64
 6   service_failure_count        72274 non-null  int64  
 7   download_avg                 71893 non-null  float64
 8   upload_avg                   71893 non-null  float64
 9   download_over_limit          72274 non-null  int64  
 10  churn                        72274 non-null  int64  
```

**Column Values**
In columns: `is_tv_subscriber`, `is_movie_package_subscriber` and `churn` have binary value (1 or 0).

### Data Cleaning
**1.**
Renaming the column -> `reamining_contract` to `remaining_contract`.
```
df.rename(columns={"reamining_contract": "remaining_contract"}, inplace=True)
```

**2.**
Column `subscription_age` contains negative value, I think which should be a typo or error. So let's delete that row(s).
```
# Drop rows where subscription_age < 0
df = df.drop(df[df['subscription_age'] < 0].index)
```

### Dealing with Outliers
The histogram plot of the `download_avg` seems so skewed, which indicates the probable outliers values. So we remove the outlier values beyond the upper bound of `Q3+IQR`.

![alt text](<assets/download.png>)

we can also see this in the box plot.

![alt text](<assets/download2.png>)

code to remove outliers in the column `download_avg` is:
```
# Calculate Q1, Q3, and IQR
Q1 = df['download_avg'].quantile(0.25)  # 25th percentile
Q3 = df['download_avg'].quantile(0.75)  # 75th percentile
IQR = Q3 - Q1  # Interquartile Range

# Define the upper threshold
upper_threshold = Q3 + 1.5 * IQR

# Filter out rows where 'service_failure_count' exceeds the upper threshold
df = df[df['download_avg'] <= upper_threshold]
```


Similarly for the columns `upload_avg` and `bill_avg` the histogram plot is so skewed, and we remove outliers similar to the above case.

After removing the outliers we have the combined box plot as follows:

![alt text](<assets/download3.png>)

The box plot and values seems okay to go further analysis.

### label
The label `churn` column class value counts seems okay (in fact, there is little imbalanced in the class value count).

```
churn
1    36332
0    26459
```

### Final dataset size
After performing outlier removal we have now `(62791, 11)` size dataset.

### ML model design

Some important libraries to import.
```
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
```

Let's first discuss about the feature standardization.

we have following columns:
```
feature_columns = ['is_tv_subscriber', 'is_movie_package_subscriber',
                   'subscription_age', 'bill_avg', 'remaining_contract',
                   'service_failure_count', 'download_avg', 'upload_avg',
                   'download_over_limit']
```
among them columns to standardize are:
```
column_2_normalize = ['subscription_age', 'bill_avg', 'remaining_contract', 'service_failure_count', 'download_avg', 'upload_avg', 'download_over_limit']
```
and the target label is
```
target = ['churn']
```

We performed data standardization as follows using the `ColumnTransformer`
```
standard_scaler = StandardScaler()

X = df[feature_columns]
y = df[target].values.ravel()

# Create the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), column_2_normalize),
        ('passthrough', 'passthrough', [col for col in feature_columns if col not in column_2_normalize])
    ]
)

# Fit and transform the data
X_scaled = preprocessor.fit_transform(df)
```

Train test split is done as follows.
```
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=True)
```

#### 1. LogisticRegression
```
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()

lr_model.fit(X_train, y_train)
```

and the evaluation step is as follows:
```
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred_lr = lr_model.predict(X_test)

print(f"Classification report:")
print(classification_report(y_test, y_pred_lr))
print(f"ROC-AUC Score: ")
print(roc_auc_score(y_test, y_pred_lr))
print(f"Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred_lr))

```

The evaluation metrics of the `LogisticRegression` model is as follows:
```
Classification report:
              precision    recall  f1-score   support

           0       0.89      0.82      0.85      5318
           1       0.88      0.92      0.90      7241

    accuracy                           0.88     12559
   macro avg       0.88      0.87      0.88     12559
weighted avg       0.88      0.88      0.88     12559

ROC-AUC Score: 
0.8731078883623036
Confusion Matrix: 
[[4387  931]
 [ 570 6671]]
```

Similarly,

#### 2. SVM
The evaluation metrics report is as follows:
```
Classification report:
              precision    recall  f1-score   support

           0       0.91      0.89      0.90      5318
           1       0.92      0.93      0.93      7241

    accuracy                           0.92     12559
   macro avg       0.91      0.91      0.91     12559
weighted avg       0.92      0.92      0.92     12559

ROC-AUC Score: 
0.9121106181584028
Confusion Matrix: 
[[4735  583]
 [ 479 6762]]
```

#### 3. DecisionTreeClassifier
```
Classification report:
              precision    recall  f1-score   support

           0       0.90      0.90      0.90      5318
           1       0.92      0.93      0.92      7241

    accuracy                           0.91     12559
   macro avg       0.91      0.91      0.91     12559
weighted avg       0.91      0.91      0.91     12559

ROC-AUC Score: 
0.9103180517070406
Confusion Matrix: 
[[4760  558]
 [ 539 6702]]
```

#### 4. RandomForestClassifier
```
Classification report:
              precision    recall  f1-score   support

           0       0.93      0.95      0.94      5318
           1       0.96      0.94      0.95      7241

    accuracy                           0.95     12559
   macro avg       0.94      0.95      0.95     12559
weighted avg       0.95      0.95      0.95     12559

ROC-AUC Score: 
0.9467486424381573
Confusion Matrix: 
[[5052  266]
 [ 409 6832]]
```
#### Conclusion

**Using these model we got best accuracy in `RandomForestClassifier` of 95 %.**

**The f1-score for both label is quite close, however precision for class `0` is low (.93) compared to class 1 (.96). And recall is almost similar .95 and .94 for respective classes.**

#### Optional

On computing the feature importance for the best model i.e., `RandomForestClassifier`.
```
# Visualize the feature importance in horizontal bar graph.
import matplotlib.pyplot as plt
import seaborn as sns

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Create the plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, hue='Feature', palette='viridis')

# Add labels and title
plt.title('Feature Importance in RandomForestClassifier', fontsize=16)
plt.xlabel('Importance Score', fontsize=14)
plt.ylabel('Feature', fontsize=14)

# Add gridlines for better readability
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Show the plot
plt.show()
```
Visualization is as follows:

![alt text](<assets/download4.png>)

**Clearly, we can see `subscription_age` contributes lots to churn prediction.**

Least contributing feature sets are: `bill_avg` and `download_avg`.
