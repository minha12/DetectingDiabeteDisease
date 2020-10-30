# Preliminary

XGBoost will be used for this analysis.

## What is XGBoost? 

> XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. In prediction problems involving unstructured data (images, text, etc.) artificial neural networks tend to outperform all other algorithms or frameworks. However, when it comes to small-to-medium structured/tabular data, decision tree based algorithms are considered best-in-class right now. 


```python
!pip install xgboost
```

    Collecting xgboost
      Downloading xgboost-1.2.1-py3-none-macosx_10_13_x86_64.macosx_10_14_x86_64.macosx_10_15_x86_64.whl (1.2 MB)
    [K     |████████████████████████████████| 1.2 MB 3.9 MB/s eta 0:00:01
    [?25hRequirement already satisfied: scipy in /opt/miniconda3/lib/python3.7/site-packages (from xgboost) (1.4.1)
    Requirement already satisfied: numpy in /opt/miniconda3/lib/python3.7/site-packages (from xgboost) (1.18.1)
    Installing collected packages: xgboost
    Successfully installed xgboost-1.2.1


# Dataset

## Context

> This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

## Content

> The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

## Source: 

https://www.kaggle.com/uciml/pima-indians-diabetes-database


```python
import pandas as pd

dataset = pd.read_csv('diabetes.csv')

dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# Pre-processing dataset

- Split train/set dataset


```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```


```python
features = dataset.iloc[:, 0:8].values
labels = dataset.iloc[:,8].values
```


```python
features
```




    array([[  6.   , 148.   ,  72.   , ...,  33.6  ,   0.627,  50.   ],
           [  1.   ,  85.   ,  66.   , ...,  26.6  ,   0.351,  31.   ],
           [  8.   , 183.   ,  64.   , ...,  23.3  ,   0.672,  32.   ],
           ...,
           [  5.   , 121.   ,  72.   , ...,  26.2  ,   0.245,  30.   ],
           [  1.   , 126.   ,  60.   , ...,  30.1  ,   0.349,  47.   ],
           [  1.   ,  93.   ,  70.   , ...,  30.4  ,   0.315,  23.   ]])




```python
labels[0:5]
```




    array([1, 0, 1, 0, 1])



## Visualizing Outcome

Plotting percentage of Outcome 0 (no disease) vs 1 (disease)


```python
dataset.Outcome.value_counts().plot(kind='pie')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a26c3ed50>




![png](Analysis_files/Analysis_10_1.png)



```python
print(f'Percentage of No Disease: {100 * labels[labels==0].shape[0] / labels.shape[0]:0.2f}')
print(f'Percentage of Disease: {100 * labels[labels==1].shape[0] / labels.shape[0]:0.2f}')
```

    Percentage of No Disease: 65.10
    Percentage of Disease: 34.90


## Normalize data


```python
from sklearn.preprocessing import MinMaxScaler

# Scalling to range (-1, 1)
scaler=MinMaxScaler( (-1, 1) )
X = scaler.fit_transform(features)
#X = features
Y = labels
```


```python
X
```




    array([[-0.29411765,  0.48743719,  0.18032787, ...,  0.00149031,
            -0.53116994, -0.03333333],
           [-0.88235294, -0.14572864,  0.08196721, ..., -0.2071535 ,
            -0.76686593, -0.66666667],
           [-0.05882353,  0.83919598,  0.04918033, ..., -0.30551416,
            -0.49274125, -0.63333333],
           ...,
           [-0.41176471,  0.2160804 ,  0.18032787, ..., -0.21907601,
            -0.85738685, -0.7       ],
           [-0.88235294,  0.26633166, -0.01639344, ..., -0.10283159,
            -0.76857387, -0.13333333],
           [-0.88235294, -0.06532663,  0.14754098, ..., -0.09388972,
            -0.79760888, -0.93333333]])



## Splitting train/test sets


```python
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.33, random_state=0)
```


```python
x_train.shape
```




    (514, 8)




```python
x_test.shape
```




    (254, 8)



# Build and fit model


```python
model = XGBClassifier()
model.fit(x_train, y_train)
```




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                  importance_type='gain', interaction_constraints='',
                  learning_rate=0.300000012, max_delta_step=0, max_depth=6,
                  min_child_weight=1, missing=nan, monotone_constraints='()',
                  n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                  tree_method='exact', validate_parameters=1, verbosity=None)



# Prediction and evaluation


```python
y_pred = model.predict(x_test)
y_pred
```




    array([1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
           0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1,
           1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0,
           1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1,
           0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
           0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0,
           0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
           0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])




```python
# accuracy
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy score: {acc * 100:0.2f}')
```

    Accuracy score: 75.98


# Summary

What we have learned:
- Quick load and visualize dataset
- Using MinMaxScaler
- Using XGBoost


```python

```
