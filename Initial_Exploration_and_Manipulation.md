#### Initial Exploration and Manipulation

```python
#Importing required packages.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
%matplotlib inline
```

```python
import os
os.chdir('C:/Users/Abhinav.Arora/Downloads/Predicting Quality of Wine')
os.getcwd()
```

```python
#Loading dataset
wine = pd.read_csv('winequality-red.csv')
```

```python
#Let's check how the data is distributed
wine.head()
```

```python
#Information about the data columns
wine.info()
```

```python
#Here we see that fixed acidity does not give any specification to classify the quality.
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = wine)
```

```python
#Here we see that its quite a downing trend in the volatile acidity as we go higher the quality 
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = wine)
```

```python
#Composition of citric acid go higher as we go higher in the quality of the wine
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'citric acid', data = wine)
```

```python
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'residual sugar', data = wine)
```

```python
#Composition of chloride also go down as we go higher in the quality of the wine
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'chlorides', data = wine)
```

```python
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = wine)
```

```python
#Alcohol level also goes higher as te quality of wine increases
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'alcohol', data = wine)
```

#### Preprocessing Data for performing Machine learning algorithms 

```python
#Making binary classificaion for the response variable.
#Dividing wine as good and bad by giving the limit for the quality
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)
```

```python
#Now lets assign a labels to our quality variable
label_quality = LabelEncoder()
```

```python
#Bad becomes 0 and good becomes 1 
wine['quality'] = label_quality.fit_transform(wine['quality'])

wine['quality'].value_counts()
```

```python
sns.countplot(wine['quality'])
```

```python
#saving the dataset
wine.to_csv('wine-quality-prediction-dataset.csv', index = None, header=True)
```

```python

```
