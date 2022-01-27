#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Load library 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings 
warnings.simplefilter('ignore')


# # Load the Dataset

# In[2]:


data = pd.read_csv('Cancer_data.csv')

pd.options.display.max_columns = 100


# In[3]:


data.head()


# In[4]:


data.shape


# # Data Analysis

# In[5]:


data.info()


# In[6]:


data.isna().any()


# In[7]:


data.isna().sum()


# In[8]:


data = data.dropna(axis = 'columns')


# In[9]:


data.diagnosis.value_counts()


# As we can see abouve result there are only one single feature is categorical and it's values are B and M

# # Data Visualization

# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')


# In[11]:


plt.figure(figsize=(7,8))

sns.countplot(data.diagnosis)
plt.title('Count of Diagnosis')
plt.xlabel('Diagnosis')
plt.show()


# In[12]:


cols = ["diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean"]

sns.pairplot(data[cols], hue = 'diagnosis')
plt.show()


# In[13]:


size = len(data['texture_mean'])

area = np.pi * (15 * np.random.rand( size ))** 2
colors = np.random.rand( size )

plt.xlabel('texture mean')
plt.ylabel('radius mean')
plt.scatter(data['texture_mean'], data['radius_mean'], s = area, c = colors, alpha = 0.5)


# # Data Filtering

# Now, we have one categorical feature, so we need to convert it into numeric values using LabelEncoder from sklearn.preprocessing packages

# In[14]:


from sklearn.preprocessing import LabelEncoder


# In[15]:


LE = LabelEncoder()
data.diagnosis = LE.fit_transform(data.diagnosis)


# In[16]:


data.head()


# In[17]:


data.diagnosis.value_counts()


# We can see in this output categorical values converted into 0 and 1.

# Find the correlation between other features, mean features only

# In[18]:


cols = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']


# In[19]:


plt.figure(figsize=(12, 9))

plt.title("Correlation Graph")
sns.heatmap(data[cols].corr(), annot = True)
plt.show()


# # Model Implementation

# In[20]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


# In[21]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.svm import SVC
from sklearn import metrics


# # Feature Selection
Attribute Information:

ID number
Diagnosis (M = malignant, B = benign)
Ten real-valued features are computed for each cell nucleus:

radius (mean of distances from center to points on the perimeter)
texture (standard deviation of gray-scale values)
perimeter
area
smoothness (local variation in radius lengths)
compactness (perimeter^2 / area - 1.0)
concavity (severity of concave portions of the contour)
concave points (number of concave portions of the contour)
symmetry
fractal dimension ("coastline approximation" - 1)
# In[22]:


data.columns


# In[23]:


prediction_feature = ['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
targeted_feature = 'diagnosis'


# In[24]:


X = data[prediction_feature]
y = data['diagnosis']


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=15)


# # Perform Feature Standerd Scalling¶

# In[26]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# # ML Model Selecting and Model PredPrediction

# In[27]:


def model_building(model):
    model_name = name
    model.fit(X_train, y_train)
    score = model.score(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(predictions, y_test)
    
    return  print('model name:', model_name,'\n' 'score:' ,score,'accuracy:', accuracy)
            


# In[28]:


models_list = {
    "LogisticRegression" :  LogisticRegression(),
    "RandomForestClassifier" :  RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5),
    "DecisionTreeClassifier" :  DecisionTreeClassifier(criterion='entropy', random_state=0),
    "SVC" :  SVC(),
}


# In[29]:


for name, model in zip(list(models_list.keys()), list(models_list.values())):
    model_building(model)


# # Classification Report

# In[30]:


def report_bulding(model):
        model_name = name
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        class_report = classification_report(y_test, predictions)
        return print('model name:', model_name, '\n''Classification report  :' , class_report)


# In[31]:


for name, model in zip(list(models_list.keys()), list(models_list.values())):
    report_bulding(model)


# # Crose Validation Score

# In[32]:


def cross_val_scorring(model):
        
    
    model.fit(data[prediction_feature], data[targeted_feature])
    predictions = model.predict(data[prediction_feature])    
    accuracy = accuracy_score(predictions, data[targeted_feature])
    print("\nFull-Data Accuracy:", round(accuracy, 2))
    print("Cross Validation Score of'"+ str(name), "'\n")
    
    kFold = KFold(n_splits=5) # define 5 diffrent data folds
    
    err = []
    
    for train_index, test_index in kFold.split(data):
        
        X_train = data[prediction_feature].iloc[train_index, :] # train_index = rows and all columns for Prediction_features
        y_train = data[targeted_feature].iloc[train_index] # all targeted features trains
        
        X_test = data[prediction_feature].iloc[test_index, :] # testing all rows and cols
        y_test = data[targeted_feature].iloc[test_index] # all targeted tests
        
        model.fit(X_train, y_train)

        err.append(model.score(X_train, y_train))
        
        print("Score:", round(np.mean(err),  2) )


# In[33]:


for name, model in zip(list(models_list.keys()), list(models_list.values())):
    cross_val_scorring(model)


# Some of the model are giving prefect scorring. it means sometimes overfitting occurs

# # HyperTunning the ML Model

# In[34]:


from sklearn.model_selection import GridSearchCV


# Apply to DecisionTreeClassifier Model

# In[35]:


model = DecisionTreeClassifier()


param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
              'min_samples_split': [2,3,4,5,6,7,8,9,10], 
              'min_samples_leaf':[2,3,4,5,6,7,8,9,10] }



gsc = GridSearchCV(model, param_grid, cv=10) 

gsc.fit(X_train, y_train)

print("\n Best Score is ")
print(gsc.best_score_)

print("\n Best Estinator is ")
print(gsc.best_estimator_)

print("\n Best Parametes are")
print(gsc.best_params_)


# Apply to KNeighborsClassifier Model

# In[36]:


model = KNeighborsClassifier()

param_grid = {'n_neighbors': list(range(1,30)),
              'leaf_size': list(range(1,30)), 
              'weights':['distance','uniform'] }

gsc = GridSearchCV(model, param_grid, cv= 10)

gsc.fit(X_train, y_train)

print("\n Best Score is ")
print(gsc.best_score_)

print("\n Best Estinator is ")
print(gsc.best_estimator_)

print("\n Best Parametes are")
print(gsc.best_params_)


# Apply to SVC Model

# In[37]:


model = SVC()

param_grid = [
              {'C': [1, 10, 100, 1000], 
               'kernel': ['linear']
              },
              {'C': [1, 10, 100, 1000], 
               'gamma': [0.001, 0.0001], 
               'kernel': ['rbf']
              }
]

gsc = GridSearchCV(model, param_grid, cv= 10)

gsc.fit(X_train, y_train)

print("\n Best Score is ")
print(gsc.best_score_)

print("\n Best Estinator is ")
print(gsc.best_estimator_)

print("\n Best Parametes are")
print(gsc.best_params_)


# Apply to RandomForestClassifier Model

# In[38]:


model = RandomForestClassifier()



random_grid = {'bootstrap': [True, False],
 'max_depth': [40, 50, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2],
 'min_samples_split': [2, 5], 
 'n_estimators': [200, 400]} 


gsc = GridSearchCV(model, random_grid, cv=10) 


gsc.fit(X_train, y_train)

print("\n Best Score is ")
print(gsc.best_score_)

print("\n Best Estinator is ")
print(gsc.best_estimator_)

print("\n Best Parametes are")
print(gsc.best_params_)


# # Observation

# So finally we have built our classification model and we can see that K Neighbors Classifier algorithm gives the best results for our dataset. 

# In[40]:





# In[41]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




