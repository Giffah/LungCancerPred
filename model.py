import numpy as np
import pandas as pd

import pickle


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, r2_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
     
import warnings
warnings.filterwarnings("ignore")


# loading the dataset
df =pd.read_csv('cancer patient data sets.csv')


df.drop(columns=['index', 'Patient Id'], axis=1, inplace=True)
print(df.head())

# Replace "level" with Integer
print('Cancer Levels: ', df['Level'].unique())
df["Level"].replace({'High': 2, 'Medium': 1, 'Low': 0}, inplace=True)
print('Cancer Levels: ', df['Level'].unique())

# Data splitting
y = df.pop('Level')
x = df

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


## RandomForest 

from sklearn.model_selection import GridSearchCV
param_RF1 = {
    'n_estimators': 50, 
    'max_depth': 3,
    'min_samples_split': 3,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'max_samples': 0.8,
    'criterion': 'gini'    
}
param_RF = {
    'n_estimators': [50, 100, 150], 
    'max_depth': [2],
    'criterion':['gini'],
    'min_samples_split': [2, 3],  
    'min_samples_leaf': [2, 3],
    'random_state' : [42],
    'max_samples': [0.4]
    
}



model_tuning = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_RF, cv=5)
model_tuning.fit(x_train, y_train)

best_params = model_tuning.best_params_
print("Best Parameters:", best_params)
model_rf = RandomForestClassifier(**best_params)
model_rf.fit(x_train, y_train) 

# Making Pickle file

pickle.dump(model_rf,open('model.pkl','wb'))
