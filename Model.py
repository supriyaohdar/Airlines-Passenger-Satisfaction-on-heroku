import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from numpy import set_printoptions
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.feature_selection import SelectKBest, RFE
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

import pickle

print("--------------")

traindata=pd.read_csv("train.csv")
testdata=pd.read_csv("test.csv")

print(traindata.head())
print(testdata.head())
print(traindata.shape)
print(testdata.shape)
print(traindata.info())
print(testdata.info())

print(traindata.describe(include="all"))
print(testdata.describe(include="all"))

traindata["Arrival Delay in Minutes"].fillna((traindata["Arrival Delay in Minutes"].mean()), inplace = True)
testdata["Arrival Delay in Minutes"].fillna((testdata["Arrival Delay in Minutes"].mean()), inplace = True)

traindata.drop(columns=[ 'Unnamed: 0','id'], axis=1, inplace=True)
testdata.drop(columns=[ 'Unnamed: 0','id'], axis=1, inplace=True)

#Label Encoding
set(traindata['Gender']) == set(testdata['Gender'])

labelencoder = LabelEncoder()
traindata['Gender'] = labelencoder.fit_transform(traindata['Gender'])
testdata['Gender'] = labelencoder.transform(testdata['Gender'])

set(traindata['Customer Type']) == set(testdata['Customer Type'])

traindata['Customer Type'] = labelencoder.fit_transform(traindata['Customer Type'])
testdata['Customer Type'] = labelencoder.transform(testdata['Customer Type'])

set(traindata['Type of Travel']) == set(testdata['Type of Travel'])

traindata['Type of Travel'] = labelencoder.fit_transform(traindata['Type of Travel'])
testdata['Type of Travel'] = labelencoder.transform(testdata['Type of Travel'])

set(traindata['Class']) == set(testdata['Class'])

traindata['Class'] = labelencoder.fit_transform(traindata['Class'])
testdata['Class'] = labelencoder.transform(testdata['Class'])

set(traindata['satisfaction']) == set(testdata['satisfaction'])

traindata['satisfaction'] = labelencoder.fit_transform(traindata['satisfaction'])
testdata['satisfaction'] = labelencoder.transform(testdata['satisfaction'])

print(traindata.dtypes)

traindata['Arrival Delay in Minutes'] = traindata['Arrival Delay in Minutes'].astype('int64')

#Data Visualization
plt.figure(figsize=(20, 8))
sns.heatmap(traindata.corr(), annot=True)
#plt.show()

plt.figure(figsize=(20, 8))
sns.heatmap(testdata.corr(), annot=True)
#plt.show()

traindata.hist(figsize=(20,8))
#plt.show()

testdata.hist(figsize=(20,8))
#plt.show()

#FEATURE ENGINEERING
array = traindata.values
X = array[:, 0:22]
Y = array[:, 22]
x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.20, random_state=1)


scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)

#FEATURE SELECTION
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(normalizedX, Y)

set_printoptions(precision=3)
#print(fit.scores_)
features = fit.transform(normalizedX)

#DATA MODELLING
model = LogisticRegression()

model.fit(x_train, y_train)

predictions = model.predict(x_valid)

score = model.score(x_valid, y_valid)
print("score of the model is:", score)
acc = model.score(x_valid, y_valid)
print( "accuracy of the model is:",acc*100)


#using k fold
num_folds = 10
seed = 7
scoring = 'accuracy'

kfold = KFold(n_splits=num_folds, random_state=seed)
cv_results = cross_val_score(model, x_train, y_train, scoring=scoring, cv=kfold)
msg = '%f (%f)'%(cv_results.mean(), cv_results.std())
print(msg)
'''
steps = [('scaler', StandardScaler()),
         ('RFE', RFE(LogisticRegression(), 6)),
         ('lda', LogisticRegression())]

pipeline = Pipeline(steps)
pipeline.fit(x_train, y_train)
predictions = pipeline.predict(x_valid)
'''
pickle.dump(model,open('Model_1.pkl','wb'))
Model_1=pickle.load(open('Model_1.pkl','rb'))