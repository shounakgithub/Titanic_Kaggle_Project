from sklearn.ensemble import RandomForestRegressor

#error metric. c-stat (aka ROC-AUC)
from sklearn.metrics import roc_auc_score

import pandas as pd

X = pd.read_csv('F:/Kaggle/Titanic Machine Learning Disaster/train.csv')
X_test= pd.read_csv('F:/Kaggle/Titanic Machine Learning Disaster/test.csv')

y = X.pop("Survived")

X.describe()

#taking care of the null values of Age
X.Age.fillna(X.Age.mean(), inplace = True)

#selecting variables with non object datatypes

#X.dtypes != 'object' --> returns all the vars that are not an object
numeric_variables = list(X.dtypes[X.dtypes != 'object'].index)
X[numeric_variables].head()


#model training building

model = RandomForestRegressor(n_estimators=100, oob_score=True,random_state=42)
model.fit(X[numeric_variables], y)

# Trailing underscores available after the model has been trained
# oob = out of bag
model.oob_score_ #calculates the R^2 value

y_oob = model.oob_prediction_ #y_oob--> every single observation has  a  prediction
print 'c_Stat: ', roc_auc_score(y,y_oob)


# So far, only numeric variables have been processed to get a rough estimate
#now, lets deal with categorical variables

def describe_categorical(X):
    from IPython.display import display, HTML
    display(HTML(X[X.columns[X.dtypes == "object"]].describe().to_html()))

#dropping unnecessary columns
X.drop(["Name","Ticket","PassengerId"], axis = 1, inplace = True)

#Deal with the categorical variable Cabin and shorten the values
def clean_cabin(x):
    try:
        return x[0]
    except TypeError:
        return "None"

X["Cabin"] = X.Cabin.apply(clean_cabin)
#Deal with the categorical variable Cabin and shorten the values

# Play with categorical variables and apply dummies
categorical_variables = ['Sex', 'Cabin', 'Embarked']

for variable in categorical_variables:
    X[variable].fillna("Missing", inplace = True)
    #Create Array of dummies
    dummies = pd.get_dummies(X[variable], prefix=variable)
    #Update X to include dummies
    X = pd.concat([X, dummies], axis=1)
    X.drop([variable],axis =1, inplace = True)
# Play with categorical variables and apply dummies

# now check the predicting capabilities of the refined model

model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42,n_jobs=-1)
model.fit(X,y )

y_oob = model.oob_prediction_
print 'c_Stat: ', roc_auc_score(y,y_oob) #checkingthe score


# Lets use Random Forest to help us with some exploratory data analysis mainly to find which variables are important in the model

model.feature_importances_ # this gives the weightage and effectiveness of each column for prediction purpose

# Making the feature selection easier

feature_importance = pd.Series(model.feature_importances_, index = X.columns)
feature_importance.sort()
feature_importance.plot(kind = 'barh')

##################################################################TEST ENVIRONMENT

X_test.Age.fillna(X.Age.mean(), inplace = True)

#selecting variables with non object datatypes

#X.dtypes != 'object' --> returns all the vars that are not an object
numeric_variables = list(X_test.dtypes[X.dtypes != 'object'].index)
X_test[numeric_variables].head()


#model training building

model = RandomForestRegressor(n_estimators=100, oob_score=True,random_state=42)
model.fit(X[numeric_variables], y)

# Trailing underscores available after the model has been trained
# oob = out of bag
model.oob_score_ #calculates the R^2 value

y_oob = model.oob_prediction_ #y_oob--> every single observation has  a  prediction
print 'c_Stat: ', roc_auc_score(y,y_oob)


# So far, only numeric variables have been processed to get a rough estimate
#now, lets deal with categorical variables

def describe_categorical(X_test):
    from IPython.display import display, HTML
    display(HTML(X_test[X_test.columns[X_test.dtypes == "object"]].describe().to_html()))

#dropping unnecessary columns
X.drop(["Name","Ticket","PassengerId"], axis = 1, inplace = True)

#Deal with the categorical variable Cabin and shorten the values
def clean_cabin(x):
    try:
        return x[0]
    except TypeError:
        return "None"

X_test["Cabin"] = X_test.Cabin.apply(clean_cabin)
#Deal with the categorical variable Cabin and shorten the values

# Play with categorical variables and apply dummies
categorical_variables = ['Sex', 'Cabin', 'Embarked']

for variable in categorical_variables:
    X_test[variable].fillna("Missing", inplace = True)
    #Create Array of dummies
    dummies = pd.get_dummies(X_test[variable], prefix=variable)
    #Update X to include dummies
    X_test = pd.concat([X_test, dummies], axis=1)
    X_test.drop([variable],axis =1, inplace = True)
# Play with categorical variables and apply dummies

# now check the predicting capabilities of the refined model
import numpy as np

model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42,n_jobs=-1)
model.fit(X_test,y[0:418] )

y_oob = model.oob_prediction_
print 'c_Stat: ', roc_auc_score(y[0:418],y_oob) #checkingthe score

feature_importance = pd.Series(model.feature_importances_, index = X_test.columns)