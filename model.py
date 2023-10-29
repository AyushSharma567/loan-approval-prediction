import numpy as np
import pandas as pd
import missingno as msno
from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

df = pd.read_csv("loan_approval_dataset.csv")

##..................... feature engineering .......................

columns_to_remove = ['loan_id']
df.drop(columns=columns_to_remove, inplace=True) # Remove the specified columns

# Movable Assets
df['Movable_assets'] = df[' bank_asset_value'] + df[' luxury_assets_value']

#Immovable Assets
df['Immovable_assets'] = df[' residential_assets_value'] + df[' commercial_assets_value']

df.drop(columns=[' bank_asset_value',' luxury_assets_value', ' residential_assets_value', ' commercial_assets_value' ], inplace=True)

# select all categorical data type and stored in one dataframe and select all other numarical and stored in one data frame
catvars = df.select_dtypes(include=['object']).columns
numvars = df.select_dtypes(include = ['int32','int64','float32','float64']).columns

##................... Data Cleaning ..............................

#as we can see  from the loan_approval.ipynb we dont have any missing values in the dataset to perform data cleaning .

##................... Data Preprocessing .........................

# Label Encoding
df[' education'] = df[' education'].map({' Not Graduate':0, ' Graduate':1})
df[' self_employed'] = df[' self_employed'].map({' No':0, ' Yes':1})
df[' loan_status'] = df[' loan_status'].map({' Rejected':0, ' Approved':1})

##................... Machine Learning model .....................

# test-train split

X_train, X_test, y_train, y_test = train_test_split(df.drop(' loan_status', axis=1), df[' loan_status'], test_size=0.2, random_state=42)

dtree = DecisionTreeClassifier() # Create decision tree object
dtree.fit(X_train, y_train) # Trainign the model using the training data
dtree_pred = dtree.predict(X_test)
dtree.score(X_train, y_train)

pickle.dump(dtree,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

df=pd.read_pickle("model.pkl")
print(df)

