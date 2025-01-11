import pandas as pd
from sklearn.model_selection import train_test_split

#load in data
data = pd.read_csv("Data/paySimDataset.csv")

'''
#print out number of transactions in dataset
print(len(data))

fraud = data[['isFraud']]
caughtFraud = data[['isFraud', 'isFlaggedFraud']]
#checking how many transactions in the dataset are caught as fraud
print(caughtFraud.query('isFraud == 1 and isFlaggedFraud == 1'))

#checking whether there is an amt for all transactions in dataset
transactionAmt = data[['amount']]
print(data.query('amount == 0'))
# This resulted to show that there's a few, they're all fraud

#checking for duplicates
data.drop_duplicates()
print(len(data))

print(data.dtypes)

#check for null:
nulls = data.select_dtypes(include=['object']).copy().isnull()
print(nulls)

# checking if customer is ever the same as the destination customer
sameCust = data[['nameOrig', 'nameDest']]
print(sameCust.query("nameOrig == nameDest")) # returns empty dataframe
# ie never the same, don't need in input data

'''

#Encoding data: (Test, since doesn't actually change the data, going to need to put in project code)

type = data[['type']]
print(type.value_counts())

data['type'] = data['type'].astype('category')
data['type'] = data['type'].cat.codes
print(data['type'].value_counts())

#print(data['isFraud'].value_counts())

X = data[['step', 'type', 'amount', 'oldbalanceOrg'
          , 'newbalanceOrig', 'isFlaggedFraud']]

# Make dataframe with feature class (output)
y = data['isFraud']

# Splitting Method Credit: https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
x, X_test, y, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)
X_train, X_validation, y_train, y_validation = train_test_split(x, y, test_size=0.25, train_size=0.75)

print(f"Training Set: X: {len(X_train)}, y: {len(y_train)}")
print(X_train.value_counts())
print(y_train.value_counts())
print("______________________________")
print(f"Validation Set: X: {len(X_validation)}, y: {len(y_validation)}")
print(X_validation.value_counts())
print(y_validation.value_counts())
print("______________________________")
print(f"Testing Set: X: {len(X_test)}, y: {len(y_test)}")
print(X_test.value_counts())
print(y_test.value_counts())

print("Done")