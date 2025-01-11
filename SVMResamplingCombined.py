import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, precision_score, accuracy_score, recall_score, f1_score, \
    precision_recall_curve, PrecisionRecallDisplay, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

#load in data
data = pd.read_csv("Data/paySimDataset.csv")

# encode the type feature - label encoding
data['type'] = data['type'].astype('category')
data['type'] = data['type'].cat.codes

# Make dataframe that holds the transaction data (input)
# (minus old/new balance destination because often empty if with merchant)
X = data[['step', 'type', 'amount', 'oldbalanceOrg'
          , 'newbalanceOrig', 'isFlaggedFraud']]

# Make dataframe with feature class (output)
y = data['isFraud']

# Split data into training, validation and testing
# First split out the 20% for testing, then into 20% validation and 60% training
# Splitting Method Credit: https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
x, X_test, y, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)
X_train, X_validation, y_train, y_validation = train_test_split(x, y, test_size=0.25, train_size=0.75)

#Apply oversampling
smote = SMOTE(sampling_strategy=0.5, random_state=123)
X_overSampled, y_overSampled = smote.fit_resample(X_train, y_train)

#Apply under-sampling:
rus = RandomUnderSampler(random_state=123)
X_resampled, y_resampled = rus.fit_resample(X_overSampled, y_overSampled)

#Create model with regularization 0.1
svm = LinearSVC(C=0.1, random_state=123)
svm.fit(X_resampled, y_resampled) # fit model with resampled data

y_pred = svm.predict(X_validation) #predict validation data
y_testPred = svm.predict(X_test) # predict test data

#Create confusion matrix
confusionMatrix = confusion_matrix(y_test, y_testPred)

# Plot the confusion matrix
ConfusionMatrixDisplay(confusionMatrix).plot()
plt.savefig('SVCconfMatrixcombined.png') #save to files

# Create and plot precision recall table
precision, recall, threshold= precision_recall_curve(y_testPred, y_test)
prRecallDisplay = PrecisionRecallDisplay(precision=precision, recall=recall).plot()
plt.savefig("svccombinedprc") #save to files

#Outputs:
print("---------------------------------------")
print("Training data distribution")
print(y_resampled.value_counts())
print("---------------------------------------")
print("Validation data:")
print(classification_report(y_pred, y_validation))
print("---------------------------------------")
print(f"Score for validation data:{svm.score(X_validation, y_validation)}")
print("---------------------------------------")
print ("Test data confusion Matrix:")
print(confusionMatrix)
print("---------------------------------------")
print(f"Precision: {precision_score(y_testPred, y_test)}")
print(f"Accuracy: {accuracy_score(y_testPred, y_test)}")
print(f"Recall: {recall_score(y_testPred, y_test)}")
print(f"F1: {f1_score(y_testPred, y_test)}")
print(f"False Negative Rate: {1-recall_score(y_testPred, y_test)}")
print(f"False Alert Rate: {1-precision_score(y_testPred, y_test)}")

print("done")
