# 4482-Final-project
Final project for 4482 - Countermeatures to Financial Crime

The goal of this project is to analyze the performance of two machine learning models trained using an imbalanced dataset to detect fraud in mobile transactions. This was completed using Python's Sklearn library and a dataset of synthetic mobile transactions from Paysim, sourced from Kaggle, a website that hosts code and datasets for machine learning purposes. 
The dataset was heavily imbalanced, the majority of transactions were not fraudulent, requiring the use of oversampling and undersampling techniques to fix this. These techniques were then compared to the unchanged dataset to analyze which methods were the most effective.

Random Forest Classification:
Random forest classification is a machine learning model that utilizes the average results of multiple decision trees to determine the class of a sample. To do this, it creates an n number of decision trees, each with their own permutation to improve the accuracy of its output. 
This model was chosen over traditional decision trees as it includes the consensus of multiple trees instead of just one. This limits the bias of a single decision tree, as each tree created for the random forest classifier could have different results depending on the way the nodes are split.
Support Vector Machine  (SVM):
Support vector machine is an older machine learning model that aims to split the samples into two hyperplanes (fraudulent and non-fraudulent transactions) using a decision boundary which helps to determine the class of samples. This decision boundary has a margin that can be larger or smaller to include more samples that are less obvious, aiming to reduce the number of misclassifications. 
This model was chosen for its good performance over large datasets and datasets with high dimensionality (ie. lots of features), both qualities of the dataset used to train and test the model. Support vector machines are also commonly used because they are easily understandable and quick. 
More specifically, the LinearSVC model was used for its better performance speed in training.

Results: 
Overall, the random forest classification model outperformed the support vector machine in all aspects. The random forest model with the best results was the one trained with combined oversampled and undersampled data, showing the best ratio of false positives to true positives and false negatives - not the best at truly catching all fraud, but the best ratio of the predictions. Of the Support vector machine models, the undersampled model performed the best, catching the most amount of fraud. It also had the best proportion of correct fraud predictions over all fraud predictions even if it was predicting more false positives than fraud itself. 
