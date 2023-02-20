#import the pandas librarie and names it pd 
import pandas as pd
import arff
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset into a pandas dataframe
with open('diabetes.arff', 'r') as f:
    data = arff.load(f)
    df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])

# Extract the features and target variable
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train a Gaussian Naive Bayes classifier on the training set
clf = GaussianNB()
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the classifier
acc = accuracy_score(y_test, y_pred)
print('Accuracy:\n', acc)

# Print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', cm)
