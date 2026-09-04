# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the SMS dataset and separate the message text and spam/ham labels.

2.Convert the text into numerical features using TF-IDF.

3.Train an SVM classifier using the training data and evaluate its accuracy.

4.Predict whether a new message is Spam or Not Spam.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: DHAYAL ABISEK R
RegisterNumber:  212225060061  
*/
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(
    "/content/spam.csv",
    encoding="latin1"
)

# Select required columns
df = df[["v1", "v2"]]
df.columns = ["Label", "Message"]

# Input and output
X = df["Message"]
Y = df["Label"]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Convert text into numerical features
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Create SVM model
model = SVC(kernel="linear")

# Train model
model.fit(X_train, Y_train)

# Predict test data
Y_pred = model.predict(X_test)

# Accuracy
print("Accuracy =", accuracy_score(Y_test, Y_pred))

# Predict a new message
new_message = ["Congratulations! You have won a free prize."]
new_message = vectorizer.transform(new_message)

prediction = model.predict(new_message)

print("Prediction:", prediction[0])
```
## Output:
```
Accuracy = 0.9829596412556054
Prediction: spam
```
## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
