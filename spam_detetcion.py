#spam classfication using a simple neural network
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df=pd.read_csv('mail_data.csv')

data=df.where(pd.notnull(df), '')
data.loc[data['Category'] == 'spam', 'Category'] = 0
data.loc[data['Category'] == 'ham', 'Category'] = 1
X = data['Message']
y = data['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=3)
print (X.shape, y.shape)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

feature_extractor = TfidfVectorizer(min_df=5, max_df=0.7, stop_words='english', lowercase=True)
X_train_features=feature_extractor.fit_transform(X_train)
X_test_features=feature_extractor.transform(X_test)
y_train= y_train.astype('int')
y_test = y_test.astype('int')

model = LogisticRegression()
model.fit(X_train_features, y_train)
predictions_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(y_train, predictions_on_training_data)

print('Accuracy on training data: ', accuracy_on_training_data)
predictions_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(y_test, predictions_on_test_data)
print('Accuracy on test data: ', accuracy_on_test_data)

input_your_mail = ["this is a test mail, please ignore it", "Congratulations! You've won a lottery! Click here to claim your prize."]
input_your_mail_features = feature_extractor.transform(input_your_mail)
predictions_on_your_mail = model.predict(input_your_mail_features)
print (predictions_on_your_mail)

for mail, prediction in zip(input_your_mail, predictions_on_your_mail):
    if prediction == 1:
        print(f"'{mail}' → Your mail is **not spam**")
    else:
        print(f"'{mail}' → Your mail is **spam**")
