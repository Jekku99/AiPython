from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("emails.csv")
print(df.head())

print(df.spam.value_counts())

X = df["text"]
y = df["spam"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=10)
print(X_test)
print(y_test)

vect = CountVectorizer(stop_words="english")
vect.fit(X_train)
X_train_df = vect.transform(X_train)
X_test_df = vect.transform(X_test)

model = svm.SVC()
model.fit(X_train_df,y_train)
y_pred = model.predict(X_test_df)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred,target_names=["not spam","spam"]))

y_test2 = np.array(y_test)
y_pred2 = np.array(y_pred)

idx = np.logical_and(y_pred2 == 0, y_test2 == 0)
spam0 = X_test[idx]
print("Not spam: ",np.array(spam0.index))
print("Not spam sample =",X_test[3886])
print("Not spam sample =",X_test[5613])

idx = np.logical_and(y_pred2 == 1, y_test2 == 1)
spam = X_test[idx]
print("spam: ",np.array(spam.index))
print("spam sample =",X_test[282])
print("spam sample =",X_test[225])