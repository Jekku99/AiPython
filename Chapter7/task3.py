import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

#0
df = pd.read_csv('suv.csv')

#1
X = df[['Age', 'EstimatedSalary']]
y = df[['Purchased']]

#2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#3
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

#4
classifier = DecisionTreeClassifier(criterion="entropy")
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#5
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#6
classifier = DecisionTreeClassifier(criterion="gini")
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#7
# Mallit antavat useasti saman tuloksen, mutta silloin kun niissä on eroa entropy antaa useammin paremman tuloksen.