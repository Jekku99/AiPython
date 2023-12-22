import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

df = pd.read_csv("bank.csv",skiprows=0,delimiter=";")

df2 = df.iloc[:, [16,1,2,4,6,15]]
print(df2)

df3 = pd.get_dummies(data=df2, columns=['job','marital','default','housing','poutcome'])
print(df3)

#sns.heatmap(data=df3.corr(), annot=True)
#plt.show()

X = df3.iloc[:, 1:-1].values
y = df3.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

metrics.ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.show()

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

metrics.ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test)
plt.show()

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


accuracy = {}
for i in range(10):
    classifier = KNeighborsClassifier(n_neighbors=i+1)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy[i+1] = metrics.accuracy_score(y_test, y_pred)
    
for i in accuracy:
    print(i, accuracy[i])