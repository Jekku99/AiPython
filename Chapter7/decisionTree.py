from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd
import graphviz

df = pd.read_csv("data_banknote_authentication.csv")
print(df.head())

X = df.drop('class', axis=1)
y = df['class']
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=11)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

dot_data = tree.export_graphviz(classifier, out_file=None,
            feature_names = X_train.columns,class_names = None,
            filled = True, rounded = True,special_characters = True)
graph = graphviz.Source(dot_data)
#graph.render("dtree")

classifier = DecisionTreeClassifier(criterion="entropy",max_depth=4)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

dot_data = tree.export_graphviz(classifier, out_file=None,
            feature_names = X_train.columns,class_names = None,
            filled = True, rounded = True,special_characters = True)
graph = graphviz.Source(dot_data)
graph.render("dtree")