import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

# 1)
df = pd.read_csv("bank.csv",skiprows=0,delimiter=";")

# 2) Tein tämän aluksi käyttäen kolumnien indexejä, mutta tajusin myöhemmin, että voin käyttää filteriä. Filterin käyttäminen vaikuttaa mielestäni paremmalta vaihtoehdolta.
#df2 = df.iloc[:, [16,1,2,4,6,15]]
df2 = df.filter({"y", "job", "marital", "default", "housing", "poutcome"})
print(df2)

# 3) Dummien luominen
df3 = pd.get_dummies(data=df2, columns=['job','marital','default','housing','poutcome'])
print(df3)

# 4) Ohjelma ei ymmärtänyt y arvon 'yes' ja 'no' arvoja. Muutin ne booleaniksi joka ratkaisi tämän. Testasin lopun ehjelman toiminnan ilman y:n muutosta ja sen kanssa ja tulokset olivat samat. 
df3['y'] = df3['y'] == 'yes'
sns.heatmap(data=df3.corr(), annot=True)
plt.show()

# 5) 
X = df3.iloc[:, 1:-1].values
y = df3.iloc[:, 0].values

# 6) Treeni ja testi data 75/25 suhteessa
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# 7) Logistical Regression mallin luominen ja testaaminen
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 8) Confusion matrix ja sen accurary arvo
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

metrics.ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.show()

print("Logistic Regression Accuracy:", metrics.accuracy_score(y_test, y_pred))

# 9) K nearest neihgbors malli arvolla K=3 ja sen accurary arvo
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

metrics.ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test)
plt.show()

print("K Neighbour 3 Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Päätin kokeilla myös eri K arvoilla joten tein loopin joka kokeilee arvot 1-10
accuracy = {}
for i in range(1,11):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy[i] = metrics.accuracy_score(y_test, y_pred)
 
# 10) Tulostaa arvot allekkain ja tulostaa erikseen vielä parhaimman K arvon. Ohjelma tulostaa yläpuolelle myös Logistical Regression mallin tuloksen jolloin niiden vertaaminen on mahdollista
print('Accuracy using different K values:')
for i in accuracy:
    print(f"K{i}: {accuracy[i]}")
    
print(f"Best K value: K{max(accuracy, key=accuracy.get)} {max(accuracy.values())}")