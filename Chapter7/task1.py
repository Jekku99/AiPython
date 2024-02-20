import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

#0
df = pd.read_csv("data_banknote_authentication.csv")

#1
X = df.drop('class', axis=1)
y = df['class']

#2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=20)

#3
print('Linear')
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)

#4
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

#5
print('Radial')
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

#6
# Kumpikin osaavat antaa hyvin tarkat ennusteet, mutta tässä tilanteessa rbf antaa hiukan paremman kuin linear. Lineaarinen antoi 2 kappaletta virheellisiä positiivia tuloksia kun taas rbf sai kaikki oikein.
# Ajoin ohjelman muutaman kerran käyttämättä random state muuttujaa ja jokaisella kerralla rbf ennustaa oikean tuloksen varmemmin. Joten tälle datalle rbf olisi parempi valinta.