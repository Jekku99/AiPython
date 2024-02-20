import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import neighbors

#0
df = pd.read_csv('weight-height.csv')

#1
X = 2.54*df[["Height"]]
y = 0.45359237*df[["Weight"]]

#2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#3
X_train_norm = MinMaxScaler().fit_transform(X_train)
X_test_norm = MinMaxScaler().fit_transform(X_test)
X_train_std = StandardScaler().fit_transform(X_train)
X_test_std = StandardScaler().fit_transform(X_test)

#4
lm = neighbors.KNeighborsRegressor(n_neighbors=5)
lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
print("R2 =",lm.score(X_test,y_test))

#5
lm.fit(X_train_norm, y_train)
print("R2 (norm) =",lm.score(X_test_norm,y_test))

#6
lm.fit(X_train_std, y_train)
print("R2 (std) =",lm.score(X_test_std,y_test))

#7
# Kaikki antoivat useimmilla kerroilla hyvin samankaltaisia tuloksia. R2 arvot pyörivät 0.77-0.84 välillä. Yleensä parhaimman tuloksen antoi Standard data ja huonoimman Normalized.