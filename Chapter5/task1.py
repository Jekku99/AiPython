import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = load_diabetes(as_frame=True)
print(data)
print(data.keys())
print(data.DESCR)

df = data.frame
print(df.head())

plt.hist(df["target"],25)
plt.xlabel("target")
plt.show()

sns.heatmap(data=df.corr().round(2), annot=True)
plt.show()

plt.subplot(1,3,1)
plt.scatter(df['bmi'], df['target'])
plt.xlabel('bmi')
plt.ylabel('target')

plt.subplot(1,3,2)
plt.scatter(df['s5'], df['target'])
plt.xlabel('s5')
plt.ylabel('target')

plt.subplot(1,3,3)
plt.scatter(df['bp'], df['target'])
plt.xlabel('bp')
plt.ylabel('target')
plt.show()

X = pd.DataFrame(df[['bmi','s5','bp','s1','s6']], columns = ['bmi','s5','bp','s1','s6'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5)

lm = LinearRegression()
lm.fit(X_train, y_train)

y_train_predict = lm.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)

y_test_predict = lm.predict(X_test)
rmse_test = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2_test = r2_score(y_test, y_test_predict)

print(rmse,r2)
print(rmse_test,r2_test)

"""
a) Lisäsin muuttujan bp, koska sillä oli seuraavaksi suurin korrelaatio target arvon kanssa.

b) RMSE ja R2 arvot paranivat hiukan, mutta uudetkaan tulokset eivät ole hirveän luotettavia
vanhat:
RMSE = 56.560890965481114, R2 = 0.4507519215172524
RMSE (test) = 57.1759740950605, R2 (test)= 0.4815610845742896

uudet:
RMSE = 55.32610611166316, R2 = 0.47447150038132146
RMSE (test) = 56.6256100515053, R2 (test)= 0.4914938186648421

c) S1 muuttujan lisääminen parantaa tuloksia hiukan lisää
RMSE = 54.96124895994681, R2 = 0.48138001446445355
RMSE (test) = 55.07934439478595, R2 (test)= 0.518886023755458

Jotkin toiset muuttujat auttoivat myös mutta ei yhtä paljon kuin S1. 

Vielä kun lisäsin S6 muuttujan tulos parani taas hieman.
RMSE = 54.94704845825654, R2 = 0.48164797462567477
RMSE (test) = 54.93538974081675, R2 (test)= 0.5213976037333918

Muiden muuttujien lisääminen ei vaikuttanut tuloksiin juuri ollenkaan tai tekivät niistä aiempia huonompia.
"""