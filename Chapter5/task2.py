import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# 0)
df = pd.read_csv('50_Startups.csv',delimiter=",")

# 1)
print(df.keys())

df['State']=df['State'].astype('category').cat.codes
# Halusin näyttää myös State muuttujan heatmapissa. State muuttuja ei vaikuttanut tuloksiin juuri mitenkään, joten olisin voinut myös jättää sen pois csv tiedostoa lukiessa.

# 2)
sns.heatmap(data=df.corr().round(2), annot=True)
plt.show()

# 3) Valitsin R&D Spend ja Marketing Spend koska näiden muuttujien korrelaatiot olivat korkeimmat voittoon verrattuna.

# 4)
plt.subplot(1,2,1)
plt.scatter(df['R&D Spend'], df['Profit'])
plt.xlabel('R&D Spend')
plt.ylabel('Profit')

plt.subplot(1,2,2)
plt.scatter(df['Marketing Spend'], df['Profit'])
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.show()

# 5)
X = pd.DataFrame(df[['R&D Spend','Marketing Spend']], columns = ['R&D Spend','Marketing Spend'])
y = df['Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# 6)
lm = LinearRegression()
lm.fit(X_train, y_train)

# 7)
y_train_predict = lm.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)

y_test_predict = lm.predict(X_test)
rmse_test = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2_test = r2_score(y_test, y_test_predict)

print('RMSE =',rmse,'R2 =',r2)
print('RMSE (test) =',rmse_test,'R2 (test) =',r2_test)
