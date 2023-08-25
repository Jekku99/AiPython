import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn import linear_model

# 1)
df = pd.read_csv('Auto.csv')

# 2)
X = pd.DataFrame(df[['cylinders','displacement','horsepower','weight','acceleration','year']], columns = ['cylinders','displacement','horsepower','weight','acceleration','year'])
y = df['mpg']

# 3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5)

# 4)
#Ridge
alphas = np.linspace(0,300,300)
r2values = {}
for alp in alphas:
    rr = linear_model.Ridge(alpha=alp)
    rr.fit(X_train, y_train)
    r2_test = r2_score(y_test, rr.predict(X_test))
    r2values[alp] = r2_test   
maxr2 = max(r2values, key=r2values.get)

#Lasso
alphas2 = np.linspace(0.01,2,300)
scores = {}
for alp in alphas2:
    lasso = linear_model.Lasso(alpha=alp)
    lasso.fit(X_train, y_train)
    sc = lasso.score(X_test, y_test)
    scores[alp] = sc
maxsc = max(scores, key=scores.get)
    
# 6)
#Ridge
maxr = round(maxr2, 2)
maxr2_value = round(r2values.get(maxr2),4)

plt.plot(alphas,r2values.values())
plt.title('Paras alpha arvo: ' + str(maxr)+ ' R2: ' + str(maxr2_value))
plt.show()

#Lasso
maxr2 = round(maxsc, 4)
maxr2_value = round(scores.get(maxsc),4)

plt.plot(alphas2, scores.values())
plt.title('Paras alpha arvo: ' + str(maxr2)+ ' R2: ' + str(maxr2_value))
plt.xlabel("alpha")
plt.ylabel("R2 score")
plt.show()

"""
5) Lasso ja Ridge regressio looppien sisällä testataan eri alphan arvoja ja nämä arvot tallennetaan sanakirjan sisälle R2 arvon kanssa.

7) Luen sanakirjoista maksimi arvon ja tulostan sen avaimen graafien otsikkoon. Näin saan mielestäni selkeästi kerrottua mikä on suurinpiirtein optimi alphan arvo.
"""