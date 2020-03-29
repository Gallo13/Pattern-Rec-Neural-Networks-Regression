# Jessica Gallo

# Created: 3/8/2020
# Last Modified: 3/24/2020

# CSC 732 Pattern Recognition and Neural Networks
# Regression (Linear, Multiple, Quadratic, Cubic etc.)
# Using Logistic Regression for Prediction
# Part 3

# ==================
# LINEAR REGRESSION |
# ==================

# LIBRARIES
import pandas as pd
from sklearn.preprocessing import OneHotEncoder as encoder
import numpy as np
from sklearn.metrics import r2_score
import statsmodels.api as sm

# READ DATASET
df = pd.read_csv('insurance.csv')

# ------------------
# PREPARING DATASET |
# ------------------
dummyVar1 = pd.get_dummies(df['sex'])
dummyVar2 = pd.get_dummies(df.region)
dummyVar3 = pd.get_dummies(df.smoker)

# dropping one column from each new dummy variables inorder to escape the dummy variable trap
dummyVar1 = dummyVar1.drop(labels = 'female', axis = 'columns')
dummyVar2 = dummyVar2.drop(labels = 'southwest', axis = 'columns')
dummyVar3 = dummyVar3.drop(labels = 'no', axis = 'columns')

mgData = pd.concat([df,dummyVar1, dummyVar2, dummyVar3], axis=1)

final = mgData.drop(['sex', 'region', 'smoker'], axis = 'columns')
# will rename the column to 'sex' if it's 1 it's male otherwise it's female
final = final.rename(columns={'male': 'sex', 'yes':'smoker'})

# function to divide dataset into 70,20 and 10%
# it will return X_train, X_validation, X_test, y_train, y_validation and y_test
from sklearn.model_selection import train_test_split
def divideData(X, y):
  X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=0)
  X_validation, X_test, y_validation, y_test = train_test_split(X_tmp, y_tmp, test_size=1/3, random_state=0)
  return X_train, X_validation, X_test, y_train, y_validation, y_test

# using age as the explanatory variable for our first linear regression

X_age = df['age'].values[: , np.newaxis]
y = df.charges.values
# linear regression

# ===================================================
# FITTING A LINEAR REGRESSION MODEL VIA SCIKIT-LEARN |
# ===================================================

from sklearn.linear_model import LinearRegression
model = LinearRegression()
X_train_age, X_validation_age, X_test_age, y_train_age, y_validation_age, y_test_age = divideData(X_age, y)
model.fit(X_train_age,y_train_age)

# OLS model
model_OLS_age = sm.OLS(y, X_age).fit()
predictions = model_OLS_age.predict(X_age)
model_OLS_age.summary()

# ============================================
# MULTIVARIATE CASES & PERFORMANCE EVALUATION |
# ============================================

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

y = final.charges.values
X = final.drop(['charges'], axis = 'columns')

X_train, X_validation, X_test, y_train, y_validation, y_test = divideData(X,y)

# Standardization
sc = StandardScaler()

sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_validation_std = sc.transform(X_validation)
X_test_std = sc.transform(X_test)

# Training
slr = LinearRegression()
slr.fit(X_train_std, y_train)

# Testing

# print(X_train_std.shape)

y_train_pred = slr.predict(X_train_std)
y_validation_pred = slr.predict(X_validation_std)
y_test_pred = slr.predict(X_test_std)



print('MSE train: %.2f, validation: %.2f, test: %.2f' % (
    mean_squared_error(y_train, y_train_pred),
    mean_squared_error(y_validation, y_validation_pred),
    mean_squared_error(y_test, y_test_pred)))

print('R^2 train: %.2f, validation: %.2f, test: %.2f' % (
    r2_score(y_train, y_train_pred),
    r2_score(y_validation, y_validation_pred),
    r2_score(y_test, y_test_pred)))



from sklearn.preprocessing import PolynomialFeatures
X_lin = df['bmi'].values[:, np.newaxis]

regr = LinearRegression()

# Create quadratic features for Polynomial Regression

quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X_lin)
X_cubic = cubic.fit_transform(X_lin)

# fit features
X_fit = np.arange(X_lin.min(), X_lin.max(), 1)[:, np.newaxis]

regr = regr.fit(X_lin, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X_lin))

regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))

print('[Linear]')
print("R^2: %.5f" % linear_r2)

print('[Quadratic]')
print("R^2: %.5f" % quadratic_r2)

print('[Cibuc]')
print("R^2: %.5f" % cubic_r2)

# ======================
# POLYNOMIAL REGRESSION |
# ======================

# -------------------
# MULTIVARIATE CASES |
# -------------------

regr = LinearRegression()


print('[Linear]')
print('#Features: %d' % X_train.shape[1])
regr = regr.fit(X_train, y_train)
y_train_pred = regr.predict(X_train)
y_validation_pred = regr.predict(X_validation)
y_test_pred = regr.predict(X_test)
print('MSE train: %.2f, validation: %.2f, test: %.2f'%(
      mean_squared_error(y_train, y_train_pred),
      mean_squared_error(y_validation, y_validation_pred),
      mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.2f, validation: %.2f, test: %.2f'%(
      r2_score(y_train, y_train_pred),
      r2_score(y_validation, y_validation_pred),
      r2_score(y_test, y_test_pred)))

print('\n[Quadratic]')
X_quad_train = quadratic.fit_transform(X_train)
X_quad_validation = quadratic.fit_transform(X_validation)
X_quad_test = quadratic.fit_transform(X_test)
print('#Features: %d' % X_quad_train.shape[1])
regr = regr.fit(X_quad_train, y_train)
y_train_pred = regr.predict(X_quad_train)
y_validation_pred = regr.predict(X_quad_validation)
y_test_pred = regr.predict(X_quad_test)
print('MSE train: %.2f, validation: %.2f, test: %.2f'%(
      mean_squared_error(y_train, y_train_pred),
      mean_squared_error(y_validation, y_validation_pred),
      mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.2f, validation: %.2f, test: %.2f'%(
      r2_score(y_train, y_train_pred),
      r2_score(y_validation, y_validation_pred),
      r2_score(y_test, y_test_pred)))

print('\n[Cubic]')
X_cubic_train = cubic.fit_transform(X_train)
X_cubic_validation = cubic.fit_transform(X_validation)
X_cubic_test = cubic.fit_transform(X_test)
print('#Features: %d' % X_cubic_train.shape[1])
regr = regr.fit(X_cubic_train, y_train)
y_train_pred = regr.predict(X_cubic_train)
y_validation_pred = regr.predict(X_cubic_validation)
y_test_pred = regr.predict(X_cubic_test)
print('MSE train: %.2f, validation: %.2f, test: %.2f'%(
      mean_squared_error(y_train, y_train_pred),
      mean_squared_error(y_validation, y_validation_pred),
      mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.2f, validation: %.2f, test: %.2f'%(
      r2_score(y_train, y_train_pred),
      r2_score(y_validation, y_validation_pred),
      r2_score(y_test, y_test_pred)))

# -------------------------
# RANDOM FOREST REGRESSION |
# -------------------------

from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators=1000,
        criterion='mse',
        random_state=1,
        n_jobs=-1)

forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_validation_pred = forest.predict(X_validation)
y_test_pred = forest.predict(X_test)

print('MSE train: %.2f, validation: %.2f test: %.2f'%(
      mean_squared_error(y_train, y_train_pred),
      mean_squared_error(y_validation, y_validation_pred),
      mean_squared_error(y_test, y_test_pred)))

print('R^2 train: %.2f, validation: %.2f, test: %.2f'%(
      r2_score(y_train, y_train_pred),
      r2_score(y_validation, y_validation_pred),
      r2_score(y_test, y_test_pred)))