# Jessica Gallo

# Created: 3/8/2020
# Last Modified: 3/24/2020

# CSC 732 Pattern Recognition and Neural Networks
# Regression (Linear, Multiple, Quadratic, Cubic etc.)
# Using Logistic Regression for Prediction
# Part 1

# ==================
# LINEAR REGRESSION |
# ==================

# LIBRARIES
from IPython.display import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder as encoder
import numpy as np

# %matplotlib inline

# READ DATASET
df = pd.read_csv('insurance.csv')
df.head()
# goal is to predict the insurance prices

sns.set(style='whitegrid', context='notebook')
sns.pairplot(df, size = 2.5)
plt.tight_layout()
# plt.savefig('./output/fig1.png', dpi =300)
plt.show()
sns.reset_orig()

"""using the scatter plot matrix we can see that 'age' and charges have some osrt of linear relation ship.
and the values of age are distributed ....
"""


# ------------------
# PREPARING DATASET |
# ------------------
# will work with it at the end
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
final.head()

# replotting scatter-plot matrix
sns.set(style='whitegrid', context='notebook')
sns.pairplot(final.drop(['northeast', 'northwest', 'southeast'], axis = 'columns'), size = 2.5)
plt.tight_layout()
# plt.savefig('./output/fig1.png', dpi =300)
plt.show()
sns.reset_orig()

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
print("X_train shape: ",X_train_age.shape)
print(y_train_age.shape)
print("X_validation shape: ", X_validation_age.shape)
print(y_validation_age.shape)
print("X_test shape: ",X_test_age.shape)
print(y_test_age.shape)

model.fit(X_train_age,y_train_age)

print("Slope (w_1): %.2f" %model.coef_[0])
print("Intercept/bias (w_0): %.2f" % model.intercept_)

# function to visualise regression model

def lin_regplot(X, y, model):
  plt.scatter(X,y, c = 'blue')
  plt.plot(X, model.predict(X), color = 'red', linewidth = 2)
  return


lin_regplot(X_age, y, model)
plt.xlabel("Age")
plt.ylabel("price")
plt.tight_layout()
plt.show()

# ============================================
# MULTIVARIATE CASES & PERFORMANCE EVALUATION |
# ============================================

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

y = final.charges.values
X = final.drop(['charges'], axis = 'columns')

X_train, X_validation, X_test, y_train, y_validation, y_test = divideData(X,y)

print("#Training data points: %d" % X_train.shape[0])
print("#Validation data points: %d" % X_validation.shape[0])
print("#Testing data points: %d" % X_test.shape[0])

#Standardization
sc = StandardScaler()

sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_validation_std = sc.transform(X_validation)
X_test_std = sc.transform(X_test)

#Training
slr = LinearRegression()
slr.fit(X_train_std, y_train)

#Testing

#print(X_train_std.shape)

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

# --------------
# RESIDUAL PLOT |
# --------------

plt.scatter(y_train_pred, y_train_pred-y_train, 
            c = 'blue', marker = 'o', label = "Training data")
plt.scatter(y_validation_pred, y_validation_pred-y_validation,
            c = 'red', marker = 'o', label = 'Validation data')
plt.scatter(y_test_pred, y_test_pred - y_test, 
            c = 'lightgreen', marker = 's', label = 'Test data')
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = 'upper left')
plt.hlines(y=0, xmin=-1000, xmax = 45000, lw=2, color = 'red')
plt.tight_layout()
plt.show()

# ===================================
# IMPLEMENTING THE LINEAR REGRESSION |
# ===================================

class LinearRegressionGD(object):
  def __init__(self, eta=0.001, n_iter=20, random_state=1):
    self.eta = eta
    self.n_iter = n_iter
    self.random_state = random_state
  def fit(self, X, y):
    rgen = np.random.RandomState(self.random_state)
    self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
    self.cost_ = []
    for i in range(self.n_iter):
      output = self.net_input(X)
      errors = (y - output)
      self.w_[1:] += self.eta * X.T.dot(errors)
      self.w_[0] += self.eta * errors.sum()
      cost = (errors**2).sum() / 2.0
      self.cost_.append(cost)
    return self
  def net_input(self, X):
    return np.dot(X, self.w_[1:]) + self.w_[0]
  def predict(self, X):
    return self.net_input(X)

# plotting the cost as a function of the number epochs...

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
sc_y = StandardScaler()

X_age_std = sc_x.fit_transform(X_age)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

lr = LinearRegressionGD()
lr.fit(X_age_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel("SSE")
plt.xlabel("Epoch")
plt.tight_layout()
plt.show()

#visualise how well the linear regression line fits the training data

lin_regplot(X_age_std, y_std, lr)
plt.xlabel('Age (standardized)')
plt.ylabel('Charge (standardized)')
plt.tight_layout()
plt.show()

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

# plot results
plt.scatter(X_lin, y, label='Training points', color='lightgray')

plt.plot(X_fit, y_lin_fit,
        label='Linear (d=1), $R^2=%.2f$' % linear_r2,
        color='blue',
        lw=2,
        linestyle=':')

plt.plot(X_fit, y_quad_fit,
        label='Quadratic (d=2), $R^2=%.2f$' % quadratic_r2,
        color='red',
        lw=2,
        linestyle='-')

plt.plot(X_fit, y_cubic_fit,
        label='Cubic (d=3), $R^2=%.2f$' % cubic_r2,
        color='green',
        lw=2,
        linestyle='--')

plt.xlabel('BMI')
plt.ylabel('Charges')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# ======================
# POLYNOMIAL REGRESSION |
# ======================

# -------------------
# MULTIVARIATE CASES |
# -------------------

# Train polynomial regressors of different degrees using all features of the dataset

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

# =========================
# DECISION TREE REGRESSION |
# =========================

from sklearn.tree import DecisionTreeRegressor

tree_3 = DecisionTreeRegressor(max_depth=3)
tree_3.fit(X_lin, y)
tree_4 = DecisionTreeRegressor(max_depth=4)
tree_4.fit(X_lin, y)
tree_5 = DecisionTreeRegressor(max_depth=5)
tree_5.fit(X_lin, y)

sort_idx = X_lin.flatten().argsort()

plt.scatter(X_lin, y, color='lightgray')

plt.plot(X_lin[sort_idx], tree_3.predict(X_lin)[sort_idx],
        color='blue',
        lw=2,
        linestyle=':')
plt.plot(X_lin[sort_idx], tree_4.predict(X_lin)[sort_idx],
        color='red',
        lw=2,
        linestyle='-')
plt.plot(X_lin[sort_idx], tree_5.predict(X_lin)[sort_idx],
        color='green',
        lw=2,
        linestyle='--')

plt.xlabel('BMI')
plt.ylabel('Charge')
plt.show()



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

# Residual plot
plt.scatter(y_train_pred,
            y_train_pred - y_train,
            c='blue',
            marker='o',
            label='Training data')
plt.scatter(y_validation_pred,
            y_validation_pred - y_validation,
            c='red',
            marker = 'o',
            label = 'Validation data')

plt.scatter(y_test_pred,
            y_test_pred - y_test,
            c='green',
            marker='s',
            label='Test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=55000, lw=2, color='red')
plt.tight_layout()

plt.show()