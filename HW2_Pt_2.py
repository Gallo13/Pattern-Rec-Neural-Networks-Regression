# Jessica Gallo

# Created: 3/8/2020
# Last Modified: 3/24/2020

# CSC 732 Pattern Recognition and Neural Networks
# Regression (Linear, Multiple, Quadratic, Cubic etc.)
# Using Logistic Regression for Prediction
# Part 2

# --------
# IMPORTS |
# --------
# Main Libraries
import numpy as np
import pandas as pd

# Visual Libraries
import matplotlib.pyplot as plt

# Sklearn Libraries
from sklearn.impute import SimpleImputer
import sklearn.model_selection as model_selection
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# from sklearn import svm
# import sys

# ------------------
# INSURANCE DATASET |
# ------------------
dataset = pd.read_csv('insurance.csv')
# reading dataset
X = dataset.iloc[:, :-1].values
# X represents a matrix of independent variables
# the 1st ':' stands for all rows
# the second ':' stands for all the columns minus the last one (-1)
y = dataset.iloc[:, 6].values
# y represents a vector of the dependent variable
# all rows included, but from the columns we only need the 7th (6th index)

print('Original:'
      '\n=========')
print(dataset)

# ===================
# DATA PREPROCESSING |
# ===================

# -------------
# MISSING DATA |
# -------------
imputer = SimpleImputer(missing_values=np.nan, strategy='constant')
# handles missing data & replaces NaN values
# strategy argument 'constant' replaces missing values with fill_value (for string/object datatypes)
imputer = imputer.fit(X[:, 1:6])
# fits the imputer on X
# # fits data to avoid data leakage during cross validation
X[:, 1:6] = imputer.transform(X[:, 1:6])
# imputes all missing values in X

print("\nImputed:"
      "\n========")
print(X)

# ------------------------------------------------------------------------
# CONVERT CATEGORICAL TEXT DATA INTO MODEL-UNDERSTANDABLE NUMBERICAL DATA |
# ------------------------------------------------------------------------
labelencoder_X = LabelEncoder()
# encodes target lables with values between 0 and n_classes -1
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# fit label encoder and return encoded labels
onehotencoder = OneHotEncoder(dtype=np.float)
# creates a binary column for each category and returns a sparse matrix or dense array
X = onehotencoder.fit_transform(X).toarray()
# fit OneHotEncoder to X, then transform X

print('\nEncoded:'
      '\n========')
print(X)

# ------------------------------------------------------
# SPLITTING DATASET INTO TRAINING, VALIDATION & TESTING |
# ------------------------------------------------------
# Splits arrays or matrices into random train and test subsets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.2, random_state=1)
# test_size represents the proportion of the dataset to include in the test split
# random_state is the seed used by the random number generator

print('\nSplitting Dataset:'
      '\n==================')
print('X_train: \n', X_train)
print('\ny_train: \n', y_train)
print('\nX_test: \n', X_test)
print('\ny_test: \n', y_test)
print('\nX_val: \n', X_val)
print('\ny_val: \n', y_val)

# -----------------
# Cross Validation |
# -----------------

# Provides train/trest indices to split data in train.test sets
# Each fold is then used once as a validation while the k-1 remaining folds from the training set
kf = KFold(n_splits=5)  # number of folds is 5
X = np.array(X)
y = np.array(y)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print('\nCross Validation:'
          '\n=================')
    print('X_test: \n', X_test)

# --------------------------------------------------------
# FITTING MULTIPLE REGRESSION MODELS TO THE TRAINGING SET |
# --------------------------------------------------------
# Fits a linear model with coefficients w to minimize the residual sum of squares between
# the observed targets in the dataset, and the targets predicted by the linear approximation
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print(regressor.intercept_)
# retrieve the intercept

print(regressor.coef_)
# retrieves the slope (coefficient of X)

# ----------------------------
# PREDICTING THE TEST RESULTS |
# ----------------------------
y_pred = regressor.predict(X_test)

print('Predicting test set results')
print(X_test)

dataset = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(dataset)

# --------------------------
# METHOD OF FEATURE SCALING |
# --------------------------

# ----------------
#  STANDARDSCALER |
# ----------------
# Computes mean and standard deviation on a training set so as to be able to later
# reapply the same transformation on the testing set.
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

print("\nStandardScalar:"
      "\n==============="
      "\nX_train:", X_train)
print('\nX_test:', X_test)

# --------------
#  MINMAXSCALAR |
# --------------
# Scaling features to lie between a given minimum and maximum value, often between 0 and 1
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)

print("\nMinMaxScalar:"
      "\n============="
      "\nX_train:", X_train)
print('\nX_test:', X_test)

# --------------
#  ROBUSTSCALAR |
# --------------
# This removed the median and scaled the data according to the quantile range
robust_scaler = RobustScaler()
X_train = robust_scaler.fit_transform(X_train)
X_test = robust_scaler.transform(X_test)

print("\nRobustScalar:"
      "\n============="
      "\nX_train:", X_train)
print('\nX_test:', X_test)

# --------------
#  NORMALIZER   |
# --------------
# Normalize samples individually to unit norm
# Each sample (each row of the data matrix) with at least one non zero component is rescaled
# indepentently o other samples so that its norm (|1 or |2) equals 1
normalizer_scaler = Normalizer()
X_train = normalizer_scaler.fit_transform(X_train)
X_test = normalizer_scaler.transform(X_test)

print("\nNormalizer:"
      "\n==========="
      "\nX_train:", X_train)
print('\nX_test:', X_test)

# -------------------------------------------------------------
# OPTIMAL MODEL USING BACKWARD ELIMINATION (FEATURE SELECTION) |
# -------------------------------------------------------------
import statsmodels.api as sm
X = np.append(arr=np.ones((1338, 1)).astype(int), values=X, axis=1)
# add one column with all 50 values as 1 to represent b0x0

X_opt = X[:, [0, 1, 2, 3, 4, 5, 6]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
# endog is the dependent variable & exog is the independent variable
print('\n', regressor_OLS.summary())

# remove index x6 as it has the highest p-value
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print('\n', regressor_OLS.summary())

# remove index x3 as it has the highest p-value
X_opt = X[:, [0, 1, 2, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print('\n', regressor_OLS.summary())

# remove index x4 as it has the highest p-value
X_opt = X[:, [0, 1, 2, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print('\n', regressor_OLS.summary())

# remove index x3 as it has the highest p-value
X_opt = X[:, [0, 1, 2]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print('\n', regressor_OLS.summary())

# remove index x2 as it has the highest p-value
X_opt = X[:, [0, 1]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print('\n', regressor_OLS.summary())
# now all variables are under the significance level of 0.05
# the only one variable left has the highest impact on the profit and is statistically significant

# ######################################################################################################################
# ================
# DATA PROCESSING |
# ================

# --------------
# Normalization |
# --------------


# Normalize all input data to 0 and 1
def _normalize_column_0_1(X, train=True, specified_column=None, X_min=None, X_max=None):
    # The output of the function will make the specified column of the training data from 0 to 1
    # When processing testing data,k we need to normalize by the value we used for processing training, so we must
    # save the max value of the training data
    if train:
        if specified_column is None:
            specified_column = np.arange(X.shape[1])
            # np.arange returns evenly spaced values within a given interval
        length = len(specified_column)
        # returns the number of items
        X_max = np.reshape(np.max(X[:, specified_column], 0), (1, length))
        # np.reshape gives a new shape to an array without changing its data
        # np.max finds the value of maximum element in the entire array (returning scalar)
        X_min = np.reshape(np.min(X[:, specified_column], 0), (1, length))
        # np.min returns the minimum of an array or minimum along an axis
        print('\n_normalize_column_0_1:')
        print('X_max is \n' + str(X_max))
        print('X_min is \n' + str(X_min))
        print('specified_column is \n' + str(specified_column))

    np.seterr(divide='ignore', invalid='ignore')
    # code is trying to divide by zero, this ignores the runtime error

    X[:, specified_column] = np.divide(np.subtract(X[:, specified_column], X_min), np.subtract(X_max, X_min))
    # np.divide returns a true division of the inputs, element-wise
    # np.subtract subtracts arguments, element-wise

    return X, X_max, X_min


print('_normalize_column_0_1:'
      '\n======================\n', X)

_normalize_column_0_1(X)


# Normalize the specified column to a normal distribution
def _normalize_column_normal(X, train=True, specified_column=None, X_mean=None, X_std=None):
    # The output of the function will make the specified column number to become a Normal Distribution
    # When processing testing data, we need to normalize by the value we used for processing traing, so we must
    # save the mean value and the variance of the training data
    if train:
        if specified_column is None:
            specified_column = np.arange(X.shape[1])
        length = len(specified_column)
        X_mean = np.reshape(np.mean(X[:, specified_column], 0), (1, length))
        X_std = np.reshape(np.std(X[:, specified_column], 0), (1, length))

    X[:, specified_column] = np.divide(np.subtract(X[:, specified_column], X_mean), X_std)

    return X, X_mean, X_std


print('\n_normalize_column_normal:'
      '\n=========================\n', X)


# Makes the data random and clean and changes the order
def _shuffle(X, y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    # shuffles the array along the 1st axis of a multi-dimensional array
    # the order of sub-arrays is changed but their contents remains the same
    return X[randomize], y[randomize]


print('\n_shuffle: '
      '\n=========\n')
_shuffle(X, y)


# Divide the data and choose according to the proportion
def train_dev_split(X, y, dev_size=0.25):
    train_len = int(round(len(X) * (1 - dev_size)))
    return X[0:train_len], y[0:train_len], X[train_len:None], y[train_len:None]


print('\ntrain_dev_split:'
      '\n================')
print(X, y)

train_dev_split(X, y)

col = [0, 1, 2, 3, 4, 5, 6]
X_train, X_mean, X_std = _normalize_column_normal(X_train, specified_column=col)


# -------------------------
# LOGISTIC REGRESSION TOOL |
# -------------------------
def _sigmoid(z):
    # sigmoid function can be used to output probability
    # limits the output to a range between 0 and 1
    return 1 / (1.0 + np.exp(-z)), 1e-6, 1-1e-6
    # 1e-6 is equivalent to 1 with 6 zeros (1,000,000)


_sigmoid(0)
# test to make sure it works


def get_prob(X, w, b):
    # the probability to output 1
    return _sigmoid(np.add(np.matmul(X, w), b))
    # np.matmul matrix product of 2 arrays
    # np.add adds arguments element-wise


def infer(X, w, b):
    # use round to infer the result
    return np.round(get_prob(X, w, b))


def _cross_entropy(y_pred, y_label):
    # compute the cross entropy
    # used to quantify the difference between 2 probability distributions
    cross_entropy = -np.dot(y_label, np.log(y_pred))-np.dot((1-y_label), np.log(1-y_pred))
    # np.dot is the dot product of 2 arrays
    # np.log = natural logarithm
    return cross_entropy


def _gradient(X, y_label, w, b):
    # return the mean of the gradient
    y_pred = get_prob(X, w, b)
    pred_error = y_label - y_pred
    w_grad = -np.mean(np.multiply(pred_error.T, X.T), 1)
    b_grad = -np.mean(pred_error)
    return w_grad, b_grad


def _gradient_regularization(X, y_label, w, b, lamda):
    # return the mean of the gradient
    y_pred = get_prob(X, w, b)
    pred_error = y_label - y_pred
    w_grad = -np.mean(np.multiply(pred_error.T, X.T), 1) + lamda * w
    # .T transposes the array
    b_grad = -np.mean(pred_error)
    return w_grad, b_grad


def _loss(y_pred, y_label, lamda, w):
    return _cross_entropy(y_pred, y_label)+lamda*np.sum(np.square(w))


def accuracy(y_pred, y_label):
    acc = np.sum(y_pred == y_label) / len(y_pred)
    return acc


# --------------------
# Logistic Regression |
# --------------------
def train(X_train, y_train):
    # split a validation set
    dev_size = 0.1155
    X_train, y_train, X_dev, y_dev = train_dev_split(X_train, y_train, dev_size=dev_size)

    # Use 0 + 0*x1 + 0*x2 + ... for weight initialization
    w = np.zeros((X_train.shape[1],))
    b = np.zeros((1,))
    regularize = True
    if regularize:
        lamda = 0.001
    else:
        lamda = 0

    max_iter = 40  # max iteration number
    batch_size = 32  # number to feed in the model for average to avoid bias
    learning_rate = 0.2  # how much the model learn for each step
    num_train = len(y_train)
    num_dev = len(y_dev)
    step = 1

    loss_train = []
    loss_validation = []
    train_acc = []
    dev_acc = []

    for epoch in range(max_iter):
        # Random shuffle for each epoch
        X_train, y_train = _shuffle(X_train, y_train)

        total_loss = 0.0
        # Logistic regression train with batch
        for idx in range(int(np.floor(len(y_train)/batch_size))):
            X = X_train[idx*batch_size:(idx+1)*batch_size]
            y = y_train[idx*batch_size:(idx+1)*batch_size]

            # Find out the gradient of the loss
            w_grad, b_grad = _gradient_regularization(X, y, w, b, lamda)
            # gradient descent update
            # learning rate decay with time
            w = w - learning_rate/np.sqrt(step) * w_grad
            b = b - learning_rate/np.sqrt(step) * b_grad

            step = step+1

        # Compute the loss and the accuracy of the training set and the validation set
        y_train_pred = get_prob(X_train, w, b)
        y_train_pred = np.round(y_train_pred)
        train_acc.append(accuracy(y_train_pred, y_train))
        loss_train.append(_loss(y_train_pred, y_train, lamda, w)/num_train)
        y_dev_pred = get_prob(X_dev, w, b)
        y_dev_pred = np.round(y_dev_pred)
        dev_acc.append(accuracy(y_dev_pred, y_dev))
        loss_validation.append(_loss(y_dev_pred, y_dev, lamda, w)/num_dev)

    return w, b, loss_train, loss_validation, train_acc, dev_acc  # return loss for plotting


# return loss is to plot the result
w, b, loss_train, loss_validation, train_acc, dev_acc = train(X_train, y_train)

plt.plot(loss_train)
plt.plot(loss_validation)
plt.legend(['train', 'dev'])
plt.show()

plt.plot(train_acc)
plt.plot(dev_acc)
plt.legend(['train', 'dev'])
plt.show()

# USE N-FOLD CROSS VALIDATION TO FIND THE BEST PARAMETERS FOR THE MODEL
# -----------------------------
#  TEST DATA AND OUTPUT RESULT |
# -----------------------------
'''
X_train_fpath = sys.argv[1]
Y_train_fpath = sys.argv[2]
X_test_fpath = sys.argv[3]
output_fpath = sys.argv[4]


X_train_fpath = 'data/X_train'
Y_train_fpath = 'data/Y_train'
X_test_fpath = 'data/X_test'
output_fpath = 'output.csv'
'''
X_train = np.genfromtxt(X, delimiter=',', skip_header=1)
Y_train = np.genfromtxt(y, delimiter=',', skip_header=1)


X_test = np.genfromtxt(X_test, delimiter=',', skip_header=1)
# Do the same data process to the test data
X_test, _, _ = _normalize_column_normal(X_test, train=False, specified_column=col, X_mean=X_mean, X_std=X_std)

result = infer(X_test, w, b)

with open(output_fpath, 'w') as f:
    f.write('id,label\n')
    for i, v in enumerate(result):
        f.write('%d,%d\n' % (i+1, v))

ind = np.argsort(np.abs(w))[::-1]
with open(X_test_fpath) as f:
    content = f.readline().rstrip('\n')
features = np.array([x for x in content.split(',')])
for i in ind[0:10]:
    print(features[i], w[i])

'''
if __name__ == '__main__':
    X_train_fpath = sys.argv[1]
    Y_train_fpath = sys.argv[2]
    X_test_fpath = sys.argv[3]
    output_fpath = sys.argv[4]
'''