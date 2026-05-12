# This is a sample Python script.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import mathplot

#
# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master"
                 "/delaney_solubility_with_descriptors.csv")

# Data preparation
# Data separation
y = df['logS']
X = df.drop('logS', axis=1)

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Model building
# LINEAR REGRESSION
# Training the model
lr = LinearRegression()
lr.fit(X_train, y_train)
# LinearRegression()

# Applying the model to make a prediction
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)
# print(y_lr_train_pred)

## Evaluation Model performance
lr_train_mse = mean_squared_error(y_train,y_lr_train_pred)
lr_train_r2 = r2_score(y_train,y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test,y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

# print(lr_train_mse, lr_test_mse)
# print(lr_train_r2, lr_test_r2)

lr_results = pd.DataFrame(['Linear regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method','Training MSE','Training R2', 'Test MSE', 'Test R2']
# print(lr_results)

# RANDOM FOREST
# Training the model
rf = RandomForestRegressor(max_depth= 2, random_state=100)
rf.fit(X_train,y_train)

# Applying the model to make a prediction
y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)
# print(y_lr_train_pred)

rf_train_mse = mean_squared_error(y_train,y_rf_train_pred)
rf_train_r2 = r2_score(y_train,y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test,y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

rf_results = pd.DataFrame(['Random forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method','Training MSE','Training R2', 'Test MSE', 'Test R2']
# print(rf_results)

# Combine table
df_models = pd.concat([lr_results,rf_results], axis=0).reset_index(drop=True)
print(df_models)

