# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 18:14:45 2024

@author: MUSTAFA
"""


# Installing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import tkinter as tk


pd.set_option('Display.max_columns', None)

# Upload Data
df = sns.load_dataset('diamonds')

# Data Prewiew
print(df.info()) # We have 53490 values ​​and 10 columns
print(df.head(10))
print(df.isnull().sum()) # We have no null values
print(df.describe()) # There may be outliers ​​in the price variable.

# Let's examine the data in more detail and eliminate errors.
sns.boxplot(data=df, x='price')
plt.title('Outliers')
plt.show()

# Filter the data to remove invalid values
df = df[df['x'] >= 0.15]
df = df[df['z'] >= 0.15]

# Divine dataset
high_price_threshold = 10000
df_high_price = df[df['price'] > high_price_threshold]
df_low_price = df[df['price'] <= high_price_threshold]

# Encoding categorical variables so we can use them later.
def encode_features(dataframe):
    cut_mapping = {'Premium': 1, 'Fair': 2, 'Very Good': 3, 'Good': 4, 'Flair': 5}
    color_mapping = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'D': 6, 'E': 7}
    clarity_mapping = {'SI2': 1, 'SI1': 2, 'I1': 3, 'VS2': 4, 'VS1': 5, 'IF': 6, 'VVS1': 7}
    
    dataframe = dataframe.copy()
    dataframe['cut_encoding'] = dataframe['cut'].map(cut_mapping)
    dataframe['color_encoding'] = dataframe['color'].map(color_mapping)
    dataframe['clarity_encoding'] = dataframe['clarity'].map(clarity_mapping)

    return dataframe

df_high_price = encode_features(df_high_price)
df_low_price = encode_features(df_low_price)

# Feature and target selection for models
def prepare_data(dataframe):
    X = dataframe.drop(columns=['price', 'color', 'cut', 'clarity'])
    y = dataframe['price']
    
    return X, y

X_high, y_high = prepare_data(df_high_price)
X_low, y_low = prepare_data(df_low_price)

# Preparing the models
X_train_high, X_test_high, y_train_high, y_test_high = train_test_split(X_high, y_high, test_size=0.3, random_state=42)
X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(X_low, y_low, test_size=0.3, random_state=42)

# Define the model
rf_high = RandomForestRegressor(random_state=42)
rf_low = RandomForestRegressor(random_state=42)

# Hyperparameter tuning for high-priced diamonds
param_grid_high = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_high = GridSearchCV(estimator=rf_high, param_grid=param_grid_high, cv=3, n_jobs=-1, verbose=2)
grid_search_high.fit(X_train_high, y_train_high)

# Best estimator for high-priced diamonds
rf_high_best = grid_search_high.best_estimator_
print(f'Best parameters for high-priced diamonds: {grid_search_high.best_params_}')

# Hyperparameter tuning for low-priced diamonds
param_grid_low = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_low = GridSearchCV(estimator=rf_low, param_grid=param_grid_low, cv=3, n_jobs=-1, verbose=2)
grid_search_low.fit(X_train_low, y_train_low)

# Best estimator for low-priced diamonds
rf_low_best = grid_search_low.best_estimator_
print(f'Best parameters for low-priced diamonds: {grid_search_low.best_params_}')

# Evaluate the best models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

mse_high, r2_high = evaluate_model(rf_high_best, X_test_high, y_test_high)
mse_low, r2_low = evaluate_model(rf_low_best, X_test_low, y_test_low)

print(f'MSE for high-priced diamonds: {mse_high}, R2: {r2_high}')
print(f'MSE for low-priced diamonds: {mse_low}, R2: {r2_low}')

# Predict price function
def predict_price(rf_high, rf_low, input_data):
    # High price threshold
    high_threshold = 10000

    # Predicting using both models
    low_price_prediction = rf_low.predict(input_data)[0]
    high_price_prediction = rf_high.predict(input_data)[0]
    
    # Determine which prediction to return
    if low_price_prediction > high_threshold:
        return high_price_prediction
    else:
        return low_price_prediction

# Create a GUI
def predict_from_gui():
    carat = float(entry_carat.get())
    depth = float(entry_depth.get())
    table = float(entry_table.get())
    x = float(entry_x.get())
    y = float(entry_y.get())
    z = float(entry_z.get())
    cut = int(entry_cut.get())
    color = int(entry_color.get())
    clarity = int(entry_clarity.get())

    input_data = pd.DataFrame({'carat': [carat], 'depth': [depth], 'table': [table],
                               'x': [x], 'y': [y], 'z': [z], 'cut_encoding': [cut],
                               'color_encoding': [color], 'clarity_encoding': [clarity]})

    # We make predictions with the new predict_price function
    prediction = predict_price(rf_high_best, rf_low_best, input_data)
    result_label.config(text=f'Estimated Price: {prediction:.2f} $')

# Tkinter GUI
root = tk.Tk()
root.title('Diamond Price Prediction Application')

# User input fields
label_carat = tk.Label(root, text="Carat:")
label_carat.pack()
entry_carat = tk.Entry(root)
entry_carat.pack()

label_depth = tk.Label(root, text="Depth:")
label_depth.pack()
entry_depth = tk.Entry(root)
entry_depth.pack()

label_table = tk.Label(root, text="Table:")
label_table.pack()
entry_table = tk.Entry(root)
entry_table.pack()

label_x = tk.Label(root, text="X:")
label_x.pack()
entry_x = tk.Entry(root)
entry_x.pack()

label_y = tk.Label(root, text="Y:")
label_y.pack()
entry_y = tk.Entry(root)
entry_y.pack()

label_z = tk.Label(root, text="Z:")
label_z.pack()
entry_z = tk.Entry(root)
entry_z.pack()

label_cut = tk.Label(root, text="Cut Encoding:")
label_cut.pack()
entry_cut = tk.Entry(root)
entry_cut.pack()

label_color = tk.Label(root, text="Color Encoding:")
label_color.pack()
entry_color = tk.Entry(root)
entry_color.pack()

label_clarity = tk.Label(root, text="Clarity Encoding:")
label_clarity.pack()
entry_clarity = tk.Entry(root)
entry_clarity.pack()

# Prediction button
predict_button = tk.Button(root, text='Predict Price', command=predict_from_gui)
predict_button.pack()

# Result label
result_label = tk.Label(root, text="")
result_label.pack()

# GUI initialization
root.mainloop()

