import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

# Read csv file
data = pd.read_csv("data/auto-mpg-training-data.csv", sep=";")
print(data)

# Shuffle data
data = data.sample(frac=1)

# Class column
y_variable = data["mpg"]

# Feature columns
x_variables = data.loc[:, data.columns != "mpg"]

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x_variables, y_variable, test_size=0.2
)

# Train model
regressor = LinearRegression()
regressor = regressor.fit(x_train, y_train)

# Predict
y_pred = regressor.predict(x_test)
print(y_pred[:5])

# Ensure models folder exists
os.makedirs("data/models", exist_ok=True)

# Save model
file_to_write = open("data/models/baumethoden_lr.pickle", "wb")
pickle.dump(regressor, file_to_write)
file_to_write.close()
