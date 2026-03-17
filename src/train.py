import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import pickle
import os

# Read csv file
data = pd.read_csv("data/auto-mpg-training-data.csv", sep=";")
print(data.head())

# Shuffle data. It randomizes the rows. models learn better when data is mixed.
# frac=1 = take 100% of data but shuffled
data = data.sample(frac=1)

# Class column. mpg means or is miles per gallon. 1 US gallon `~ 3.8 liters`
y_variable = data["mpg"]

# Feature columns
x_variables = data.loc[:, data.columns != "mpg"]

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x_variables, y_variable, test_size=0.2
)

# Train model
model = LinearRegression()
model.fit(x_train, y_train)
# regressor = LinearRegression()
# regressor = regressor.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)
print(y_pred[:5])

# Evaluate model perfomance
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Visualize the output
plt.scatter(y_test, y_pred)
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.title("Actual vs Predicted MPG")
plt.show(block=False)
# Ensure models folder exists
os.makedirs("data/models", exist_ok=True)

# Save model
with open("data/models/baumethoden_lr.pickle", "wb") as f:
    pickle.dump(model, f)
