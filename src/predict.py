import pickle
import pandas as pd

with open("data/models/baumethoden_lr.pickle", "rb") as f:
    model = pickle.load(f)

sample = pd.DataFrame([[130, 3000]], columns=["horsepower", "weight"])

prediction = model.predict(sample)

print("Predicted MPG:", prediction[0])
