import pandas as pd

data = pd.read_csv("student_data.csv")

print(data)
print(data.shape)
print(data.isnull().sum())

X = data[["study_hours", "attendance", "sleep_hours"]]
y = data["marks"]

print(X)
print(y)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train.shape)
print(X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained successfully")

prediction = model.predict([[5, 85, 7]])
print(prediction)

print(model.score(X_test, y_test))
import matplotlib.pyplot as plt

print(model.score(X_test, y_test))

plt.scatter(data["study_hours"], data["marks"])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.show()