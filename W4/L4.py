import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
data = pd.read_csv(data_url, header=None, names=columns)

X = data["Glucose"].values.reshape(-1, 1)
y = data["Outcome"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy of the model: {accuracy * 100:.2f}%")

ARBITRAY_GLUCOSE_VALUES = [150, 230, 2, 69 -23, 98, 87, 420, 59, 12, 800]
for arbitrary_glucose in ARBITRAY_GLUCOSE_VALUES:
    arbitrary_glucose_scaled = scaler.transform(np.array([[arbitrary_glucose]]))
    arbitrary_prediction = model.predict(arbitrary_glucose_scaled)
    arbitrary_probability = model.predict_proba(arbitrary_glucose_scaled)

    print(f"Predicted Outcome for Glucose={arbitrary_glucose}: {arbitrary_prediction[0]} (1=Diabetes, 0=No Diabetes)")
    print(f"Probability of Diabetes: {arbitrary_probability[0][1] * 100:.2f}%")


scaled_glucose = scaler.transform(X)
probabilities = model.predict_proba(scaled_glucose)[:, 1]

plt.figure(figsize=(10, 6))
plt.scatter(X, probabilities, color='blue', alpha=0.5, label='Probability of Diabetes')
plt.title('Probability of Diabetes as a Function of Glucose Levels')
plt.xlabel('Glucose Level')
plt.ylabel('Probability of Diabetes')
plt.axhline(0.5, color='red', linestyle='--', label='Decision Boundary')
plt.legend()
plt.grid()
plt.show()
