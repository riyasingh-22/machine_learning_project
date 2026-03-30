import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


data = pd.read_csv("student_attention.csv")

print(" Data Loaded")
print(data.head())


X = data.drop("attentive", axis=1)
y = data["attentive"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(" Model Trained")


y_pred = model.predict(X_test)


print(" Accuracy:", accuracy_score(y_test, y_pred))


importances = model.feature_importances_
features = X.columns

plt.figure()
plt.bar(features, importances)
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.show()


new_student = [[1, 0, 1, 0, 0, 1, 0]]
prediction = model.predict(new_student)

print("\n New Student Prediction:")

if prediction[0] == 1:
    print(" Student is Attentive")
else:
    print(" Student is Not Attentive")