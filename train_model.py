# train_model.py
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Sample dataset (features and labels)
X = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [1.0, 3.0, 5.0],
    [2.0, 4.0, 6.0],
    [3.0, 5.0, 7.0],
    [1.5, 2.5, 3.5],
    [4.5, 5.5, 6.5],
    [7.5, 8.5, 9.5],
    [1.5, 3.5, 5.5]
])

y = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the model to a file
joblib.dump(model, "decision_tree_model.joblib")

# Predict on a new sample
new_sample = np.array([[2.0, 3.0, 4.0]])
prediction = model.predict(new_sample)
print(f"Prediction for new sample {new_sample}: {prediction[0]}")
