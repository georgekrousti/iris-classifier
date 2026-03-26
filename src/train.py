#Load iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data  # shape (150, 4)
y = iris.target  # shape (150,)
print(iris.feature_names, iris.target_names)
#Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Decision Tree training
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#Predictions
print("Predictions:", y_pred[:5])
print("True labels:", y_test[:5])
#Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)
print("k-NN accuracy:", accuracy_score(y_test, y_pred_knn))
#Evaluation
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#new code to save confusion matrix to a different file
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.savefig("outputs/confusion_matrix.png")

# Save trained models
import joblib
from pathlib import Path

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

joblib.dump(model, output_dir / "iris_model.joblib")
joblib.dump(model_knn, output_dir / "knn_model.joblib")



