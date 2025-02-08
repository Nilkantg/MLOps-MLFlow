import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine # type: ignore
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='Nilkantg', repo_name='MLOps-MLFlow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/Nilkantg/MLOps-MLFlow.mlflow")

mlflow.autolog()
mlflow.set_experiment('YT-MLOps-Autolog')

wine_dataset = load_wine()
X = wine_dataset.data
Y = wine_dataset.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

max_depth = 3
n_estimators = 9

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, Y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)

    # creating confusion matrix graph
    cm  =confusion_matrix(Y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=wine_dataset.target_names, yticklabels=wine_dataset.target_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")

    # saving plot
    plt.savefig("Confusion-matrix.png")

    # log artifacts using mlflow
    mlflow.log_artifact(__file__)

    # tags
    mlflow.set_tags({"Author": "Nilkant", "Project": "Wine Classification"})

    print(accuracy)