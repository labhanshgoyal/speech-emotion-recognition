import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from load_data import load_data

if __name__ == "__main__":
    print("Loading data...")
    x_train, x_test, y_train, y_test = load_data()

    with open("models/ser_model.pkl", "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print ("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix - SER")
    plt.xlabel("Predicted Emotion")
    plt.ylabel("Actual Emotion")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()
    print("Confusion matrix saved to confusion_matrix.png")