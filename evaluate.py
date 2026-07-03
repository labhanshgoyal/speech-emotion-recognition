import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from load_data import load_data

if __name__ == "__main__":
    print("=" * 50)
    print("  SPEECH EMOTION RECOGNITION — EVALUATION")
    print("=" * 50)

    print("\nLoading data (no augmentation for evaluation)...")
    x_train, x_test, y_train, y_test = load_data(augment=False)

    # Load scaler, model, and label encoder
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("models/ser_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    # Scale test data and predict
    x_test_scaled = scaler.transform(x_test)
    y_pred_enc = model.predict(x_test_scaled)

    # Decode predictions back to emotion strings
    y_pred = le.inverse_transform(y_pred_enc)

    # ── Metrics ──
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print(f"\nAccuracy         : {accuracy * 100:.2f}%")
    print(f"Weighted F1-Score: {f1_weighted:.4f}")
    print(f"Macro F1-Score   : {f1_macro:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ── Confusion Matrix ──
    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix — Speech Emotion Recognition")
    plt.xlabel("Predicted Emotion")
    plt.ylabel("Actual Emotion")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("\nConfusion matrix saved to confusion_matrix.png")
    print("=" * 50)