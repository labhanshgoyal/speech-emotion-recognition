import pickle 
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from load_data import load_data

if __name__ == "__main__":
    print("=" * 50)
    print("  SPEECH EMOTION RECOGNITION - MODEL TRAINING")
    print("=" * 50)

    print("\nLoading data (180 features)...")
    x_train, x_test, y_train, y_test = load_data(augment=False)

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Scale features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    print(f"Training samples : {len(x_train)}")
    print(f"Testing samples  : {len(x_test)}")
    print(f"Feature dimension: {x_train.shape[1]}")

    # -- Train multiple classifiers, pick the best --
    models = {
        "MLP (300)": MLPClassifier(
            hidden_layer_sizes=(300,), activation="relu", alpha=0.01,
            batch_size=256, learning_rate="adaptive", max_iter=500,
            random_state=42
        ),
        "MLP (256,128)": MLPClassifier(
            hidden_layer_sizes=(256, 128), activation="relu", alpha=0.01,
            batch_size=256, learning_rate="adaptive", max_iter=1000,
            random_state=42
        ),
        "SVM (RBF)": SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
        "SVM (RBF C=50)": SVC(kernel="rbf", C=50, gamma="scale", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=None, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }

    best_acc = 0
    best_model = None
    best_name = ""

    print("\n" + "-" * 50)
    print("Training models...")
    print("-" * 50)

    for name, model in models.items():
        print(f"  Training {name}...", end=" ", flush=True)
        model.fit(x_train_scaled, y_train_enc)
        acc = accuracy_score(y_test_enc, model.predict(x_test_scaled))
        marker = ""
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
            marker = " <-- BEST"
        print(f"{acc*100:.2f}%{marker}")

    print("-" * 50)
    print(f"\nBest model: {best_name} ({best_acc*100:.2f}%)")

    # Full report for best model
    y_pred = le.inverse_transform(best_model.predict(x_test_scaled))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # -- Save --
    os.makedirs("models", exist_ok=True)
    with open("models/ser_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("Model, scaler, and label encoder saved to models/")
    print("=" * 50)