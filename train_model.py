import pickle 
import os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from load_data import load_data

if __name__ == "__main__":
    print("Loading data...")
    x_train, x_test, y_train, y_test = load_data()
    print(f"Training samples: {len(x_train)}, Testing samples: {len(x_test)}")

    model = MLPClassifier(
        hidden_layer_sizes=(300,),
        activation="relu",
        alpha=0.01,
        batch_size=256,
        learning_rate="adaptive",
        max_iter=500,
        random_state=42,
    )

    print("\nTraining model...")
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

    os.makedirs("models", exist_ok=True)
    with open("models/ser_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("\nModel saved to models/ser_model.pkl")