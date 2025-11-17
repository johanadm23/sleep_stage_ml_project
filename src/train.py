import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

def train_model():
    df = pd.read_csv("data/features.csv")
    X = df.drop(columns=["label", "subject"])
    y = LabelEncoder().fit_transform(df["label"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(classification_report(y_test, preds))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/sleep_stage_rf.pkl")
    print("✅ Model saved to models/sleep_stage_rf.pkl")

if __name__ == "__main__":
    train_model()
