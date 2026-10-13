"""Train the placement prediction model.

End-to-end pipeline: load data -> train/test split -> scale -> train
logistic regression -> evaluate -> save model + scaler.
"""
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    df = pd.read_csv("placement_data.csv")
    X = df[["cgpa", "iq"]].values
    y = df["placement"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_s, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test_s))
    cv = cross_val_score(
        LogisticRegression(max_iter=1000),
        scaler.fit_transform(X), y, cv=5,
    ).mean()
    print(f"Test accuracy : {acc:.3f}")
    print(f"5-fold CV     : {cv:.3f}")
    print(classification_report(
        y_test, clf.predict(X_test_s),
        target_names=["Not placed", "Placed"],
    ))

    with open("model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Saved model.pkl and scaler.pkl")


if __name__ == "__main__":
    main()
