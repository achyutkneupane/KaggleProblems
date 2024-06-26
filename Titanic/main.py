import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

pd_train_data = pd.read_csv("dataset/train.csv")
pd_test_data = pd.read_csv("dataset/test.csv")


def median_if_not_null(df, column):
    if not df[column].isnull().all():
        df[column] = df[column].fillna(df[column].median())
    else:
        df[column] = df[column].replace(0, np.nan)
        df[column] = df[column].fillna(df[column].median())


def main():
    train_data = pd_train_data.drop([
        "PassengerId", "Name", "Ticket", "Cabin"
    ], axis=1)
    test_data = pd_test_data.drop(["Name", "Ticket", "Cabin"], axis=1)

    y = train_data["Survived"]

    for df in [train_data, test_data]:
        median_if_not_null(df, "Fare")
        median_if_not_null(df, "Age")

        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = 0
        df.loc[df["FamilySize"] == 1, "IsAlone"] = 1
        df["Sex"] = df["Sex"].map({'male': 0, 'female': 1})
        df["Embarked"] = df["Embarked"].map({'S': 0, 'C': 1, 'Q': 2})

    features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Fare", "Embarked", "FamilySize", "IsAlone"]
    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=param_grid,
        cv=5
    )

    grid_search.fit(X, y)
    print(grid_search.best_params_)
    model = grid_search.best_estimator_

    predictions = model.predict(X_test)
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    main()
