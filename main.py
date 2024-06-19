import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Train:
# train.csv contains the details of a subset of the passengers on board (891 passengers, to be exact -- where each passenger gets a different row in the table). To investigate this data, click on the name of the file on the left of the screen. Once you've done this, you can view all of the data in the window.
# The values in the second column ("Survived") can be used to determine whether each passenger survived or not:
#
# if it's a "1", the passenger survived.
# if it's a "0", the passenger died.
# For instance, the first passenger listed in train.csv is Mr. Owen Harris Braund. He was 22 years old when he died on the Titanic.
# \begin{itemize}
#             \item \textbf{PassengerId}: Unique ID of the passenger
#             \item \textbf{Survived}: Whether the passenger survived or not (0 = No, 1 = Yes)
#             \item \textbf{Pclass}: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
#             \item \textbf{Name}: Name of the passenger
#             \item \textbf{Sex}: Gender of the passenger
#             \item \textbf{Age}: Age of the passenger
#             \item \textbf{SibSp}: Number of siblings/spouses aboard the Titanic
#             \item \textbf{Parch}: Number of parents/children aboard the Titanic
#             \item \textbf{Ticket}: Ticket number
#             \item \textbf{Fare}: Passenger fare
#             \item \textbf{Cabin}: Cabin number
#             \item \textbf{Embarked}: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
#         \end{itemize}


# Test:
# Using the patterns you find in train.csv, you have to predict whether the other 418 passengers on board (in test.csv) survived.
#
# Click on test.csv (on the left of the screen) to examine its contents. Note that test.csv does not have a "Survived" column - this information is hidden from you, and how well you do at predicting these hidden values will determine how highly you score in the competition!
# \begin{itemize}
#             \item \textbf{PassengerId}: Unique ID of the passenger
#             \item \textbf{Pclass}: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
#             \item \textbf{Name}: Name of the passenger
#             \item \textbf{Sex}: Gender of the passenger
#             \item \textbf{Age}: Age of the passenger
#             \item \textbf{SibSp}: Number of siblings/spouses aboard the Titanic
#             \item \textbf{Parch}: Number of parents/children aboard the Titanic
#             \item \textbf{Ticket}: Ticket number
#             \item \textbf{Fare}: Passenger fare
#             \item \textbf{Cabin}: Cabin number
#             \item \textbf{Embarked}: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
#         \end{itemize}

# Gender Submission:
# The gender_submission.csv file is provided as an example that shows how you should structure your predictions. It predicts that all female passengers survived, and all male passengers died. Your hypotheses regarding survival will probably be different, which will lead to a different submission file. But, just like this file, your submission should have:
#
# a "PassengerId" column containing the IDs of each passenger from test.csv.
# a "Survived" column (that you will create!) with a "1" for the rows where you think the passenger survived, and a "0" where you predict that the passenger died.
# \begin{itemize}
#             \item \textbf{PassengerId}: Unique ID of the passenger
#             \item \textbf{Survived}: Whether the passenger survived or not (0 = No, 1 = Yes)
#         \end{itemize}


train_data = pd.read_csv("dataset/train.csv")
test_data = pd.read_csv("dataset/test.csv")
gs = pd.read_csv("dataset/gender_submission.csv")


def main():
    women = train_data.loc[train_data.Sex == 'female']["Survived"]
    rate_women = sum(women) / len(women)

    men = train_data.loc[train_data.Sex == 'male']["Survived"]
    rate_men = sum(men) / len(men)

    print("% of men who survived:", rate_men)
    print("% of women who survived:", rate_women)

    y = train_data["Survived"]

    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)
    predictions = model.predict(X_test)

    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    main()
