import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier


url = "https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv"
data = pd.read_csv(url)

X = data.drop("Class", axis=1)
y = data["Class"]


samplers = {
    "Sampling1": RandomOverSampler(random_state=7),
    "Sampling2": RandomUnderSampler(random_state=7),
    "Sampling3": SMOTE(random_state=7),
    "Sampling4": SMOTEENN(random_state=7),
    "Sampling5": None
}


models = {
    "M1": LogisticRegression(max_iter=2000),
    "M2": DecisionTreeClassifier(),
    "M3": RandomForestClassifier(n_estimators=150),
    "M4": SVC(),
    "M5": GaussianNB()
}


results = pd.DataFrame(index=models.keys(), columns=samplers.keys())


for s_name, sampler in samplers.items():

    if sampler is not None:
        X_res, y_res = sampler.fit_resample(X, y)
    else:
        X_res, y_res = X, y

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.30, random_state=7
    )

    for m_name, model in models.items():

        if s_name == "Sampling5":
            clf = BalancedRandomForestClassifier(
                n_estimators=150, random_state=7
            )
        else:
            clf = model

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred) * 100
        results.loc[m_name, s_name] = round(acc, 2)


print(results)

best_score = results.astype(float).values.max()
best_pair = results.astype(float).stack().idxmax()

print("\nFinal Accuracy Table\n")
print(results.to_string())

print("\nBest Result\n")
print("Model :", best_pair[0])
print("Sampling :", best_pair[1])
print("Accuracy :", round(best_score, 2))
