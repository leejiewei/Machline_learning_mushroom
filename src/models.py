"""The three base models, their hyperparameter grids, tuning and the ensemble."""
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .data_preprocessing import RANDOM_STATE

PARAM_GRIDS = {
    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__solver": ["liblinear", "lbfgs"],
    },
    "Decision Tree": {
        "model__criterion": ["gini", "entropy"],
        "model__max_depth": [None, 3, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "Random Forest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
    },
}


def base_estimators():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }


def build_pipelines(preprocessor):
    return {name: Pipeline([("preprocessor", preprocessor), ("model", est)])
            for name, est in base_estimators().items()}


def tune_models(pipelines, X_train, y_train, cv, param_grids=PARAM_GRIDS, scoring="accuracy"):
    grids = {}
    for name, pipe in pipelines.items():
        g = GridSearchCV(pipe, param_grids[name], cv=cv, scoring=scoring, n_jobs=-1)
        g.fit(X_train, y_train)
        grids[name] = g
    return grids


def build_ensemble(models, member_names):
    """Soft voting: averages predicted probabilities of the chosen models."""
    return VotingClassifier(
        estimators=[(n.replace(" ", "_").lower(), models[n]) for n in member_names],
        voting="soft")
