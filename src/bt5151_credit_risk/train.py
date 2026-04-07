from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from bt5151_credit_risk.config import RANDOM_SEED


# Create the candidate models we compare in the project.
def build_candidate_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_SEED,
        ),
    }
