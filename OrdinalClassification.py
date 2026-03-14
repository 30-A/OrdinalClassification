#!pip install scikit-fingerprints feature_engine scikit-lego -q

import numpy as np
import pandas as pd
import warnings

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    cohen_kappa_score,
    balanced_accuracy_score,
    accuracy_score,
    make_scorer,
    mean_squared_error
)

from skfp.fingerprints import MACCSFingerprint
from feature_engine.selection import DropConstantFeatures, DropCorrelatedFeatures

warnings.filterwarnings("ignore")

# ==========================================
# 1. Frank & Hall Ordinal Classifier
# ==========================================
class FrankHallOrdinalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, use_calibration=False):
        self.estimator = estimator
        self.use_calibration = use_calibration

    def fit(self, X, y):
        # Identify the unique sorted classes (0: L, 1: M, 2: H)
        self.classes_ = np.sort(np.unique(y))
        self.estimators_ = []

        # Train K-1 binary classifiers
        for i in range(len(self.classes_) - 1):
            threshold = self.classes_[i]
            # Create cumulative binary target: 1 if y > threshold, 0 otherwise
            y_binary = (y > threshold).astype(int)

            clf = clone(self.estimator)

            # Apply Platt Scaling to smooth probabilities
            if self.use_calibration:
                clf = CalibratedClassifierCV(clf)

            clf.fit(X, y_binary)
            self.estimators_.append(clf)

        return self

    def predict_proba(self, X):
        # Extract the probability of the positive class (y > threshold)
        probs_greater = np.array([clf.predict_proba(X)[:, 1] for clf in self.estimators_]).T
        n_samples = X.shape[0]

        # Pad with 1.0 (P(y > -inf)) and 0.0 (P(y > inf))
        padded_probs = np.c_[np.ones(n_samples), probs_greater, np.zeros(n_samples)]

        # P(y = k) = P(y > k-1) - P(y > k)
        probs = padded_probs[:, :-1] - padded_probs[:, 1:]

        # Clip to prevent mathematical floating-point negatives, then re-normalize
        probs = np.clip(probs, 0, 1)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


# ==========================================
# 2. Load and Prepare Data
# ==========================================
df_train = pd.read_csv("data/train.csv")
df_test  = pd.read_csv("data/test.csv")

X_train = df_train["SMILES"]
X_test  = df_test["SMILES"]

y_train = df_train["Class"].map({"L": 0, "M": 1, "H": 2}).values
y_test  = df_test["Class"].map({"L": 0, "M": 1, "H": 2}).values


# ==========================================
# 3. Setup Components
# ==========================================
ordinal_mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

preprocessing_steps = [
    ("fingerprint", MACCSFingerprint(n_jobs=-1)),
    ("to_dataframe", FunctionTransformer(pd.DataFrame)),
    ("drop_constant", DropConstantFeatures(tol=.9)),
    ("drop_correlated", DropCorrelatedFeatures(threshold=.9))
]


# ==========================================
# 4. Build Models
# ==========================================
# Standard Nominal Logistic Regression
std_lr = Pipeline(steps=preprocessing_steps + [
    ("clf", LogisticRegressionCV(
        Cs=10, cv=3, penalty="l2", scoring=ordinal_mse_scorer,
        solver="lbfgs", class_weight="balanced", max_iter=5000, n_jobs=-1,
    ))
])

# Frank & Hall Ordinal Logistic Regression
ord_lr = Pipeline(steps=preprocessing_steps + [
    ("clf", FrankHallOrdinalClassifier(
        estimator=LogisticRegressionCV(
            Cs=10, cv=3, penalty="l2", scoring=ordinal_mse_scorer,
            solver="lbfgs", class_weight="balanced", max_iter=5000, n_jobs=-1,
        ),
        use_calibration=True
    ))
])


# ==========================================
# 5. Fit & Evaluate
# ==========================================
std_lr.fit(X_train, y_train)
ord_lr.fit(X_train, y_train)

def evaluate(model, X, y_true):
    y_pred = model.predict(X)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    h_to_l = (cm[2, 0] / cm[2, :].sum() * 100) if cm[2, :].sum() > 0 else 0
    return y_pred, acc, bacc, qwk, h_to_l, cm

std_tr_p, std_tr_acc, std_tr_bacc, std_tr_qwk, std_tr_hl, std_tr_cm = evaluate(std_lr, X_train, y_train)
std_te_p, std_te_acc, std_te_bacc, std_te_qwk, std_te_hl, std_te_cm = evaluate(std_lr, X_test, y_test)
ord_tr_p, ord_tr_acc, ord_tr_bacc, ord_tr_qwk, ord_tr_hl, ord_tr_cm = evaluate(ord_lr, X_train, y_train)
ord_te_p, ord_te_acc, ord_te_bacc, ord_te_qwk, ord_te_hl, ord_te_cm = evaluate(ord_lr, X_test, y_test)


# ==========================================
# 6. Print Results
# ==========================================
print(f"\n{'':12}  {'── Train ──':^35}  {'── Test ──':^35}")
print(f"{'Model':<12}  {'Acc':>7}  {'BAcc':>7}  {'QWK':>7}  {'H→L%':>7}  {'Acc':>7}  {'BAcc':>7}  {'QWK':>7}  {'H→L%':>7}")
print("─" * 84)
print(f"{'Standard':<12}  {std_tr_acc:>7.2f}  {std_tr_bacc:>7.2f}  {std_tr_qwk:>7.2f}  {std_tr_hl:>6.1f}%  {std_te_acc:>7.2f}  {std_te_bacc:>7.2f}  {std_te_qwk:>7.2f}  {std_te_hl:>6.1f}%")
print(f"{'Ordinal':<12}  {ord_tr_acc:>7.2f}  {ord_tr_bacc:>7.2f}  {ord_tr_qwk:>7.2f}  {ord_tr_hl:>6.1f}%  {ord_te_acc:>7.2f}  {ord_te_bacc:>7.2f}  {ord_te_qwk:>7.2f}  {ord_te_hl:>6.1f}%")
print("─" * 84)

labels = ["L", "M", "H"]
cms = [
    ("Standard — Train", std_tr_cm), ("Standard — Test", std_te_cm),
    ("Ordinal — Train", ord_tr_cm), ("Ordinal — Test", ord_te_cm)
]

for title, cm in cms:
    print(f"\n{title}")
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string(col_space=6))

