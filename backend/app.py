"""Heart Disease Prediction - Flask backend (data prep, 6 models, REST API)."""
import os, warnings
import numpy as np, pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE, "..", "data", "heart_dataset.csv")
FRONT = os.path.join(BASE, "..", "frontend")
NUM = ["age", "resting bp s", "cholesterol", "max heart rate", "oldpeak"]
CAT = ["chest pain type", "resting ecg", "ST slope"]
BIN = ["sex", "fasting blood sugar", "exercise angina"]


class LinearRegressionClassifier(BaseEstimator, ClassifierMixin):
    """Linear Regression used as a classifier: predict value, threshold at 0.5."""
    def fit(self, X, y):
        self.classes_ = np.array([0, 1]); self.m_ = LinearRegression().fit(X, y); return self
    def _score(self, X): return np.clip(self.m_.predict(X), 0, 1)
    def predict(self, X): return (self._score(X) >= 0.5).astype(int)
    def predict_proba(self, X):
        s = self._score(X); return np.c_[1 - s, s]


def build_models():
    return {
        "Linear Regression": LinearRegressionClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }


def make_pipe(model):
    # Gaussian Naive Bayes assumes continuous inputs, so it gets the codes as numbers instead of one-hot columns
    cat = (Pipeline([("sc", StandardScaler())]) if isinstance(model, GaussianNB)
           else OneHotEncoder(handle_unknown="ignore"))
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
        ("cat", cat, CAT),
        ("bin", "passthrough", BIN)])
    return Pipeline([("pre", pre), ("model", model)])


def clean(df, dedupe=True):
    df = df.drop_duplicates().reset_index(drop=True) if dedupe else df.copy()
    # zeros in these columns are impossible -> treat as missing, imputed inside the pipeline (no leakage)
    for c in ("cholesterol", "resting bp s"):
        df.loc[df[c] == 0, c] = np.nan
    return df


def evaluate(df):
    X, y = df.drop(columns="target"), df["target"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    out, fitted = [], {}
    for name, m in build_models().items():
        pipe = make_pipe(m).fit(Xtr, ytr)
        pred, prob = pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, prob)
        step = max(1, len(fpr) // 40)
        out.append(dict(
            name=name, accuracy=accuracy_score(yte, pred), precision=precision_score(yte, pred),
            recall=recall_score(yte, pred), f1=f1_score(yte, pred), auc=roc_auc_score(yte, prob),
            cv=cross_val_score(make_pipe(build_models()[name]), X, y, cv=5).mean(),
            cm=confusion_matrix(yte, pred).tolist(),
            roc=dict(fpr=fpr[::step].round(3).tolist(), tpr=tpr[::step].round(3).tolist())))
        fitted[name] = pipe
    rf = fitted["Random Forest"]
    names = NUM + list(rf["pre"].named_transformers_["cat"].get_feature_names_out(CAT)) + BIN
    imp = sorted(zip(names, rf["model"].feature_importances_), key=lambda t: -t[1])[:8]
    return out, fitted, [dict(feature=a, importance=round(float(b), 4)) for a, b in imp], len(Xtr), len(Xte)


RAW = pd.read_csv(CSV)
DF = clean(RAW)
RESULTS, FITTED, IMPORTANCE, NTRAIN, NTEST = evaluate(DF)
BEST = max(RESULTS, key=lambda r: r["accuracy"])["name"]
# Same models on data with duplicates kept (as in the original notebook) to show the leakage effect
RAW_ACC = {r["name"]: round(r["accuracy"], 4) for r in evaluate(clean(RAW, dedupe=False))[0]}

app = Flask(__name__, static_folder=FRONT, static_url_path="")


def counts(s): return {str(k): int(v) for k, v in s.value_counts().sort_index().items()}


@app.get("/")
def index(): return send_from_directory(FRONT, "index.html")


@app.get("/api/overview")
def overview():
    return jsonify(rows_raw=len(RAW), rows_clean=len(DF), features=RAW.shape[1] - 1,
                   duplicates=int(RAW.duplicated().sum()), models=len(FITTED), best=BEST,
                   best_acc=max(r["accuracy"] for r in RESULTS), positive=float(DF["target"].mean()))


@app.get("/api/preprocessing")
def preprocessing():
    cols = [dict(column=c, dtype=str(RAW[c].dtype), min=float(RAW[c].min()), max=float(RAW[c].max()),
                 mean=round(float(RAW[c].mean()), 2), missing=int(RAW[c].isnull().sum()),
                 invalid=int((RAW[c] == 0).sum()) if c in ("cholesterol", "resting bp s") else 0)
            for c in RAW.columns]
    return jsonify(columns=cols, rows_raw=len(RAW), rows_clean=len(DF), train=NTRAIN, test=NTEST,
                   chol_zero=int((RAW["cholesterol"] == 0).sum()), bp_zero=int((RAW["resting bp s"] == 0).sum()),
                   duplicates=int(RAW.duplicated().sum()), numeric=NUM, categorical=CAT, binary=BIN)


@app.get("/api/eda")
def eda():
    d = DF.copy()
    d["grp"] = pd.cut(d["age"], [25, 35, 40, 45, 50, 55, 60, 65, 70, 80])
    chol = d.groupby("grp", observed=True)["cholesterol"].mean().round(1)
    ages = pd.cut(DF["age"], range(25, 85, 5), right=False).value_counts().sort_index()
    corr = DF.corr()["target"].drop("target").sort_values()
    return jsonify(
        target=counts(DF["target"]),
        age=dict(labels=[f"{int(i.left)}-{int(i.right) - 1}" for i in ages.index], values=ages.tolist()),
        sex={str(k): {str(a): int(b) for a, b in v.items()} for k, v in pd.crosstab(DF["sex"], DF["target"]).iterrows()},
        chest=counts(DF["chest pain type"]),
        chol=dict(labels=[f"{int(i.left) + 1}-{int(i.right)}" for i in chol.index], values=chol.tolist()),
        corr=dict(labels=corr.index.tolist(), values=corr.round(3).tolist()))


@app.get("/api/models")
def models():
    return jsonify(results=RESULTS, best=BEST, importance=IMPORTANCE, raw_accuracy=RAW_ACC,
                   train=NTRAIN, test=NTEST)


@app.post("/api/predict")
def predict():
    p = request.get_json(force=True, silent=True) or {}
    try:
        row = pd.DataFrame([{c: float(p[c]) for c in NUM + CAT + BIN}])
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Please fill in all fields with valid numbers."), 400
    return jsonify(probabilities={n: float(m.predict_proba(row)[0, 1]) for n, m in FITTED.items()}, best=BEST)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Open http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
