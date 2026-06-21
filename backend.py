"""
Wildfire Prediction System — Backend
-------------------------------------
Pure ML logic: data loading, preprocessing, training 3 supervised classifiers
(Random Forest, SVM, KNN) + 1 unsupervised model (K-Means clustering), and
single-case prediction. No Streamlit import here on purpose — this file can
be tested or reused on its own.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
NUMERIC_FEATURES = ["X", "Y", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]
CATEGORICAL_FEATURES = ["month", "day"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def load_data(path="forestfires.csv"):
    """Loads the dataset and engineers the binary target.

    area == 0  -> fire caused negligible/no measurable damage (Low Risk)
    area > 0   -> fire caused measurable burn damage           (High Risk)
    """
    df = pd.read_csv(path)
    df["fire_risk"] = (df["area"] > 0).astype(int)
    return df


def _build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def _evaluate(name, y_true, y_pred):
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_all(df: pd.DataFrame) -> dict:
    """Trains Random Forest, SVM, KNN (supervised) + K-Means (unsupervised).

    Returns a single bundle dict containing fitted pipelines, evaluation
    metrics, the clustering assets, and the test split (for confusion
    matrices / reproducibility).
    """
    X = df[ALL_FEATURES]
    y = df["fire_risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    metrics = []
    pipelines = {}

    # ---- Random Forest ----
    rf = Pipeline(steps=[
        ("preprocessor", _build_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=RANDOM_STATE
        )),
    ])
    rf.fit(X_train, y_train)
    metrics.append(_evaluate("Random Forest", y_test, rf.predict(X_test)))
    pipelines["Random Forest"] = rf

    # ---- SVM ----
    svm = Pipeline(steps=[
        ("preprocessor", _build_preprocessor()),
        ("classifier", SVC(kernel="rbf", C=1.0, probability=True, random_state=RANDOM_STATE)),
    ])
    svm.fit(X_train, y_train)
    metrics.append(_evaluate("SVM", y_test, svm.predict(X_test)))
    pipelines["SVM"] = svm

    # ---- KNN ----
    knn = Pipeline(steps=[
        ("preprocessor", _build_preprocessor()),
        ("classifier", KNeighborsClassifier(n_neighbors=9)),
    ])
    knn.fit(X_train, y_train)
    metrics.append(_evaluate("KNN", y_test, knn.predict(X_test)))
    pipelines["KNN"] = knn

    # ---- K-Means clustering (unsupervised, no target leakage) ----
    cluster_scaler = StandardScaler()
    X_cluster_scaled = cluster_scaler.fit_transform(df[NUMERIC_FEATURES])
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)

    df_clustered = df.copy()
    df_clustered["cluster"] = kmeans.fit_predict(X_cluster_scaled)

    profile = (
        df_clustered.groupby("cluster")[["fire_risk", "temp", "wind", "FFMC", "DMC", "DC"]]
        .mean()
        .round(3)
    )
    order = profile["fire_risk"].sort_values().index.tolist()
    risk_labels = {order[0]: "Low Risk", order[1]: "Moderate Risk", order[2]: "High Risk"}
    profile["risk_label"] = [risk_labels[i] for i in profile.index]

    return {
        "pipelines": pipelines,
        "metrics": metrics,
        "kmeans": kmeans,
        "cluster_scaler": cluster_scaler,
        "risk_labels": risk_labels,
        "cluster_profile": profile,
        "df_clustered": df_clustered,
        "X_test": X_test,
        "y_test": y_test,
    }


def predict_one(pipelines: dict, row_df: pd.DataFrame) -> dict:
    """row_df: single-row DataFrame with ALL_FEATURES columns.
    Returns {model_name: probability_of_high_risk}.
    """
    return {name: float(pipe.predict_proba(row_df)[0][1]) for name, pipe in pipelines.items()}


def nearest_cluster(kmeans, cluster_scaler, risk_labels: dict, row_df: pd.DataFrame):
    """Returns (cluster_id, risk_label) for a single-row DataFrame."""
    scaled = cluster_scaler.transform(row_df[NUMERIC_FEATURES])
    cluster_id = int(kmeans.predict(scaled)[0])
    return cluster_id, risk_labels.get(cluster_id, "Unknown")
