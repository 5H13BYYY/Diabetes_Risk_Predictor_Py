"""Diabetes Risk Prediction — Final Project (Python version)

Pipeline: data prep -> EDA -> modeling -> threshold tuning -> save model.
Requires: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib
Data: diabetes_prediction_dataset.csv in the same folder (Kaggle: iammustafatz/diabetes-prediction-dataset)
"""

# %% [markdown]
# # Diabetes Risk Prediction — Final Project
#
# **Pipeline:** data prep → EDA → modeling (Logistic Regression, Decision Tree, Random Forest) → threshold tuning → save final model.
#
# **Dataset:** [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) (Kaggle: add it via **Add Input**; GitHub: download the CSV into the same folder as this file).

# %% [markdown]
# ## 0. Setup

# %%
import os
import glob
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (RocCurveDisplay, accuracy_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
RANDOM_STATE = 123

# %% [markdown]
# ## 1. Data Loading & Preprocessing

# %%
# Works on Kaggle (searches /kaggle/input) and locally (CSV next to this file)
LOCAL_CSV = "diabetes_prediction_dataset.csv"

if os.path.exists(LOCAL_CSV):
    csv_path = LOCAL_CSV
else:
    matches = glob.glob("/kaggle/input/**/*.csv", recursive=True)
    preferred = [m for m in matches if "diabetes" in os.path.basename(m).lower()]
    found = preferred or matches
    if not found:
        raise FileNotFoundError(
            "CSV not found. On Kaggle, click 'Add Input' and add the "
            "'Diabetes Prediction Dataset'. Locally, place "
            "diabetes_prediction_dataset.csv next to this file."
        )
    csv_path = found[0]

print("Loading:", csv_path)
df = pd.read_csv(csv_path)
df.info()
df.head()

# %% [markdown]
# **Checks performed before cleaning**
# - `smoking_history` "No Info" = 35.8% of rows → kept as its own category, not imputed, since it carries real signal (lowest diabetes rate of any group)
# - `age` min = 0.08 (infant ages) → legitimate data, zero diabetic cases, consistent with clinical expectations, not removed
# - `bmi` range 10.01–95.69 → extremes checked, both plausible (low end = children, high end = documented obesity), not capped
# - `gender` "Other" = 18 rows → dropped (too few to be a reliable category)

# %%
df = df[df["gender"] != "Other"].copy()

# smoking_history: 6 -> 4 categories
smoking_map = {
    "never": "never",
    "former": "former",
    "current": "current_or_past",
    "not current": "current_or_past",
    "ever": "current_or_past",
    "No Info": "No_Info",
}
df["smoking_history"] = df["smoking_history"].map(smoking_map)

df["hypertension"] = df["hypertension"].map({0: "No", 1: "Yes"})
df["heart_disease"] = df["heart_disease"].map({0: "No", 1: "Yes"})
df["diabetes_f"] = df["diabetes"].map({0: "No", 1: "Yes"})

df.head()

# %% [markdown]
# `smoking_history` collapsed from 6 → 4 categories: **former** kept separate since it had a clearly higher diabetes rate (0.17) than everything else; **current**, **not current**, **ever** merged into `current_or_past` since their rates were all close together (0.10–0.12).

# %% [markdown]
# ## 2. Exploratory Data Analysis

# %% [markdown]
# ### Target distribution

# %%
target_dist = df["diabetes_f"].value_counts().rename("n").to_frame()
target_dist["pct"] = (target_dist["n"] / target_dist["n"].sum() * 100).round(1)
target_dist

# %%
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="diabetes_f", order=["No", "Yes"], hue="diabetes_f", legend=False)
plt.title("Diabetes Class Distribution")
plt.xlabel("Diabetes")
plt.ylabel("Count")
plt.show()

# %% [markdown]
# ### Numeric predictors vs diabetes

# %%
numeric_vars = ["bmi", "age", "HbA1c_level", "blood_glucose_level"]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, var in zip(axes, numeric_vars):
    sns.boxplot(data=df, x="diabetes_f", y=var, order=["No", "Yes"],
                hue="diabetes_f", legend=False, ax=ax)
    ax.set_title(var)
    ax.set_xlabel("Diabetes")
    ax.set_ylabel("")
fig.suptitle("Numeric Predictors by Diabetes Status", y=1.03)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Categorical predictors vs diabetes

# %%
def rate_plot(col, color, horizontal=False, sort=False):
    rates = df.groupby(col)["diabetes"].mean()
    if sort:
        rates = rates.sort_values()
    plt.figure(figsize=(6, 4))
    if horizontal:
        plt.barh(rates.index.astype(str), rates.values, color=color)
        plt.xlabel("Diabetes Rate")
    else:
        plt.bar(rates.index.astype(str), rates.values, color=color)
        plt.ylabel("Diabetes Rate")
        plt.xlabel(col)
    plt.title(f"Diabetes Rate by {col.replace('_', ' ').title()}")
    plt.show()

rate_plot("smoking_history", "steelblue", horizontal=True, sort=True)

# %%
rate_plot("gender", "steelblue")

# %%
rate_plot("hypertension", "darkorange")

# %%
rate_plot("heart_disease", "darkorange")

# %% [markdown]
# ### Correlation matrix (numeric variables)

# %%
corr = df[numeric_vars].corr()
mask = np.tril(np.ones_like(corr, dtype=bool), k=-1)  # show upper triangle only

plt.figure(figsize=(6, 5))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1)
plt.title("Correlation Matrix — Numeric Predictors")
plt.show()

# %% [markdown]
# ### Age distribution

# %%
plt.figure(figsize=(7, 4))
sns.histplot(df["age"], binwidth=5, color="steelblue", edgecolor="white")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# %%
plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="age", hue="diabetes_f", hue_order=["No", "Yes"],
             binwidth=5, multiple="layer", alpha=0.5)
plt.title("Age Distribution by Diabetes Status")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# %% [markdown]
# The age spike at 80 is a **top-coding artifact** (80+ likely binned as 80), not a real population spike:

# %%
print("Rows at exactly age 80:", (df["age"] == 80).sum())
print("Rows aged 75-79:", ((df["age"] >= 75) & (df["age"] < 80)).sum())

# %% [markdown]
# ## 3. Train/Test Split
#
# Stratified 80/20 split on the target to preserve the ~91.5% / 8.5% class balance in both sets (a plain random split could shift this given the imbalance).

# %%
features = ["gender", "age", "hypertension", "heart_disease",
            "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]
num_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
cat_cols = ["gender", "hypertension", "heart_disease", "smoking_history"]

X = df[features]
y = df["diabetes"]  # 0 = No, 1 = Yes

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

for name, target in [("train", y_train), ("test", y_test)]:
    print(f"{name}:")
    print(pd.DataFrame({"n": target.value_counts(),
                        "prop": target.value_counts(normalize=True).round(4)}), "\n")

# %% [markdown]
# ### Helper functions

# %%
def make_preprocessor(scale=False):
    """One-hot encode categoricals; optionally standardise numerics (needed for logistic regression)."""
    num_step = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num_step, num_cols),
        ("cat", OneHotEncoder(drop="first" if scale else None), cat_cols),
    ])


def feature_names(pipe):
    names = pipe.named_steps["prep"].get_feature_names_out()
    return [n.split("__", 1)[1] for n in names]


def evaluate(y_true, prob, threshold=0.5, title=""):
    """Confusion matrix + metrics at a threshold, with 'Yes' (1) as the positive class."""
    pred = (prob > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    metrics = {
        "Accuracy": accuracy_score(y_true, pred),
        "Sensitivity (Recall)": recall_score(y_true, pred),
        "Specificity": tn / (tn + fp),
        "Precision": precision_score(y_true, pred),
        "AUC": roc_auc_score(y_true, prob),
    }
    print(f"=== {title} (threshold = {threshold}) ===")
    print(pd.DataFrame([[tn, fp], [fn, tp]],
                       index=["Actual No", "Actual Yes"],
                       columns=["Pred No", "Pred Yes"]))
    print(pd.Series(metrics).round(4).to_string())
    return metrics


def plot_roc(y_true, prob, title):
    RocCurveDisplay.from_predictions(y_true, prob)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.title(title)
    plt.show()

# %% [markdown]
# ## 4. Model 1 — Logistic Regression

# %%
log_pipe = Pipeline([
    ("prep", make_preprocessor(scale=True)),   # scaling helps the solver converge
    ("model", LogisticRegression(max_iter=1000)),
])
log_pipe.fit(X_train, y_train)

# Coefficients are on standardised numeric features (log-odds per 1 SD) and
# one-hot categoricals (log-odds vs. the reference level)
coefs = pd.DataFrame({
    "feature": feature_names(log_pipe),
    "coef": log_pipe.named_steps["model"].coef_[0],
})
coefs["odds_ratio"] = np.exp(coefs["coef"])
coefs.sort_values("coef", key=abs, ascending=False).round(4)

# %%
log_prob = log_pipe.predict_proba(X_test)[:, 1]
log_metrics = evaluate(y_test, log_prob, 0.5, "Logistic Regression")

# %%
plot_roc(y_test, log_prob, "ROC Curve — Logistic Regression")

# %% [markdown]
# ## 5. Model 2 — Decision Tree

# %%
# Settings chosen to give a small, readable tree similar to rpart's defaults
tree_pipe = Pipeline([
    ("prep", make_preprocessor(scale=False)),
    ("model", DecisionTreeClassifier(max_depth=5, min_samples_split=20,
                                     min_samples_leaf=7, random_state=RANDOM_STATE)),
])
tree_pipe.fit(X_train, y_train)

plt.figure(figsize=(20, 9))
plot_tree(tree_pipe.named_steps["model"], feature_names=feature_names(tree_pipe),
          class_names=["No", "Yes"], filled=True, rounded=True,
          max_depth=3, fontsize=9)
plt.title("Decision Tree — Diabetes Prediction (top 3 levels shown)")
plt.show()

# %%
tree_prob = tree_pipe.predict_proba(X_test)[:, 1]
tree_metrics = evaluate(y_test, tree_prob, 0.5, "Decision Tree")

# %%
plot_roc(y_test, tree_prob, "ROC Curve — Decision Tree")

# %%
# Overall contribution of each feature to the tree's decisions
tree_importance = pd.Series(tree_pipe.named_steps["model"].feature_importances_,
                            index=feature_names(tree_pipe)).sort_values(ascending=False)
tree_importance.round(4)

# %% [markdown]
# ## 6. Model 3 — Random Forest (final model)

# %%
rf_pipe = Pipeline([
    ("prep", make_preprocessor(scale=False)),
    ("model", RandomForestClassifier(n_estimators=200, oob_score=True,
                                     random_state=RANDOM_STATE, n_jobs=-1)),
])
rf_pipe.fit(X_train, y_train)

print("Out-of-bag accuracy:", round(rf_pipe.named_steps["model"].oob_score_, 4))

# %%
rf_prob = rf_pipe.predict_proba(X_test)[:, 1]
rf_metrics = evaluate(y_test, rf_prob, 0.5, "Random Forest")

# %%
plot_roc(y_test, rf_prob, "ROC Curve — Random Forest")

# %%
rf_importance = pd.Series(rf_pipe.named_steps["model"].feature_importances_,
                          index=feature_names(rf_pipe)).sort_values()

plt.figure(figsize=(7, 5))
rf_importance.plot(kind="barh", color="steelblue")
plt.title("Random Forest — Variable Importance")
plt.xlabel("Importance (mean decrease in impurity)")
plt.show()

# %% [markdown]
# ### Model comparison

# %%
pd.DataFrame({
    "Logistic Regression": log_metrics,
    "Decision Tree": tree_metrics,
    "Random Forest": rf_metrics,
}).T.round(4)

# %% [markdown]
# ## 7. Threshold Tuning (Random Forest)

# %%
rows = []
for t in [0.5, 0.4, 0.3, 0.2]:
    pred_t = (rf_prob > t).astype(int)
    rows.append({
        "threshold": t,
        "Sensitivity (Recall)": recall_score(y_test, pred_t),
        "Precision": precision_score(y_test, pred_t),
        "Accuracy": accuracy_score(y_test, pred_t),
    })

pd.DataFrame(rows).set_index("threshold").round(4)

# %% [markdown]
# **Final threshold: 0.3.** In the original R analysis this gave a meaningful recall gain (0.655 → 0.683) while precision stayed strong (0.946). Exact numbers can differ slightly in Python, so check the table above before finalising.

# %%
FINAL_THRESHOLD = 0.3
final_metrics = evaluate(y_test, rf_prob, FINAL_THRESHOLD, "Random Forest (final)")

# %% [markdown]
# ## 8. Save Final Model

# %%
out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
model_path = os.path.join(out_dir, "rf_model.joblib")

# Saves the full pipeline (encoding + model), so raw input rows can be passed straight in
joblib.dump(rf_pipe, model_path)
print("Saved:", model_path)

# Example: reload and predict with the final threshold
loaded = joblib.load(model_path)
sample = X_test.iloc[[0]]
prob = loaded.predict_proba(sample)[0, 1]
print(f"Predicted diabetes probability: {prob:.3f} ->",
      "Yes" if prob > FINAL_THRESHOLD else "No")