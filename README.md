# Diabetes Risk Prediction (Python)

A machine learning project that predicts diabetes risk from patient health data. It compares three models (Logistic Regression, Decision Tree, Random Forest), tunes the decision threshold for the final Random Forest, and saves the trained pipeline for reuse.

An R version of this project also exists; this repository is the Python version.

## Dataset

[Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) by iammustafatz (Kaggle).

| Feature | Type | Description |
|---|---|---|
| `gender` | categorical | Female / Male |
| `age` | numeric | Age in years |
| `hypertension` | binary | 0 = No, 1 = Yes |
| `heart_disease` | binary | 0 = No, 1 = Yes |
| `smoking_history` | categorical | never, former, current, not current, ever, No Info |
| `bmi` | numeric | Body mass index |
| `HbA1c_level` | numeric | Average blood sugar level over ~3 months |
| `blood_glucose_level` | numeric | Blood glucose level |
| `diabetes` | target | 0 = No, 1 = Yes |

The target is imbalanced: about 91.5% No and 8.5% Yes.

## Workflow

1. **Preprocessing**
   - Dropped the 18 rows with gender "Other" (too few to be reliable).
   - Collapsed `smoking_history` from 6 to 4 categories: `never`, `former`, `current_or_past`, `No_Info`.
   - Kept "No Info" as its own category instead of imputing it, since it carries signal.
   - Kept infant ages and extreme BMI values, which are plausible.
2. **Exploratory analysis**: class balance, numeric predictors by diabetes status, diabetes rate by category, correlation matrix, and age distribution. The spike at age 80 is a top-coding artifact (80+ binned as 80).
3. **Stratified 80/20 train/test split** to preserve the class balance.
4. **Models**: Logistic Regression, Decision Tree, Random Forest (200 trees).
5. **Evaluation**: confusion matrix, accuracy, sensitivity (recall), specificity, precision, ROC curve and AUC.
6. **Threshold tuning** on the Random Forest (0.5, 0.4, 0.3, 0.2). A threshold of **0.3** is used as the final choice to improve recall on diabetic cases.
7. **Model saving**: the full pipeline (encoding + model) is saved with `joblib`.

## Results

Fill in after running the script.

| Model | Accuracy | Sensitivity | Specificity | Precision | AUC |
|---|---|---|---|---|---|
| Logistic Regression | | | | | |
| Decision Tree | | | | | |
| Random Forest | | | | | |

In the original R analysis, lowering the Random Forest threshold from 0.5 to 0.3 raised recall from 0.655 to 0.683 while precision stayed at 0.946.

## Project structure

```
.
├── diabetes_risk_predictor_py.py        # Main script
├── diabetes_risk_prediction_python.ipynb # Kaggle notebook version
├── diabetes_prediction_dataset.csv      # Dataset (download from Kaggle)
├── requirements.txt
├── SETUP_GUIDE.md                       # Windows virtual environment setup
└── README.md
```

## Getting started

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

**2. Download the dataset** from the Kaggle link above and place `diabetes_prediction_dataset.csv` in the project folder.

**3. Create a virtual environment and install the libraries**

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for a step-by-step Windows walkthrough and common errors.

**4. Run the script**

```powershell
.\.venv\Scripts\python.exe diabetes_risk_predictor_py.py
```

The script shows the plots, prints the metrics for each model, and saves `rf_model.joblib`.

## Run on Kaggle

1. Upload `diabetes_risk_prediction_python.ipynb` via **File → Import Notebook**.
2. Click **Add Input** and add the Diabetes Prediction Dataset.
3. Click **Run All**.

## Use the saved model

```python
import joblib
import pandas as pd

model = joblib.load("rf_model.joblib")

patient = pd.DataFrame([{
    "gender": "Female",
    "age": 54,
    "hypertension": "No",
    "heart_disease": "No",
    "smoking_history": "never",
    "bmi": 31.2,
    "HbA1c_level": 6.6,
    "blood_glucose_level": 160,
}])

prob = model.predict_proba(patient)[0, 1]
print("Diabetes risk:", "Yes" if prob > 0.3 else "No", f"({prob:.2f})")
```

Note that `hypertension` and `heart_disease` use the strings `"No"` / `"Yes"`, and `smoking_history` must use the collapsed categories (`never`, `former`, `current_or_past`, `No_Info`).

## Differences from the R version

- Logistic Regression uses standardised numeric features so the solver converges.
- The Decision Tree is configured to give a small, readable tree (`max_depth=5`), since scikit-learn has no exact equivalent of `rpart`.
- Random Forest variable importance uses mean decrease in impurity, which can rank features slightly differently from R.

## Tech stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, joblib

## Disclaimer

This is a student project built for learning. It is not a medical tool and must not be used for diagnosis.
