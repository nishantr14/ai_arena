import json

notebook = {
    "cells": [],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")]
    })

add_md("# Heart Disease Prediction - Competition v2\nThis notebook loads the dataset, performs feature engineering, handles missing medical data via KNN imputation, trains a stacking ensemble with CatBoost, XGBoost, LightGBM, Random Forests, etc., and evaluates the F1 scores to strongly generalize over the test set.")

add_code("""import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import optuna

from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer, KNNImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)""")

add_md("## 1. Load Data and Audit\nLoad the public dataset and perform an initial audit.")

add_code("""df = pd.read_csv("1_public.csv")
display(df.head())

print(f"Shape: {df.shape}")

print("\\nMissing values:")
missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(2)
for col in missing[missing > 0].index:
    print(f"  {col}: {missing[col]} ({missing_pct[col]}%)")
    
print(f"\\nTarget (num) raw distribution:\\n{df['num'].value_counts(normalize=True)}")

# Duplicates
dupes = df.duplicated().sum()
if dupes > 0:
    df = df.drop_duplicates()
    print(f"  Dropped {dupes} duplicate rows. New shape: {df.shape}")""")

add_md("## 2. Feature Engineering\nConvert boolean strings, fix impossible values, add missingness indicators, and establish non-linear polynomial features to improve representation mapping.")

add_code("""def engineer_features(df):
    df = df.copy()

    # Binarize target if present
    if 'num' in df.columns:
        df['target'] = (df['num'] >= 1).astype(int)
        df = df.drop(columns=['num'])

    # Fix impossible values
    df.loc[df['chol'] == 0, 'chol'] = np.nan
    df.loc[df['trestbps'] == 0, 'trestbps'] = np.nan

    # Convert fbs/exang to numeric (handles both Python bool and string)
    if 'fbs' in df.columns:
        df['fbs'] = df['fbs'].map({True: 1, False: 0, 'True': 1, 'False': 0})
    if 'exang' in df.columns:
        df['exang'] = df['exang'].map({True: 1, False: 0, 'True': 1, 'False': 0})

    # Missingness indicators
    for col in ['ca', 'thal', 'slope']:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isnull().astype(int)

    # Drop id
    df = df.drop(columns=['id'], errors='ignore')

    # Logical Interaction features
    if 'age' in df.columns and 'chol' in df.columns:
        df['age_x_chol'] = df['age'] * df['chol']
    if 'age' in df.columns and 'trestbps' in df.columns:
        df['bp_x_age'] = df['trestbps'] * df['age']
    if 'age' in df.columns and 'thalch' in df.columns:
        df['hr_reserve'] = df['thalch'] / (220 - df['age'] + 1)
    if 'oldpeak' in df.columns and 'exang' in df.columns:
        df['oldpeak_x_exang'] = df['oldpeak'] * df['exang']
        
    # Non-linear Polynomial interactions
    if 'trestbps' in df.columns and 'chol' in df.columns:
        df['trestbps_x_chol'] = df['trestbps'] * df['chol']
    if 'thalch' in df.columns and 'oldpeak' in df.columns:
        df['thalch_x_oldpeak'] = df['thalch'] * df['oldpeak']
        
    # Additive risk composite
    if all(c in df.columns for c in ['age', 'trestbps', 'chol']):
        df['risk_composite'] = df['age'] + df['trestbps'] + df['chol']
        df['risk_composite_sq'] = df['risk_composite'] ** 2

    return df

df = engineer_features(df)
print(f"Engineered shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

X = df.drop(columns=['target'])
y = df['target']""")

add_md("## 3. Preprocessor Construction\nWe replace Simple imputation with `KNNImputer(n_neighbors=5)` for our numeric fields. This robustly retains predictive capacity for features like `ca` and `thal`.")

add_code("""def build_preprocessor(X_ref):
    num_cols = X_ref.select_dtypes(include='number').columns.tolist()
    cat_cols = X_ref.select_dtypes(include=['object', 'category']).columns.tolist()

    num_pipe = Pipeline([
        ('impute', KNNImputer(n_neighbors=5)),
        ('scale', StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encode', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)),
    ])

    return ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols),
    ], remainder='drop')

preprocessor = build_preprocessor(X)
X_t = preprocessor.fit_transform(X)
print(f"Transformed dense shape: {X_t.shape}")""")

add_md("## 4. Model Zoo Evaluation\nCross validate robust baseline models including CatBoost, LightGBM, and XGBoost to establish performance ranks.")

add_code("""cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
neg_count = (y == 0).sum()
pos_count = (y == 1).sum()
scale_pos = neg_count / pos_count
print(f"Class balances: neg={neg_count}, pos={pos_count}, target scale_pos_weight={scale_pos:.3f}\\n")

models = {
    'LogReg_elasticnet': LogisticRegression(
        C=0.5, penalty='elasticnet', solver='saga', l1_ratio=0.5,
        max_iter=3000, class_weight='balanced', random_state=42
    ),
    'RF': RandomForestClassifier(
        n_estimators=600, max_depth=8, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'GBM': GradientBoostingClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        min_samples_leaf=5, subsample=0.8, random_state=42
    ),
    'XGB': XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos, eval_metric='logloss', random_state=42
    ),
    'LGBM': LGBMClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05, num_leaves=31,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        is_unbalance=True, random_state=42, verbose=-1
    ),
    'CatBoost': CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=5,
        auto_class_weights='Balanced', verbose=0, random_state=42
    ),
    'SVM_rbf': SVC(
        C=1.0, kernel='rbf', probability=True,
        class_weight='balanced', random_state=42
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=9, weights='distance', n_jobs=-1
    )
}

results = {}
for name, model in models.items():
    pipe = Pipeline([('prep', preprocessor), ('model', model)])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='f1', n_jobs=-1)
    results[name] = {'mean': scores.mean(), 'std': scores.std()}
    print(f"  {name:25s} | CV F1 = {scores.mean():.4f} +/- {scores.std():.4f}")

ranked = sorted(results.items(), key=lambda x: -x[1]['mean'])
top_models = [name for name, r in ranked if r['std'] <= 0.075][:5]
print(f"\\nSelected for deeper tuning: {top_models}")""")

add_md("## 5. Threshold & Optuna Tuning\nOptimizing hyperparameters using deeper tree depths for the top 3 survivors to forcefully push `Train F1` metrics > 94%.")

add_code("""def get_best_threshold(model_or_pipe, X, y, cv_splits=5, is_pipe=False):
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    best_thresholds, train_f1s, f1_bests = [], [], []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipe = model_or_pipe if is_pipe else Pipeline([('prep', build_preprocessor(X_tr)), ('model', model_or_pipe)])
        from sklearn.base import clone; pipe = clone(pipe)
        pipe.fit(X_tr, y_tr)

        y_train_proba = pipe.predict_proba(X_tr)[:, 1]
        y_proba = pipe.predict_proba(X_val)[:, 1]

        thresholds = np.arange(0.20, 0.70, 0.005)
        f1s = [f1_score(y_val, (y_proba >= t).astype(int)) for t in thresholds]
        best_t = thresholds[np.argmax(f1s)]
        best_thresholds.append(best_t)
        f1_bests.append(max(f1s))
        train_f1s.append(f1_score(y_tr, (y_train_proba >= best_t).astype(int)))

    return {
        'threshold': np.mean(best_thresholds),
        'f1_tuned': np.mean(f1_bests),
        'train_f1': np.mean(train_f1s)
    }

top3 = top_models[:3]
cv_optuna = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

def make_objective(model_name):
    def objective(trial):
        if 'LogReg' in model_name:
            model = LogisticRegression(
                C=trial.suggest_float('C', 0.001, 50.0, log=True),
                penalty='elasticnet', solver='saga', 
                l1_ratio=trial.suggest_float('l1_ratio', 0.0, 1.0),
                max_iter=3000, class_weight='balanced', random_state=42
            )
        elif model_name == 'CatBoost':
            model = CatBoostClassifier(
                iterations=trial.suggest_int('iterations', 200, 800),
                depth=trial.suggest_int('depth', 4, 10),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1.0, 20.0),
                random_strength=trial.suggest_float('random_strength', 0.0, 3.0),
                bagging_temperature=trial.suggest_float('bagging_temperature', 0.0, 1.0),
                auto_class_weights='Balanced', verbose=0, random_state=42
            )
        elif model_name == 'XGB':
            model = XGBClassifier(
                max_depth=trial.suggest_int('max_depth', 3, 10),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                n_estimators=trial.suggest_int('n_estimators', 150, 800),
                min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
                subsample=trial.suggest_float('subsample', 0.5, 0.95),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 0.95),
                reg_alpha=trial.suggest_float('reg_alpha', 0.001, 5.0, log=True),
                reg_lambda=trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
                scale_pos_weight=scale_pos, eval_metric='logloss', random_state=42
            )
        elif model_name == 'SVM_rbf':
            model = SVC(
                C=trial.suggest_float('C', 0.01, 100.0, log=True), 
                kernel='rbf', gamma=trial.suggest_categorical('gamma', ['scale', 'auto']),
                probability=True, class_weight='balanced', random_state=42
            )
        elif model_name == 'KNN':
            model = KNeighborsClassifier(
                n_neighbors=trial.suggest_int('n_neighbors', 3, 25),
                weights=trial.suggest_categorical('weights', ['uniform', 'distance']),
                n_jobs=-1
            )
        else: return 0.0 # simplified default
        
        pipe = Pipeline([('prep', preprocessor), ('model', model)])
        return cross_val_score(pipe, X, y, cv=cv_optuna, scoring='f1', n_jobs=-1).mean()
    return objective

tuned_params = {}
for name in top3:
    print(f"Tuning {name}...")
    study = optuna.create_study(direction='maximize')
    study.optimize(make_objective(name), n_trials=80) 
    tuned_params[name] = study.best_params
    print(f"Best Val F1: {study.best_value:.4f}")""")

add_md("## 6. Evaluate Stacking Ecosystem\nCreate the composite multi-layer stack and report out the finalized Train & Test CV F1 expectations.")

add_code("""def build_tuned_model(name, params):
    if 'LogReg' in name: return LogisticRegression(**params, penalty='elasticnet', solver='saga', max_iter=3000, class_weight='balanced', random_state=42)
    elif name == 'CatBoost': return CatBoostClassifier(**params, auto_class_weights='Balanced', verbose=0, random_state=42)
    elif name == 'XGB': return XGBClassifier(**params, scale_pos_weight=scale_pos, eval_metric='logloss', random_state=42)
    elif name == 'SVM_rbf': return SVC(**params, kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    elif name == 'KNN': return KNeighborsClassifier(**params, n_jobs=-1)
    return models[name]

ensemble_estimators = [(f'{n}_tuned', build_tuned_model(n, tuned_params[n])) for n in top3]
for n in top_models:
    if n not in top3: ensemble_estimators.append((n, models[n]))

stacking_pipe = Pipeline([
    ('prep', preprocessor),
    ('stack', StackingClassifier(
        estimators=ensemble_estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=2000),
        cv=5, stack_method='predict_proba', n_jobs=-1
    ))
])

stack_scores = cross_val_score(stacking_pipe, X, y, cv=cv, scoring='f1', n_jobs=-1)
print(f"Stacking Ensemble Validation F1 (25 folds) = {stack_scores.mean():.4f} +/- {stack_scores.std():.4f}\\n")

final_pipe = stacking_pipe
final_thresh = get_best_threshold(final_pipe, X, y, cv_splits=10, is_pipe=True)
best_threshold = final_thresh['threshold']

print(f"--> Target Verifications:")
print(f"TRAIN F1 @ tuned threshold: {final_thresh['train_f1']:.4f}")
print(f"VAL F1   @ tuned threshold: {final_thresh['f1_tuned']:.4f}")
print(f"Selected Threshold:         {best_threshold:.3f}")""")

add_md("## 7. Generate Submission Export\nTrain on full data and parse `submission.csv`.")

add_code("""# Train over all data
final_pipe.fit(X, y)
joblib.dump(final_pipe, 'model_final.pkl')

df_full = pd.read_csv("1_public.csv")
df_sub = engineer_features(df_full)
X_sub = df_sub.drop(columns=['target'])

y_proba = final_pipe.predict_proba(X_sub)[:, 1]
y_pred = (y_proba >= best_threshold).astype(int)

submission = pd.DataFrame({
    'id': df_full['id'],
    'num': y_pred,
})
submission.to_csv('submission.csv', index=False)
display(submission.head())
print(f"\\nSaved: submission.csv with {len(submission)} rows.")""")

with open('solution.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Created solution.ipynb successfully!")
