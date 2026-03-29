"""
Heart Disease Prediction — Competition v2
Maximize F1 on hidden test set. Generalization is everything.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold, cross_val_score, cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, PolynomialFeatures
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
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# STEP 1: LOAD AND AUDIT
# ============================================================
print("=" * 60)
print("STEP 1: LOAD AND AUDIT")
print("=" * 60)

df = pd.read_csv("1_public.csv")

print(f"\nShape: {df.shape}")
print(f"\nDtypes:\n{df.dtypes}")

# Missing values
missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(2)
print(f"\nMissing values:")
for col in missing[missing > 0].index:
    print(f"  {col}: {missing[col]} ({missing_pct[col]}%)")

# Target distribution (raw multi-class)
print(f"\nTarget (num) raw distribution:\n{df['num'].value_counts(normalize=True)}")

# Garbage value check
for col in df.select_dtypes(include='number').columns:
    print(f"  {col}: min={df[col].min()}, max={df[col].max()}")

# Categorical value counts
for col in df.select_dtypes(include=['object', 'category']).columns:
    print(f"  {col}: {df[col].value_counts().to_dict()}")

# Duplicates
dupes = df.duplicated().sum()
print(f"\nDuplicate rows: {dupes}")
if dupes > 0:
    df = df.drop_duplicates()
    print(f"  Dropped. New shape: {df.shape}")

print("\n--- AUDIT FINDINGS ---")
print("- ca: KEEP and impute via KNN, create missingness flag")
print("- thal: KEEP and impute via KNN, create missingness flag")
print("- slope: KEEP and impute, create missingness flag")
print("- chol has 0 values (impossible) → set to NaN")
print("- trestbps has 0 values (impossible) → set to NaN")
print("- Target is multi-class (0-4) → binarize: 0=Low, 1+=High")
print("- fbs/exang are boolean strings → convert to numeric")

# ============================================================
# STEP 2 & 3: FEATURE ENGINEERING + PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("STEP 2 & 3: FEATURE ENGINEERING + PREPROCESSING")
print("=" * 60)

def engineer_features(df):
    """Feature engineering — IDENTICAL for train and test."""
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

    # Missingness indicators for high-missing columns
    for col in ['ca', 'thal', 'slope']:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isnull().astype(int)

    # Note: We are NO LONGER dropping 'ca' and 'thal', keeping them to retain signal

    # Drop id
    df = df.drop(columns=['id'], errors='ignore')

    # Interaction features
    if 'age' in df.columns and 'chol' in df.columns:
        df['age_x_chol'] = df['age'] * df['chol']
    if 'age' in df.columns and 'trestbps' in df.columns:
        df['bp_x_age'] = df['trestbps'] * df['age']
    if 'age' in df.columns and 'thalch' in df.columns:
        df['hr_reserve'] = df['thalch'] / (220 - df['age'] + 1)
    if 'oldpeak' in df.columns and 'exang' in df.columns:
        df['oldpeak_x_exang'] = df['oldpeak'] * df['exang']
        
    # Extra Polynomial Interactions for non-linear patterns
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
print(f"\nTarget distribution:\n{df['target'].value_counts(normalize=True)}")
print(f"\nRemaining missing:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# Separate X, y
X = df.drop(columns=['target'])
y = df['target']


def build_preprocessor(X_ref):
    """Build preprocessor. Uses KNNImputer for numeric instead of median."""
    num_cols = X_ref.select_dtypes(include='number').columns.tolist()
    cat_cols = X_ref.select_dtypes(include=['object', 'category']).columns.tolist()

    # Robust numeric imputation
    num_pipe = Pipeline([
        ('impute', KNNImputer(n_neighbors=5)),
        ('scale', StandardScaler()),
    ])

    # Categorical handles NaNs by creating a 'missing_value' bucket
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encode', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols),
    ], remainder='drop')

    return preprocessor


preprocessor = build_preprocessor(X)

X_t = preprocessor.fit_transform(X)
print(f"\nTransformed shape: {X_t.shape}")

# ============================================================
# STEP 4: MODEL ZOO
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: MODEL ZOO — ALL MODELS, SAME CV")
print("=" * 60)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

neg_count = (y == 0).sum()
pos_count = (y == 1).sum()
scale_pos = neg_count / pos_count
print(f"Class balance: neg={neg_count}, pos={pos_count}, scale_pos_weight={scale_pos:.3f}")

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
    ),
}

results = {}
for name, model in models.items():
    pipe = Pipeline([('prep', preprocessor), ('model', model)])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='f1', n_jobs=-1)
    results[name] = {'mean': scores.mean(), 'std': scores.std(), 'scores': scores}
    print(f"  {name:25s}  F1 = {scores.mean():.4f} +/- {scores.std():.4f}")

print("\n=== RANKED ===")
ranked = sorted(results.items(), key=lambda x: -x[1]['mean'])
for i, (name, r) in enumerate(ranked):
    flag = " ** UNSTABLE" if r['std'] > 0.06 else ""
    print(f"  {i+1}. {name:25s}  F1 = {r['mean']:.4f} +/- {r['std']:.4f}{flag}")

# Keep top 5, exclude severely unstable
top_models = []
for name, r in ranked:
    if r['std'] <= 0.075 and len(top_models) < 5:
        top_models.append(name)
print(f"\nSurvivors for threshold tuning: {top_models}")

# ============================================================
# STEP 5: THRESHOLD TUNING FOR SURVIVORS
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: THRESHOLD TUNING")
print("=" * 60)

def get_best_threshold(model_or_pipe, X, y, cv_splits=5, is_pipe=False):
    """Tune threshold inside CV."""
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    best_thresholds = []
    f1_defaults = []
    f1_bests = []
    
    # Track training F1 as well to monitor hitting >94%
    train_f1s = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if is_pipe:
            pipe = model_or_pipe
        else:
            pipe = Pipeline([('prep', build_preprocessor(X_tr)), ('model', model_or_pipe)])
        from sklearn.base import clone
        pipe = clone(pipe)
        pipe.fit(X_tr, y_tr)

        # Train prediction
        y_train_proba = pipe.predict_proba(X_tr)[:, 1]
        
        # Val prediction
        y_proba = pipe.predict_proba(X_val)[:, 1]

        f1_def = f1_score(y_val, (y_proba >= 0.5).astype(int))
        f1_defaults.append(f1_def)

        thresholds = np.arange(0.20, 0.70, 0.005)
        f1s = [f1_score(y_val, (y_proba >= t).astype(int)) for t in thresholds]
        best_t = thresholds[np.argmax(f1s)]
        best_f1 = max(f1s)
        
        train_f1 = f1_score(y_tr, (y_train_proba >= best_t).astype(int))
        train_f1s.append(train_f1)

        best_thresholds.append(best_t)
        f1_bests.append(best_f1)

    return {
        'threshold': np.mean(best_thresholds),
        'threshold_std': np.std(best_thresholds),
        'f1_default': np.mean(f1_defaults),
        'f1_tuned': np.mean(f1_bests),
        'gain': np.mean(f1_bests) - np.mean(f1_defaults),
        'train_f1': np.mean(train_f1s)
    }

threshold_results = {}
for name in top_models:
    model = models[name]
    tr = get_best_threshold(model, X, y, cv_splits=5)
    threshold_results[name] = tr
    print(f"  {name:25s}  t={tr['threshold']:.3f} +/- {tr['threshold_std']:.2f} | "
          f"Train F1={tr['train_f1']:.4f} | Val F1@best={tr['f1_tuned']:.4f}")

# ============================================================
# STEP 6: OPTUNA TUNING FOR TOP 3 MODELS
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: OPTUNA TUNING (top 3 models, 80 trials each)")
print("=" * 60)

# Pick top 3 by CV F1 mean
top3 = top_models[:3]
print(f"Tuning: {top3}")

cv_optuna = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

def make_objective(model_name):
    """Create an Optuna objective for the given model type."""

    def objective(trial):
        if 'LogReg' in model_name:
            C = trial.suggest_float('C', 0.001, 50.0, log=True)
            l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
            model = LogisticRegression(
                C=C, penalty='elasticnet', solver='saga', l1_ratio=l1_ratio,
                max_iter=3000, class_weight='balanced', random_state=42
            )
        elif model_name == 'CatBoost':
            params = {
                'iterations': trial.suggest_int('iterations', 200, 800),
                'depth': trial.suggest_int('depth', 4, 10), # deeper bounds
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 20.0),
                'random_strength': trial.suggest_float('random_strength', 0.0, 3.0),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            }
            model = CatBoostClassifier(**params, auto_class_weights='Balanced', verbose=0, random_state=42)
        elif model_name == 'XGB':
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10), # deeper bounds
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 150, 800),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 5.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            }
            model = XGBClassifier(**params, scale_pos_weight=scale_pos, eval_metric='logloss', random_state=42)
        elif model_name == 'LGBM':
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10), # deeper bounds
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 150, 800),
                'num_leaves': trial.suggest_int('num_leaves', 7, 127),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
                'subsample': trial.suggest_float('subsample', 0.5, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 5.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            }
            model = LGBMClassifier(**params, is_unbalance=True, random_state=42, verbose=-1)
        elif model_name == 'GBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 150, 800),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 20),
                'subsample': trial.suggest_float('subsample', 0.5, 0.95),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),
            }
            model = GradientBoostingClassifier(**params, random_state=42)
        elif model_name in ('RF', 'ExtraTrees'):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 15),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),
            }
            cls = RandomForestClassifier if model_name == 'RF' else ExtraTreesClassifier
            model = cls(**params, class_weight='balanced', random_state=42, n_jobs=-1)
        elif model_name == 'SVM_rbf':
            C = trial.suggest_float('C', 0.01, 100.0, log=True)
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
            model = SVC(C=C, kernel='rbf', gamma=gamma, probability=True,
                        class_weight='balanced', random_state=42)
        elif model_name == 'KNN':
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 25),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            }
            model = KNeighborsClassifier(**params, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        pipe = Pipeline([('prep', preprocessor), ('model', model)])
        scores = cross_val_score(pipe, X, y, cv=cv_optuna, scoring='f1', n_jobs=-1)
        return scores.mean()

    return objective

tuned_params = {}
tuned_f1 = {}
for name in top3:
    print(f"\n  Tuning {name} (80 trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(make_objective(name), n_trials=80)
    tuned_params[name] = study.best_params
    tuned_f1[name] = study.best_value
    print(f"  {name} best Validation F1: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")


def build_tuned_model(name, params):
    """Reconstruct model from Optuna params."""
    if 'LogReg' in name:
        return LogisticRegression(
            C=params['C'], penalty='elasticnet', solver='saga',
            l1_ratio=params['l1_ratio'], max_iter=3000,
            class_weight='balanced', random_state=42
        )
    elif name == 'CatBoost':
        return CatBoostClassifier(**params, auto_class_weights='Balanced', verbose=0, random_state=42)
    elif name == 'XGB':
        return XGBClassifier(**params, scale_pos_weight=scale_pos, eval_metric='logloss', random_state=42)
    elif name == 'LGBM':
        return LGBMClassifier(**params, is_unbalance=True, random_state=42, verbose=-1)
    elif name == 'GBM':
        return GradientBoostingClassifier(**params, random_state=42)
    elif name in ('RF', 'ExtraTrees'):
        cls = RandomForestClassifier if name == 'RF' else ExtraTreesClassifier
        return cls(**params, class_weight='balanced', random_state=42, n_jobs=-1)
    elif name == 'SVM_rbf':
        return SVC(**params, kernel='rbf', probability=True,
                    class_weight='balanced', random_state=42)
    elif name == 'KNN':
        return KNeighborsClassifier(**params, n_jobs=-1)
    else:
        return models[name]


# ============================================================
# STEP 7: STACKING ENSEMBLE
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: STACKING ENSEMBLE")
print("=" * 60)

model_families = {
    'LogReg_elasticnet': 'linear',
    'RF': 'bagging', 'ExtraTrees': 'bagging',
    'GBM': 'boosting', 'XGB': 'boosting', 'LGBM': 'boosting', 'CatBoost': 'boosting', 
    'SVM_rbf': 'svm', 'KNN': 'distance',
}

ensemble_estimators = []
used_families = set()
for name in top3:
    est = build_tuned_model(name, tuned_params[name])
    ensemble_estimators.append((f'{name}_tuned', est))
    used_families.add(model_families.get(name, name))

for name in top_models:
    if name in top3:
        continue
    family = model_families.get(name, name)
    if family not in used_families:
        ensemble_estimators.append((name, models[name]))
        used_families.add(family)

if len(ensemble_estimators) < 4:
    for name in top_models:
        if name not in [n for n, _ in ensemble_estimators]:
            ensemble_estimators.append((name, models[name]))
            if len(ensemble_estimators) >= 4:
                break

print(f"Ensemble members ({len(ensemble_estimators)}):")
for name, _ in ensemble_estimators:
    print(f"  - {name} (family: {model_families.get(name.replace('_tuned', ''), '?')})")

stacking_pipe = Pipeline([
    ('prep', preprocessor),
    ('stack', StackingClassifier(
        estimators=ensemble_estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=2000),
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1,
    ))
])

print("\n  Scoring stacking ensemble (25 folds)...")
stack_scores = cross_val_score(stacking_pipe, X, y, cv=cv, scoring='f1', n_jobs=-1)
print(f"  Stacking Val F1 = {stack_scores.mean():.4f} +/- {stack_scores.std():.4f}")

best_single_name = max(tuned_f1, key=tuned_f1.get)
best_single_model = build_tuned_model(best_single_name, tuned_params[best_single_name])
best_single_pipe = Pipeline([('prep', preprocessor), ('model', best_single_model)])
single_scores = cross_val_score(best_single_pipe, X, y, cv=cv, scoring='f1', n_jobs=-1)
print(f"  Best single ({best_single_name}) Val F1 = {single_scores.mean():.4f} +/- {single_scores.std():.4f}")

stack_advantage = stack_scores.mean() - single_scores.mean()
print(f"\n  Stacking advantage: {stack_advantage:+.4f}")
if stack_advantage >= 0.005:
    final_pipe = stacking_pipe
    final_name = "Stacking Ensemble"
    final_cv_scores = stack_scores
else:
    final_pipe = best_single_pipe
    final_name = f"Single: {best_single_name}"
    final_cv_scores = single_scores

# ============================================================
# STEP 8: FINAL THRESHOLD TUNING (Tracks Test & Train F1 metrics carefully)
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: FINAL METRICS VERIFICATION")
print("=" * 60)

final_thresh = get_best_threshold(final_pipe, X, y, cv_splits=10, is_pipe=True)
best_threshold = final_thresh['threshold']
print(f"  Final threshold selected: {best_threshold:.3f}")
print(f"  --> TRAIN F1 @ tuned threshold: {final_thresh['train_f1']:.4f}")
print(f"  --> VAL F1   @ tuned threshold: {final_thresh['f1_tuned']:.4f}")

# Target verification:
if final_thresh['train_f1'] > 0.94 and final_thresh['f1_tuned'] > 0.89:
    print(f"  [SUCCESS] Both targets met! (Train > 94%, Val > 89%)")
else:
    print(f"  [WARNING] Target check: Train F1={final_thresh['train_f1']:.4f}, Val F1={final_thresh['f1_tuned']:.4f}")

# ============================================================
# STEP 9: TRAIN FINAL + PREDICTIONS
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: TRAIN FULL PIPELINE & EXPORT")
print("=" * 60)

final_pipe.fit(X, y)
joblib.dump(final_pipe, 'model_final.pkl')
print("Saved: model_final.pkl")

with open('best_threshold.txt', 'w') as f:
    f.write(str(best_threshold))

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
print(f"Saved: submission.csv ({len(submission)} rows)")

print("\n" + "=" * 60)
print("FINAL COMPETITION SUMMARY")
print("=" * 60)
print(f"  Model:              {final_name}")
print(f"  TRAIN F1:           {final_thresh['train_f1']:.4f}")
print(f"  VAL F1:             {final_thresh['f1_tuned']:.4f}")
print(f"  Optimal threshold:  {best_threshold:.3f}")
print("=" * 60)
