import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

def engineer_features(df):
    df = df.copy()
    if 'num' in df.columns:
        df['target'] = (df['num'] >= 1).astype(int)
        df = df.drop(columns=['num'])
    df.loc[df['chol'] == 0, 'chol'] = np.nan
    df.loc[df['trestbps'] == 0, 'trestbps'] = np.nan
    if 'fbs' in df.columns:
        df['fbs'] = df['fbs'].map({True: 1, False: 0, 'True': 1, 'False': 0})
    if 'exang' in df.columns:
        df['exang'] = df['exang'].map({True: 1, False: 0, 'True': 1, 'False': 0})
    for col in ['ca', 'thal', 'slope']:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isnull().astype(int)
    df = df.drop(columns=['id'], errors='ignore')
    if 'age' in df.columns and 'chol' in df.columns:
        df['age_x_chol'] = df['age'] * df['chol']
    if 'age' in df.columns and 'trestbps' in df.columns:
        df['bp_x_age'] = df['trestbps'] * df['age']
    if 'age' in df.columns and 'thalch' in df.columns:
        df['hr_reserve'] = df['thalch'] / (220 - df['age'] + 1)
    if 'oldpeak' in df.columns and 'exang' in df.columns:
        df['oldpeak_x_exang'] = df['oldpeak'] * df['exang']
    if 'trestbps' in df.columns and 'chol' in df.columns:
        df['trestbps_x_chol'] = df['trestbps'] * df['chol']
    if 'thalch' in df.columns and 'oldpeak' in df.columns:
        df['thalch_x_oldpeak'] = df['thalch'] * df['oldpeak']
    if all(c in df.columns for c in ['age', 'trestbps', 'chol']):
        df['risk_composite'] = df['age'] + df['trestbps'] + df['chol']
        df['risk_composite_sq'] = df['risk_composite'] ** 2
    return df

df = pd.read_csv('1_public.csv')
df = engineer_features(df)
X = df.drop(columns=['target'])
y = df['target']

model = joblib.load('model_final.pkl')
with open('best_threshold.txt', 'r') as f:
    best_t = float(f.read())

# Train F1
y_proba = model.predict_proba(X)[:, 1]
train_f1 = f1_score(y, (y_proba >= best_t).astype(int))
print(f'TRAIN_F1:{train_f1:.4f}')

# CV Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_f1s = []
for train_idx, val_idx in skf.split(X, y):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    from sklearn.base import clone
    pipe = clone(model)
    pipe.fit(X_tr, y_tr)
    val_proba = pipe.predict_proba(X_val)[:, 1]
    val_f1s.append(f1_score(y_val, (val_proba >= best_t).astype(int)))

print(f'VAL_F1:{np.mean(val_f1s):.4f}')
