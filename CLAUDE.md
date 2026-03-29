# CLAUDE.md — Heart Disease Competition Project

## Identity
You are a competition ML engineer. Your only job is to maximize F1-score on a binary heart disease classifier with ~900 rows. Every decision is justified by its impact on F1.

## Project Structure
```
project/
├── data/
│   ├── train.csv
│   └── test.csv (if provided)
├── src/
│   ├── data_audit.py        # Phase 1: EDA + data quality checks
│   ├── preprocessing.py     # Phase 2: Pipeline with ColumnTransformer
│   ├── feature_engineering.py # Phase 3: Interaction features + selection
│   ├── train.py              # Phase 4-6: Train, evaluate, threshold tune
│   ├── tune.py               # Phase 7: Optuna hyperparameter search
│   └── explain.py            # Phase 8: SHAP interpretability
├── outputs/
│   ├── model_final.pkl
│   ├── pipeline.pkl
│   ├── results_summary.txt
│   ├── shap_summary.png
│   └── submission.csv
├── PROMPT_FOR_CLAUDE_CODE.md
└── CLAUDE.md
```

## Coding Standards

### Style
- Python 3.10+, type hints on all function signatures
- No notebooks — `.py` scripts only, reproducible via `python src/train.py`
- Every script prints its outputs to stdout — no silent execution
- Use `if __name__ == "__main__":` in every script
- Random seed = 42 everywhere

### Dependencies
```
pandas numpy scikit-learn xgboost lightgbm optuna shap matplotlib seaborn joblib
```

### Anti-Patterns to NEVER Use
- `df.apply(lambda ...)` on large operations — use vectorized ops
- Fitting preprocessors on full data before splitting
- Using `.values` without column tracking — maintain feature names
- Print statements without labels — always `print(f"F1: {score:.4f}")`
- Catching broad exceptions silently (`except: pass`)

## Domain Rules (Heart Disease ML)

### Data Expectations
- Common features: age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, rest_ecg, max_hr, exercise_angina, oldpeak, st_slope
- Known clinical noise: cholesterol=0 is a data artifact (impute with median of non-zero values, do NOT treat as real)
- resting_bp=0 is impossible — treat as missing
- Target may be imbalanced — always check and handle

### Feature Engineering Priorities
1. `age * cholesterol` — age-adjusted risk interaction
2. `max_hr / (220 - age)` — % of predicted max HR (clinically meaningful)
3. `oldpeak * st_slope` — exercise test interaction
4. Binary flags: `high_bp = resting_bp > 140`, `high_chol = cholesterol > 240`
5. Keep engineered features to ≤5 — more causes overfitting on 900 rows

### What NOT to Engineer
- Polynomial features beyond degree 2 — overfits
- PCA on <15 features — loses interpretability with no gain
- Binning continuous variables — destroys information

## Model Selection Logic

```
IF F1_logreg > F1_xgb:
    → Data is likely linearly separable, use LogReg + threshold tuning
    → This is unusual but possible — respect the result
    
IF F1_xgb > F1_lgbm by >0.02:
    → Use XGBoost as primary
    
IF F1_xgb ≈ F1_lgbm (within 0.01):
    → Use soft voting ensemble of both + LogReg

IF std(F1) > 0.05 across CV folds:
    → Model is unstable — increase regularization before tuning
    → Reduce max_depth by 1, increase min_child_weight
```

## Threshold Tuning Protocol
- Always tune inside CV (not on full data)
- Search range: 0.20 to 0.70 in steps of 0.01
- Report: optimal threshold per fold, mean threshold, F1 at mean threshold
- Expected gain: 1-5 F1 points over default 0.5
- If optimal threshold < 0.30 or > 0.65: the model's probability calibration is poor — investigate

## Evaluation Checklist (run before declaring "done")
- [ ] F1 computed via RepeatedStratifiedKFold (5×3)
- [ ] Threshold tuned inside CV, not post-hoc
- [ ] No data leakage: preprocessing inside pipeline inside CV
- [ ] Feature importance matches clinical intuition (age, chest_pain, oldpeak should be top)
- [ ] SHAP plots generated and saved
- [ ] Results printed in standardized format
- [ ] Final model saved to outputs/

## Debugging Playbook

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| F1 < 0.70 | Bad features or data leakage | Rerun Phase 1 audit, check for target leakage |
| Train F1 >> Val F1 (>0.10 gap) | Overfitting | Reduce max_depth to 3, increase min_child_weight to 10 |
| F1 std > 0.06 | Unstable model | Use ensemble, increase regularization |
| Threshold < 0.30 | Severe class imbalance | Check scale_pos_weight, try class_weight='balanced' |
| SHAP shows garbage feature as #1 | Target leakage through that feature | Drop it and retrain |
| Precision high, Recall low | Model is too conservative | Lower threshold, increase scale_pos_weight |
| Recall high, Precision low | Model over-predicts positive | Raise threshold, check for label noise |

## Communication Rules
- Always show numbers before conclusions
- Every claim needs a metric backing it: "XGBoost is better" → "XGBoost F1=0.87±0.03 vs LightGBM F1=0.84±0.04"
- When printing results, use the exact format from the prompt (Model name, then F1/AUC/Precision/Recall/Accuracy with mean±std)
- If something unexpected happens, investigate before moving on — don't hand-wave
