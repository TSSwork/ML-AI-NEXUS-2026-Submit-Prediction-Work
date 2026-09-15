"""
NEW — direct test of whether the train/test distribution shift your EDA
flagged (region/hospital_type/care_pathway/rurality mix) is large enough
to matter, using adversarial validation:

  1. Label every train row 0, every test row 1.
  2. Train a 5-fold LightGBM to predict which is which, on the SAME
     feature set (X.pkl / X_test.pkl) the Core-3 stack uses.
  3. If OOF ROC-AUC is meaningfully > 0.50, the two distributions really
     are separable -- i.e. there IS a real transportability gap, not
     just noise. If it's close to 0.50, the shift your EDA found in
     individual columns isn't translating into a jointly separable
     distribution, and the leaderboard gap likely has a different cause.

Also saves each train row's OOF "looks-like-test" probability, which
33_shift_reweighted_core3.py can turn into importance-sampling weights
if this script finds a real gap worth correcting for.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

ARTIFACTS = Path(__file__).resolve().parents[1] / "pipeline" / "artifacts"

with open(ARTIFACTS / "X.pkl", "rb") as f:
    X = pickle.load(f)
with open(ARTIFACTS / "X_test.pkl", "rb") as f:
    X_test = pickle.load(f)
with open(ARTIFACTS / "cat_features.pkl", "rb") as f:
    cat_features = pickle.load(f)

n_train, n_test = len(X), len(X_test)
X_adv = pd.concat([X, X_test], axis=0, ignore_index=True)
y_adv = np.concatenate([np.zeros(n_train), np.ones(n_test)])

for c in cat_features:
    X_adv[c] = X_adv[c].astype('category')

lgb_params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'learning_rate': 0.03, 'num_leaves': 16, 'max_depth': 4, 'min_child_samples': 35,
    'feature_fraction': 0.80, 'subsample': 0.85, 'reg_alpha': 0.5, 'reg_lambda': 1.0,
    'random_state': 42, 'verbose': -1, 'n_estimators': 500,
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_adv = np.zeros(len(X_adv))
importances = np.zeros(X_adv.shape[1])

print("--> Adversarial validation: can a model tell train rows from test rows?")
for trn_idx, val_idx in skf.split(X_adv, y_adv):
    X_tr, y_tr = X_adv.iloc[trn_idx], y_adv[trn_idx]
    X_va, y_va = X_adv.iloc[val_idx], y_adv[val_idx]
    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
    oof_adv[val_idx] = m.predict_proba(X_va)[:, 1]
    importances += m.feature_importances_ / 5

adv_auc = roc_auc_score(y_adv, oof_adv)

print("\n" + "=" * 58)
print("  ADVERSARIAL VALIDATION RESULT")
print("=" * 58)
print(f"OOF ROC-AUC (train vs. test separability): {adv_auc:.4f}")
if adv_auc < 0.55:
    verdict = "Weak/no separability -- the two distributions look statistically similar as a WHOLE."
elif adv_auc < 0.65:
    verdict = "Mild but real separability -- a genuine, moderate shift exists."
else:
    verdict = "Strong separability -- train and test are clearly drawn from different distributions."
print(verdict)
print("=" * 58)

fi = pd.DataFrame({'feature': X_adv.columns, 'importance': importances}).sort_values(
    'importance', ascending=False)
print("\nTop 10 features driving train/test separability:")
print(fi.head(10).to_string(index=False))

# Save the OOF "looks like test" probability for the TRAIN rows only --
# this is what a reweighting step would use as raw material for
# importance-sampling weights (w_i = p_test(x_i) / (1 - p_test(x_i))).
oof_adv_train_only = oof_adv[:n_train]
np.save(ARTIFACTS / "oof_adversarial_train.npy", oof_adv_train_only)
print(f"\nSaved train-row 'looks-like-test' probabilities to {ARTIFACTS / 'oof_adversarial_train.npy'}")
