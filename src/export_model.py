"""
export_model.py
Martin Silar, SPSE Jecna C4c

Nacte health_daily_v5.csv, natrenuje klasifikacni model
a ulozi ho do models/sleep_model.pkl.
Spustte po transformation.py.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, 'data', 'health_daily_v5.csv')
MODEL_DIR = os.path.join(_ROOT, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'sleep_model.pkl')

N_LAGS = 7

MODEL_FEATURES = [
    'prev1_hr', 'prev2_hr', 'prev3_hr', 'prev4_hr',
    'prev5_hr', 'prev6_hr', 'prev7_hr',
    'rolling7_hr', 'rolling3_hr',
    'dow_sin', 'dow_cos', 'is_weekend',
    'steps_today', 'kcal_today',
]


def build_features(df):
    df = df.copy().sort_values('date').reset_index(drop=True)

    boundary = float(df['sleep_score'].median())
    df['good_night'] = (df['sleep_score'] > boundary).astype(int)

    for i in range(1, N_LAGS + 1):
        df[f'prev{i}_hr'] = df['hr_night_avg'].shift(i)

    df['rolling3_hr'] = df['hr_night_avg'].shift(1).rolling(3).mean()
    df['rolling7_hr'] = df['hr_night_avg'].shift(1).rolling(7).mean()

    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    df['steps_today'] = df['steps_total']
    df['kcal_today']  = df['active_kcal']

    return df, boundary


def tune_threshold(clf, X_val, y_val):
    probas = clf.predict_proba(X_val)[:, 1]
    best_t, best_acc = 0.5, 0.0
    for t in np.arange(0.35, 0.66, 0.01):
        pred = (probas >= t).astype(int)
        acc = accuracy_score(y_val, pred)
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t


def main():
    if not os.path.exists(DATA_PATH):
        print(f"Chyba: {DATA_PATH} nenalezen - spustte nejdrive transformation.py")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Nacitam {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    print(f"  Nactenych radku: {len(df)}")

    df, boundary = build_features(df)

    tmp = df[['date', 'good_night'] + MODEL_FEATURES].dropna()
    tmp = tmp.sort_values('date').reset_index(drop=True)
    print(f"  Po sestaveni features: {len(tmp)} dni")
    print(f"  Hranice skore (median): {boundary:.2f}")

    split = int(len(tmp) * 0.8)
    val_size = max(60, int(split * 0.2))

    X_sub, y_sub = tmp[MODEL_FEATURES].iloc[:split - val_size], tmp['good_night'].iloc[:split - val_size]
    X_val, y_val = tmp[MODEL_FEATURES].iloc[split - val_size:split], tmp['good_night'].iloc[split - val_size:split]
    X_train, y_train = tmp[MODEL_FEATURES].iloc[:split], tmp['good_night'].iloc[:split]
    X_test, y_test = tmp[MODEL_FEATURES].iloc[split:], tmp['good_night'].iloc[split:]

    clf = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        LogisticRegression(max_iter=4000, class_weight='balanced', random_state=42)
    )

    clf.fit(X_sub, y_sub)
    threshold = tune_threshold(clf, X_val, y_val)

    clf.fit(X_train, y_train)

    probas = clf.predict_proba(X_test)[:, 1]
    preds = (probas >= threshold).astype(int)
    acc = accuracy_score(y_test, preds)
    bacc = balanced_accuracy_score(y_test, preds)
    baseline = float(max(y_test.mean(), 1 - y_test.mean()))

    print(f"\nVysledky na test setu ({len(X_test)} dni):")
    print(f"  Baseline (naivni tip): {baseline * 100:.1f}%")
    print(f"  Accuracy:              {acc * 100:.1f}%")
    print(f"  Balanced accuracy:     {bacc * 100:.1f}%")
    print(f"  Threshold:             {threshold:.3f}")
    print()
    print(classification_report(y_test, preds, target_names=['spatna noc', 'dobra noc']))

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({
            'model': clf,
            'threshold': threshold,
            'boundary': boundary,
            'features': MODEL_FEATURES,
            'accuracy': float(acc),
            'n_train': len(X_train),
            'n_test': len(X_test),
        }, f)

    print(f"Model ulozen: {MODEL_PATH}")


if __name__ == '__main__':
    main()
