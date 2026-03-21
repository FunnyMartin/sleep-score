# Nacitani dat z Apple Health
# Martin Silar, SPSE Jecna C4c
# v1 - zakladni nacteni a zobrazeni dat

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import warnings
warnings.filterwarnings('ignore')

# nacteni CSV - Apple Health exportuje s ruznymi oddelovaci


def load_hk_csv(filename):
    df = None
    for sep in [',', ';', '\t']:
        try:
            tmp = pd.read_csv(filename, sep=sep, low_memory=False)
            if len(tmp.columns) >= 3:
                df = tmp
                break
        except:
            continue
    if df is None:
        raise ValueError(f"Nepodarilo se nacist: {filename}")
    print(f"  {filename}: {len(df)} radku")
    rename_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if 'start' in cl:   rename_map[col] = 'ts_start'
        elif 'end' in cl and 'ts_end' not in rename_map.values():
            rename_map[col] = 'ts_end'
        elif 'duration' in cl: rename_map[col] = 'duration_sec'
        elif cl == 'value': rename_map[col] = 'value'
        elif 'source' in cl: rename_map[col] = 'source'
    return df.rename(columns=rename_map)


def parse_timestamps(df):
    df = df.copy()
    df['ts_start'] = pd.to_numeric(df['ts_start'], errors='coerce')
    if 'ts_end' in df.columns:
        df['ts_end'] = pd.to_numeric(df['ts_end'], errors='coerce')
    df['datetime'] = pd.to_datetime(df['ts_start'], unit='s', utc=True)
    df['datetime'] = df['datetime'].dt.tz_convert('Europe/Prague')
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    return df.dropna(subset=['datetime'])


print("Nacitam soubory...")
df_hr = load_hk_csv('HKQuantityTypeIdentifierHeartRate.csv')
df_act = load_hk_csv('HKQuantityTypeIdentifierActiveEnergyBurned.csv')
df_steps = load_hk_csv('HKQuantityTypeIdentifierStepCount.csv')
df_sleep = load_hk_csv('HKCategoryTypeIdentifierSleepAnalysis.csv')

df_hr = parse_timestamps(df_hr)
df_act = parse_timestamps(df_act)
df_steps = parse_timestamps(df_steps)
df_sleep = parse_timestamps(df_sleep)

print(f"\nHR: {df_hr['date'].min()} az {df_hr['date'].max()}")
print(f"Sleep: {df_sleep['date'].min()} az {df_sleep['date'].max()}")

# zakladni distribuce hodnot
df_hr['value'] = pd.to_numeric(df_hr['value'], errors='coerce')
print(f"\nHR hodnoty: min={df_hr['value'].min():.0f} max={df_hr['value'].max():.0f} mean={df_hr['value'].mean():.1f}")

df_sleep['value'].value_counts().head(10).plot(kind='bar')
plt.title('Distribuce typu spanku')
plt.tight_layout()
plt.savefig('v1_sleep_types.png', dpi=120)
plt.show()
print("Graf ulozen: v1_sleep_types.png")
