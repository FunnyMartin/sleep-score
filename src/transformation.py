# Agregace dat na den
# Martin Silar, SPSE Jecna C4c
# v2 - pridana agregace sleep, HR a steps na denni zaznam

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import warnings
warnings.filterwarnings('ignore')


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
    rename_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if 'start' in cl: rename_map[col] = 'ts_start'
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


def classify_sleep_stage(val):
    if pd.isna(val): return 'unknown'
    v = str(val).strip().lower()
    if 'deep' in v or v == '3': return 'deep'
    elif 'rem' in v or v == '4': return 'rem'
    elif 'core' in v or 'unspecified' in v or v in ['1','2']:
        return 'core'
    elif v == 'asleep' or (v.endswith('asleep') and 'deep' not in v and 'rem' not in v and 'core' not in v):
        return 'legacy'
    elif 'inbed' in v or v == '0': return 'inbed'
    elif 'awake' in v: return 'awake'
    else: return 'unknown'


def get_sleep_date(row):
    if row['hour'] < 12:
        return row['date'] - datetime.timedelta(days=1)
    return row['date']


# jednoducha agregace bez deduplikace
def agg_sleep_simple(df_day):
    deep = df_day[df_day['stage'] == 'deep']['duration_min'].sum()
    rem = df_day[df_day['stage'] == 'rem']['duration_min'].sum()
    core = df_day[df_day['stage'] == 'core']['duration_min'].sum()
    total = deep + rem + core
    awakenings = int((df_day['stage'] == 'awake').sum())
    return pd.Series({
        'sleep_total_min': total,
        'sleep_deep_min':  deep,
        'sleep_rem_min':   rem,
        'sleep_awakenings': awakenings,
    })


print("Nacitam...")
df_hr = parse_timestamps(load_hk_csv('HKQuantityTypeIdentifierHeartRate.csv'))
df_act = parse_timestamps(load_hk_csv('HKQuantityTypeIdentifierActiveEnergyBurned.csv'))
df_steps = parse_timestamps(load_hk_csv('HKQuantityTypeIdentifierStepCount.csv'))
df_sleep = parse_timestamps(load_hk_csv('HKCategoryTypeIdentifierSleepAnalysis.csv'))

df_sleep['stage'] = df_sleep['value'].apply(classify_sleep_stage)
df_sleep['duration_min'] = df_sleep['duration_sec'] / 60.0
df_sleep['sleep_date'] = df_sleep.apply(get_sleep_date, axis=1)

df_sleep_daily = (
    df_sleep.groupby('sleep_date', group_keys=False)
    .apply(agg_sleep_simple).reset_index()
    .rename(columns={'sleep_date':'date'})
)

df_hr['value'] = pd.to_numeric(df_hr['value'], errors='coerce')
df_hr = df_hr.dropna(subset=['value'])
df_hr['is_night'] = df_hr['hour'].apply(lambda h: h >= 22 or h < 6)
df_hr_daily = df_hr.groupby('date').apply(
    lambda g: pd.Series({'hr_night_avg': g[g['is_night']]['value'].mean()
    if (g['is_night']).sum() >= 3 else np.nan})).reset_index()


df_steps['value'] = pd.to_numeric(df_steps['value'], errors='coerce')
df_steps_daily = df_steps.groupby('date').agg(steps_total=('value','sum')).reset_index()

df_act['value'] = pd.to_numeric(df_act['value'], errors='coerce')
df_act_daily = df_act.groupby('date').agg(active_kcal=('value','sum')).reset_index()

# merge
df = df_sleep_daily.copy()
df['date'] = pd.to_datetime(df['date'])
for other, name in [(df_hr_daily,'HR'),(df_steps_daily,'steps'),(df_act_daily,'kcal')]:
    other = other.copy(); other['date'] = pd.to_datetime(other['date'])
    df = df.merge(other, on='date', how='left')

print(f"Mergnutych dni: {len(df)}")
print(df[['sleep_total_min','hr_night_avg','steps_total','active_kcal']].describe().round(1))

# vizualizace
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].hist(df['sleep_total_min'].dropna()/60, bins=40, color='navy', edgecolor='white')
axes[0].set_title('Delka spanku (h)')
axes[1].hist(df['hr_night_avg'].dropna(), bins=40, color='crimson', edgecolor='white')
axes[1].set_title('Nocni HR (BPM)')
axes[2].hist(df['steps_total'].dropna(), bins=40, color='green', edgecolor='white')
axes[2].set_title('Kroky za den')
plt.tight_layout()
plt.savefig('v2_daily_overview.png', dpi=120)
plt.show()
print("Graf ulozen: v2_daily_overview.png")