# Vypocet sleep score
# Martin Silar, SPSE Jecna C4c
# v4 - pridany vzorec pro sleep score (target promenna modelu)
#      korelacni matice, export datasetu

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import warnings
warnings.filterwarnings('ignore')

# [funkce load_hk_csv, parse_timestamps, classify_sleep_stage,
#  get_sleep_date, agg_sleep jsou stejne jako v v3]


def load_hk_csv(filename):
    df = None
    for sep in [',', ';', '\t']:
        try:
            tmp = pd.read_csv(filename, sep=sep, low_memory=False)
            if len(tmp.columns) >= 3: df = tmp; break
        except: continue
    if df is None: raise ValueError(f"Nepodarilo se nacist: {filename}")
    rename_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if 'start' in cl:   rename_map[col] = 'ts_start'
        elif 'end' in cl and 'ts_end' not in rename_map.values():
            rename_map[col] = 'ts_end'
        elif 'duration' in cl: rename_map[col] = 'duration_sec'
        elif cl == 'value':  rename_map[col] = 'value'
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
    elif 'core' in v or 'unspecified' in v or v in ['1', '2']: return 'core'
    elif v == 'asleep' or (v.endswith('asleep') and 'deep' not in v
                           and 'rem' not in v and 'core' not in v): return 'legacy'
    elif 'inbed' in v or v == '0': return 'inbed'
    elif 'awake' in v: return 'awake'
    else: return 'unknown'


def get_sleep_date(row):
    if row['hour'] < 12:
        return row['date'] - datetime.timedelta(days=1)
    return row['date']


def agg_sleep(df_day):
    deep = df_day[df_day['stage'] == 'deep']['duration_min'].sum()
    rem = df_day[df_day['stage'] == 'rem']['duration_min'].sum()
    core = df_day[df_day['stage'] == 'core']['duration_min'].sum()
    total = deep + rem + core
    awake = df_day[df_day['stage'] == 'awake']['duration_min'].sum()
    ib = df_day[df_day['stage'] == 'inbed']['duration_min'].sum()
    awakenings = int((df_day['stage'] == 'awake').sum())
    time_in_bed = (total+awake) if (ib > total*3 and total > 0) else (total+awake+ib)
    return pd.Series({
        'sleep_total_min': total, 'sleep_deep_min': deep,
        'sleep_rem_min': rem, 'sleep_awakenings': awakenings,
    })

# --- NACITANI A AGREGACE (stejne jako v3) ---


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
    .apply(agg_sleep).reset_index()
    .rename(columns={'sleep_date': 'date'})
)

df_hr['value'] = pd.to_numeric(df_hr['value'], errors='coerce')
df_hr = df_hr[df_hr['value'].notna() & (df_hr['value'] <= 220)]
df_hr['is_night'] = df_hr['hour'].apply(lambda h: h >= 22 or h < 6)
df_hr_daily = df_hr.groupby('date').apply(
    lambda g: pd.Series({'hr_night_avg': g[g['is_night']]['value'].mean()
    if (g['is_night']).sum() >= 3 else np.nan})).reset_index().sort_values('date').reset_index(drop=True)
frozen = (df_hr_daily['hr_night_avg'].eq(df_hr_daily['hr_night_avg'].shift(1)) &
          df_hr_daily['hr_night_avg'].eq(df_hr_daily['hr_night_avg'].shift(2)) &
          df_hr_daily['hr_night_avg'].eq(df_hr_daily['hr_night_avg'].shift(3)) &
          df_hr_daily['hr_night_avg'].eq(df_hr_daily['hr_night_avg'].shift(4)))
df_hr_daily.loc[frozen, 'hr_night_avg'] = np.nan

df_steps['value'] = pd.to_numeric(df_steps['value'], errors='coerce')
df_steps_daily = df_steps.dropna(subset=['value']).groupby('date').agg(
    steps_total=('value', 'sum')).reset_index()
df_act['value'] = pd.to_numeric(df_act['value'], errors='coerce')
df_act_daily = df_act.dropna(subset=['value']).groupby('date').agg(
    active_kcal=('value', 'sum')).reset_index()

df = df_sleep_daily.copy()
df['date'] = pd.to_datetime(df['date'])
for other in [df_hr_daily, df_steps_daily, df_act_daily]:
    other = other.copy(); other['date'] = pd.to_datetime(other['date'])
    df = df.merge(other, on='date', how='left')

df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['data_era'] = (df['date'] >= pd.Timestamp('2022-12-01')).astype(int)
df.loc[df['data_era'] == 0, 'sleep_awakenings'] = 0
df = df[(df['sleep_total_min'] >= 180) & (df['sleep_total_min'] <= 840)]
df = df[df['active_kcal'] > 0]
for col in ['hr_night_avg', 'steps_total']:
    df[col] = df[col].fillna(df[col].median())

# --- SLEEP SCORE ---
# Kompozitni skore 0-100 podle vedeckych zdroju:
#   Walker (2017), NSF (2015), Oura Ring methodology
#
# Era 0: jen delka spanku + nocni HR (faze nedostupne)
# Era 1: delka + HR + deep/REM/probuzeni
#
# Vahy:            Era 0   Era 1
#   delka spanku:    55      35
#   nocni HR:        45      20
#   deep sleep:       0      20
#   REM sleep:        0      15
#   probuzeni:        0      10
#   MAX:            100     100


def compute_sleep_score(row):
    hours = row['sleep_total_min'] / 60.0
    hr    = row['hr_night_avg']
    if row['data_era'] == 0:
        dur = max(0.0, 55.0 - abs(hours - 7.5) * 15.0)
        hrt = max(0.0, 45.0 - max(0.0, hr - 55.0) * 1.5)
        return round(min(100.0, max(0.0, dur + hrt)), 2)
    else:
        dur = max(0.0, 35.0 - abs(hours - 7.5) * 10.0)
        hrt = max(0.0, 20.0 - max(0.0, hr - 55.0) * 0.7)
        deep = min(20.0, (row['sleep_deep_min']/row['sleep_total_min'])*100) if row['sleep_total_min'] > 0 else 0
        rem = min(15.0, (row['sleep_rem_min']/row['sleep_total_min'])*75) if row['sleep_total_min'] > 0 else 0
        wake = max(0.0, 10.0 - row['sleep_awakenings'] * 1.2)
        return round(min(100.0, max(0.0, dur+hrt+deep+rem+wake)), 2)


df['sleep_score'] = df.apply(compute_sleep_score, axis=1)
print("Sleep score po erach:")
print(df.groupby('data_era')['sleep_score'].describe().round(2))

# korelacni matice - viz jestli features dava smysl
feat_cols = ['sleep_score', 'sleep_total_min', 'hr_night_avg',
             'steps_total', 'active_kcal', 'sleep_deep_min',
             'sleep_rem_min', 'sleep_awakenings', 'data_era']
corr = df[feat_cols].corr()

fig, ax = plt.subplots(figsize=(9,7))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(feat_cols)))
ax.set_yticks(range(len(feat_cols)))
ax.set_xticklabels(feat_cols, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(feat_cols, fontsize=8)
for i in range(len(feat_cols)):
    for j in range(len(feat_cols)):
        ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center',
                fontsize=6, color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')
plt.colorbar(im, ax=ax)
ax.set_title('Korelacni matice')
plt.tight_layout()
plt.savefig('v4_correlation.png', dpi=120)
plt.show()

FINAL_COLS = ['date', 'sleep_score', 'sleep_total_min', 'hr_night_avg',
              'steps_total', 'active_kcal', 'sleep_deep_min', 'sleep_rem_min',
              'sleep_awakenings', 'data_era', 'day_of_week', 'month']
df[FINAL_COLS].to_csv('health_daily_v4.csv', index=False)
print(f"Ulozeno: {len(df)} dni -> health_daily_v4.csv")