# Minimalizace imputovanych HR zaznamu
# Martin Silar, SPSE Jecna C4c
# v4.3 - greedy odstraneni bloku kde hr_night_avg chybi
#        zachovavam >= MIN_DAYS zaznamu
#        imputace medianem jen pro zbyle izolované NaN

# [v teto verzi pridavam sekci Minimalizace do existujiciho kodu]

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
            if len(tmp.columns) >= 3: df = tmp; break
        except: continue
    if df is None: raise ValueError(f"Nepodarilo se nacist: {filename}")
    rename_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if 'start' in cl: rename_map[col] = 'ts_start'
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
    elif 'core' in v or 'unspecified' in v or v in ['1','2']: return 'core'
    elif v == 'asleep' or (v.endswith('asleep') and 'deep' not in v
                           and 'rem' not in v and 'core' not in v): return 'legacy'
    elif 'inbed' in v or v == '0': return 'inbed'
    elif 'awake' in v: return 'awake'
    else: return 'unknown'


def get_sleep_date(row):
    if row['hour'] < 12:
        return row['date'] - datetime.timedelta(days=1)
    return row['date']

def get_preferred_source(group):
    if 'source' not in group.columns: return group
    for s in group['source'].unique():
        if 'watch' in str(s).lower():
            return group[group['source'] == s]
    return group


def merge_intervals(intervals):
    if not intervals: return 0.0
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(e - s for s, e in merged) / 60.0


def agg_sleep(df_day):
    awakenings = int((df_day['stage'] == 'awake').sum())
    has_ts = 'ts_start' in df_day.columns and 'ts_end' in df_day.columns
    if has_ts:
        def get_iv(stage):
            sub = df_day[df_day['stage'] == stage][['ts_start', 'ts_end']].dropna()
            return [(r.ts_start, r.ts_end) for _, r in sub.iterrows()]
        deep_min = merge_intervals(get_iv('deep'))
        rem_min = merge_intervals(get_iv('rem'))
        awake_min = merge_intervals(get_iv('awake'))
        ib_min = merge_intervals(get_iv('inbed'))
        all_iv = get_iv('deep') + get_iv('rem') + get_iv('core')
        if not all_iv: all_iv = get_iv('legacy')
        total = merge_intervals(all_iv)
        core_min = max(0.0, total - deep_min - rem_min) if (deep_min+rem_min) > 0 else merge_intervals(get_iv('core') or get_iv('legacy'))
    else:
        deep_min = df_day[df_day['stage'] == 'deep']['duration_min'].sum()
        rem_min = df_day[df_day['stage'] == 'rem']['duration_min'].sum()
        core_min = df_day[df_day['stage'] == 'core']['duration_min'].sum()
        legacy = df_day[df_day['stage'] == 'legacy']['duration_min'].sum()
        awake_min = df_day[df_day['stage'] == 'awake']['duration_min'].sum()
        ib_min = df_day[df_day['stage'] == 'inbed']['duration_min'].sum()
        total = deep_min+rem_min+core_min if (deep_min+rem_min) > 0 else (legacy or core_min)
        core_min = max(0.0, total - deep_min - rem_min) if (deep_min+rem_min) > 0 else core_min
    time_in_bed = (total+awake_min) if (ib_min > total*3 and total > 0) else (total+awake_min+ib_min)
    return pd.Series({
        'sleep_total_min': total, 'sleep_deep_min': deep_min,
        'sleep_rem_min': rem_min, 'sleep_awakenings': awakenings,
    })


def compute_sleep_score(row):
    hours = row['sleep_total_min'] / 60.0
    hr = row['hr_night_avg']
    if row['data_era'] == 0:
        dur = max(0.0, 55.0 - abs(hours - 7.5) * 15.0)
        hrt = max(0.0, 45.0 - max(0.0, hr - 55.0) * 1.5)
        return round(min(100.0, max(0.0, dur + hrt)), 2)
    else:
        dur = max(0.0, 35.0 - abs(hours - 7.5) * 10.0)
        hrt = max(0.0, 20.0 - max(0.0, hr - 55.0) * 0.7)
        deep = min(20.0, (row['sleep_deep_min']/row['sleep_total_min'])*100) if row['sleep_total_min'] > 0 else 0
        rem = min(15.0, (row['sleep_rem_min']/row['sleep_total_min'])*75)  if row['sleep_total_min'] > 0 else 0
        wake = max(0.0, 10.0 - row['sleep_awakenings'] * 1.2)
        return round(min(100.0, max(0.0, dur+hrt+deep+rem+wake)), 2)


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
    if (g['is_night']).sum() >= 3 else np.nan})
).reset_index().sort_values('date').reset_index(drop=True)
frozen = (df_hr_daily['hr_night_avg'].eq(df_hr_daily['hr_night_avg'].shift(1)) &
          df_hr_daily['hr_night_avg'].eq(df_hr_daily['hr_night_avg'].shift(2)) &
          df_hr_daily['hr_night_avg'].eq(df_hr_daily['hr_night_avg'].shift(3)) &
          df_hr_daily['hr_night_avg'].eq(df_hr_daily['hr_night_avg'].shift(4)))
df_hr_daily.loc[frozen, 'hr_night_avg'] = np.nan

df_steps['value'] = pd.to_numeric(df_steps['value'], errors='coerce')
df_steps = df_steps.dropna(subset=['value'])
df_steps_daily = (
    df_steps.groupby('date', group_keys=False).apply(get_preferred_source)
    .groupby('date').agg(steps_total=('value', 'sum')).reset_index()
)
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
n_nan_steps = df['steps_total'].isna().sum()
if n_nan_steps > 0:
    df['steps_total'] = df['steps_total'].fillna(df['steps_total'].median())

# nova sekce: minimalizace imputovanych HR
# dny kde hr_night_avg == NaN = hodinky nebyly nasazeny tu noc
# chci jich mit co nejmene ale zachovat >= MIN_DAYS dni celkem
MIN_DAYS = 1500

df['_imp'] = df['hr_night_avg'].isna()
n_imp = df['_imp'].sum()
n_remove = max(0, min(n_imp, len(df) - MIN_DAYS))

print(f"\nMinimalizace imputovanych HR:")
print(f"  Celkem dni: {len(df)}, imputovanych: {n_imp}, mozu odstranit: {n_remove}")

if n_remove > 0:
    # najdu bloky po sobe jdoucich NaN dni
    df['_block'] = (df['_imp'] != df['_imp'].shift()).cumsum()
    blocks = []
    for bid, grp in df[df['_imp']].groupby('_block'):
        blocks.append({'block_id': bid, 'length': len(grp),
                       'start': grp['date'].min(), 'end': grp['date'].max()})
    blocks_df = pd.DataFrame(blocks).sort_values('length', ascending=False)

    # greedy: mazam od nejdelsich bloku
    to_remove = set()
    removed = 0
    for _, row in blocks_df.iterrows():
        if removed >= n_remove: break
        can = min(row['length'], n_remove - removed)
        dates = sorted(df[(df['_block'] == row['block_id']) & df['_imp']]['date'].tolist())
        start_cut = (len(dates) - can) // 2
        to_remove.update(dates[start_cut:start_cut+can])
        removed += can
        print(f"  Vyrazen: {can} dni z bloku {row['start'].date()}-{row['end'].date()}")

    df = df[~df['date'].isin(to_remove)].copy()
    df = df.drop(columns=['_block'])

# zbyle NaN doplnime medianem
_nan_mask = df['hr_night_avg'].isna()
N_HR_IMPUTED = int(_nan_mask.sum())
if N_HR_IMPUTED > 0:
    df['hr_night_avg'] = df['hr_night_avg'].fillna(df['hr_night_avg'].median())

df = df.drop(columns=['_imp'])
print(f"\n  Po minimalizaci: {len(df)} dni, zbyvajicich imputovanych: {N_HR_IMPUTED}")

df['sleep_score'] = df.apply(compute_sleep_score, axis=1)

FINAL_COLS = ['date', 'sleep_score', 'sleep_total_min', 'hr_night_avg',
              'steps_total', 'active_kcal', 'sleep_deep_min', 'sleep_rem_min',
              'sleep_awakenings', 'data_era', 'day_of_week', 'month']
df[FINAL_COLS].to_csv('health_daily_v4_3.csv', index=False)
print(f"Ulozeno: health_daily_v4_3.csv")
