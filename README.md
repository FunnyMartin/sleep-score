# Sleep Score Predictor

**Autor:** Martin Šilar  
**Škola:** SPŠE Ječná, Praha, C4c  
**Rok:** 2026

---

## Co to dělá

Projekt predikuje jestli bude dnešní noc dobrá nebo špatná na základě 7 let osobních biometrických dat z Apple Health (2017–2026). Pipeline zahrnuje čistění dat, výpočet sleep score a klasifikační model, který dostane historii noční HR za posledních 7 nocí a dnešní aktivitu a vrátí predikci.

---

## Jak spustit

```bash
# 1. závislosti
pip install -r requirements.txt

# 2. předzpracování dat
python transformation.py
# health_daily_v5.csv

# 3. export modelu
python export_model.py
# models/sleep_model.pkl

# 4. webová aplikace
python app_web.py
# http://localhost:5000

# 5. testy
python -m unittest tests/test_basic.py -v
```

---

## Dataset

Data pochází z osobního exportu Apple Health (iPhone + Apple Watch).

| Soubor | Řádků |
|---|---|
| HeartRate.csv | ~1 504 194 |
| ActiveEnergyBurned.csv | ~615 878 |
| StepCount.csv | ~380 951 |
| SleepAnalysis.csv | ~132 131 |

Po čistění: **1636 dní**, 0 chybějících hodnot.

**Dvě éry dat:**
- Éra 0 (před 1. 12. 2022): bez fází spánku, jen délka + HR
- Éra 1 (od 1. 12. 2022): watchOS 9+, dostupné deep/REM/core fáze

**Klíčová oprava v transformation.py v5.1:**  
Noční HR měřená po půlnoci (00:00–06:00) patří k předchozí noci, ne k aktuálnímu kalendářnímu dni. Předchozí verze toto ignorovala -> v5.1 opravuje přiřazení přes `sleep_date` příznak.

---

## Sleep score vzorec

```
Éra 0 (max 100 bodů):
  délka:  max(0, 55 - |hodiny - 7.5| * 15)    → optimum 7.5h
  HR:     max(0, 45 - max(0, HR - 55) * 1.5)  → optimum ≤ 55 BPM

Éra 1 (max 100 bodů):
  délka:  max(0, 35 - |hodiny - 7.5| * 10)
  HR:     max(0, 20 - max(0, HR - 55) * 0.7)
  deep:   min(20, deep/total * 100)
  REM:    min(15, rem/total * 75)
  awake:  max(0,  10 - probuzení * 1.2)
```

Zdroje: Walker (2017), NSF (Hirshkowitz 2015), Oura Ring metodologie.

---

## Model

### Co predikuje

Binární klasifikace: bude dnešní spánek **nad nebo pod osobním mediánem** (75.6 bodů)?

### Proč Logistic Regression

- Koeficienty přímo interpretovatelné -> ví se proč model rozhodl jak rozhodl
- Na malém datasetu (1636 dní) se přefituje méně než Gradient Boosting
- Pipeline s `StandardScaler` zajišťuje správné měřítko HR hodnot
- `class_weight='balanced'` kompenzuje nevyvážené třídy v trénovacích datech

### Features modelu (14)

| Feature | Popis |
|---|---|
| `prev1_hr` až `prev7_hr` | Noční HR za posledních 7 nocí (BPM) |
| `rolling7_hr` | Klouzavý průměr HR za 7 nocí |
| `rolling3_hr` | Klouzavý průměr HR za 3 noci |
| `dow_sin`, `dow_cos` | Cyklické kódování dne v týdnu |
| `is_weekend` | 1 = sobota nebo neděle |
| `steps_today` | Kroky za dnešní den |
| `kcal_today` | Aktivní kalorie za dnešní den |

### Cyklické kódování dne v týdnu

Den v týdnu je cyklická proměnná -> neděle (6) je blíže pondělí (0) než středě (3). Přímé číslo by model nespojil. Sin a cos transformace tohle zachovají:

```python
dow_sin = sin(2π * day / 7)
dow_cos = cos(2π * day / 7)
```

### Threshold tuning

Standardní práh 0.5 nemusí být optimální. Model hledá nejlepší práh na validační sadě (odděleně od test setu) v rozsahu 0.35–0.65, vybírá ten s nejvyšší accuracy.

### Výsledky

| Metrika | Hodnota              |
|---|----------------------|
| Accuracy | ~66%                 |
| Baseline (naivní tip) | 50%                  |
| Zlepšení | +16 procentních bodů |

### Dataset
| Metoda                     | Hodnota |
|---------------------------|--------:|
| Načtených řádků           | 1636    |
| Po sestavení features     | 1629 dní|
| Hranice skóre (median)    | 75.56   |

### Výsledky (test set)
| Metrika                  | Hodnota |
|--------------------------|--------:|
| Baseline (naivní tip)    | 65.6 %  |
| Accuracy                 | 66.6 %  |
| Balanced accuracy        | 53.7 %  |
| Threshold                | 0.370   |

### Classification report
| Třída        | Precision | Recall | F1-score | Support |
|--------------|----------:|-------:|---------:|--------:|
| špatná noc   | 0.56      | 0.12   | 0.20     | 112     |
| dobrá noc    | 0.67      | 0.95   | 0.79     | 214     |

### Souhrn
| Metrika        | Precision | Recall | F1-score | Support |
|----------------|----------:|-------:|---------:|--------:|
| accuracy       |           |        | 0.67     | 326     |
| macro avg      | 0.62      | 0.54   | 0.50     | 326     |
| weighted avg   | 0.64      | 0.67   | 0.59     | 326     |

### Tradeoffs a limitace

**Éra drift:** Novější data (éra 1) mají systematicky vyšší sleep score protože watch měří víc. Model se částečně naučil "novější = lepší", ne jen skutečné HR vzory. Skutečný prediktivní signál je slabší než accuracy naznačuje.

**Kroky a kalorie přidávají marginální hodnotu.** Korelace s cílovým labelem je jen 0.08 a -0.06. Jsou ve formuláři pro úplnost vstupu, ale accuracy skoro neovlivňují.

**Pouze jeden uživatel.** Model by nefungoval pro jinou osobu bez přetrénování na jejích datech.

**Špatné noci jsou těžší predikovat.** Recall pro špatné noci je ~40 % -> přicházejí z důvodů co hodinky nezměří (stres, nemoc, hluk).

---

## Konfigurace

| Proměnná | Výchozí | Soubor |
|---|---|---|
| `MODEL_PATH` | `models/sleep_model.pkl` | `app_web.py` |
| `MIN_DAYS` | `1500` | `transformation.py` |
| `N_LAGS` | `7` | `export_model.py` |

---

## Chybové stavy

| Chyba | Příčina | Řešení |
|---|---|---|
| `Model neni nacten` | Chybí `sleep_model.pkl` | Spusť `export_model.py` |
| `health_daily_v5.csv nenalezen` | Chybí výstup transformation.py | Spusť `transformation.py` |
| `HR mimo rozsah` | Hodnota mimo 30–130 BPM | Zadej platnou hodnotu |
| `Port 5000 obsazen` | Jiný proces na portu | `lsof -i :5000` a ukonči ho |
