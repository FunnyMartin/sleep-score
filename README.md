**Autor:** Martin Šilar  
**Škola:** SPŠE Ječná, Praha  
**Ročník:** C4c  
**Předmět:** PV - školní softwarový projekt (strojové učení)  
**Kontakt:** silar@spsejecna.cz  
**Datum vypracování:** Březen 2026

---

## 1. Popis projektu

**Název projektu:** Sleep Score Predictor

Projekt predikuje kvalitu spánku jako číslo 0-100 (*sleep score*) na základě biometrických vstupů. Model strojového učení bude natrénovaný na 7 letech osobních dat z Apple Health (2017-2026) - celkem 1636 denních záznamů.

Komerční aplikace jako Oura Ring nebo Whoop nenabízejí podobnou funkcionalitu. Tento projekt ukazuje, že predikci lze postavit zdarma na vlastních datech.

**Aktuální stav projektu:**

| Část | Stav |
|---|---|
| `data/data.zip` - CSV soubory z Apple Health exportu | hotovo |
| `src/data.py` - načítání a agregace dat | hotovo |
| `src/transformation.py` - čistění, výpočet sleep score, vizualizace | hotovo |
| `docs/overview.png`, `docs/correlation.png` - grafy datasetu | hotovo |
| NB02 - trénování modelu | plánováno |
| Lokální webová aplikace (Flask, localhost) | plánováno |

---

## 2. Analýza požadavků

### 2.1 Funkční požadavky

| ID | Požadavek | Stav |
|---|---|---|
| FR-01 | Načtení CSV souborů z Apple Health exportu | hotovo |
| FR-02 | Agregace surových dat na denní záznamy | hotovo |
| FR-03 | Výpočet sleep score (0-100) podle definovaného vzorce | hotovo |
| FR-04 | Vizualizace datasetu (histogramy, korelační matice) | hotovo |
| FR-05 | Trénování regresního modelu, srovnání algoritmů | plánováno |
| FR-06 | Webová aplikace pro zadání parametrů a zobrazení predikce | plánováno |

### 2.2 Nefunkční požadavky

| ID | Požadavek |
|---|---|
| NFR-01 | Skripty jsou spustitelné bez IDE (`python src/transformation.py`). |
| NFR-02 | Dataset obsahuje min. 1500 záznamů a min. 5 atributů (splněno: 1636 dní, 10 atributů). |
| NFR-03 | Data jsou reálná a sesbíraná autorem - osobní export z Apple Health. |
| NFR-04 | Zdrojový kód je verzovaný pomocí Git (10 postupných commitů). |

### 2.3 Use Case

**Aktér:** Uživatel se zájmem o sledování kvality spánku

**UC1 - Přehled datasetu (hotovo):**
1. Uživatel spustí `src/transformation.py`.
2. Skript zpracuje data z `data/data.zip`.
3. Vygeneruje `health_daily_v5.csv` a uloží grafy do `docs/`.

**UC2 - Predikce skóre (plánováno):**
1. Uživatel otevře `http://localhost:5000`.
2. Vyplní parametry spánku.
3. Aplikace vrátí skóre a doporučení.

---

## 3. Architektura projektu

### 3.1 Struktura souborů

```
sleep-score/
|
+-- data/
|   +-- data.zip            <- CSV soubory z Apple Health
|
+-- docs/
|   +-- overview.png        <- 9 histogramů datasetu
|   +-- correlation.png     <- korelační matice features
|
+-- src/
|   +-- data.py             <- načítání CSV, parsování timestampů, agregace
|   +-- transformation.py   <- čistění dat, sleep score vzorec, vizualizace
|
+-- docs/
    +-- README.md
```

### 3.2 Tok dat

```
data/data.zip
(CSV soubory z Apple Health)
      |
      v
src/data.py
- načte CSV soubory
- parsuje Unix timestamps na datetime (časová zóna Praha)
- klasifikuje fáze spánku (deep/REM/core/legacy)
- agreguje na denní záznamy
      |
      v
src/transformation.py
- čistění a filtrování outlierů
- oprava double countingu (merge_intervals)
- minimalizace chybějících HR hodnot
- výpočet sleep score
- vizualizace (overview.png, correlation.png)
- export: health_daily_v5.csv
      |
      v
health_daily_v5.csv
(1636 dní, 10 atributů, 0 NaN)
```

---

## 4. Data a předzpracování

### 4.1 Zdroj dat

Data pochází z osobního exportu Apple Health autora projektu (iPhone + Apple Watch, 2017-2026). Export byl proveden přes *Nastavení -> Zdraví -> Exportovat veškerá zdravotní data*. Protože soubor `export.xml` má stovky MB, byla data předem rozbalena do CSV souborů a uložena jako `data/data.zip`.

Obsah archivu:

| Soubor | Obsah | Počet řádků |
|---|---|---|
| `HKQuantityTypeIdentifierHeartRate.csv` | Tepová frekvence | ~1 504 194 |
| `HKQuantityTypeIdentifierActiveEnergyBurned.csv` | Aktivní kalorie | ~615 878 |
| `HKQuantityTypeIdentifierStepCount.csv` | Kroky | ~380 951 |
| `HKCategoryTypeIdentifierSleepAnalysis.csv` | Spánek a fáze | ~132 131 |

Každý soubor má sloupce: `timestamp_start`, `timestamp_end`, `duration_sec`, `value`, `unit`, `source`, `device`.

### 4.2 Výsledný dataset - atributy

| Atribut | Popis | Poznámka |
|---|---|---|
| `sleep_total_min` | Celková délka spánku (minuty) | hlavní feature |
| `hr_night_avg` | Průměrná HR v noci 22:00-06:00 (BPM) | |
| `steps_total` | Kroky za den | Watch má přednost před iPhonem |
| `active_kcal` | Aktivní kalorie za den | |
| `sleep_deep_min` | Deep spánek (minuty) | 0 bez watchOS 9 |
| `sleep_rem_min` | REM spánek (minuty) | 0 bez watchOS 9 |
| `sleep_awakenings` | Počet probuzení | 0 bez watchOS 9 |
| `data_era` | 0 = bez fází, 1 = s fázemi | příznak zařízení |
| `day_of_week` | Den v týdnu (0 = pondělí) | |
| `month` | Měsíc (1-12) | |
| `sleep_score` | **TARGET** - kvalita spánku 0-100 | vypočteno vzorcem |

**Celkem: 1636 dní, 0 chybějících hodnot**

### 4.3 Klíčové kroky předzpracování

**Oprava double countingu (Apple Health 2023):**  
Apple Health od konce roku 2022 exportuje spánkové záznamy ve dvou formátech najednou - starý (`asleep`, celá noc jako jeden blok) i nový (`asleepCore/Deep/REM`, skutečné fáze). Prostý součet by vrátil 12-14 hodin místo reálných 7-8 hodin. Funkce `merge_intervals()` seřadí časové úseky a překryvy sloučí.

```
Příklad:
[23:00-02:00] + [01:00-07:00] = [23:00-07:00] = 8h, ne 11h
```

**Data era:**
- Éra 0 (před 1. 12. 2022): fáze spánku nedostupné (starší zařízení)
- Éra 1 (od 1. 12. 2022): watchOS 9+, fáze deep/REM/core dostupné

**Minimalizace chybějícího HR:**  
V letech 2017-2018 hodinky nebyly nasazeny - noční HR chybí. Bloky s chybějícím HR jsou postupně odstraňovány (od nejdelšího) dokud neklesáme pod 1500 záznamů. Výsledek: odstraněno 270 dní, zbývá 1636 záznamů bez imputace.

### 4.4 Vzorec sleep score

```
Éra 0 - max 100 bodů (délka + HR):
  délka:  max(0, 55 - |hours - 7.5| * 15)    optimum 7.5h
  HR:     max(0, 45 - max(0, HR - 55) * 1.5) optimum <= 55 BPM

Éra 1 - max 100 bodů (délka + HR + fáze):
  délka:  max(0, 35 - |hours - 7.5| * 10)
  HR:     max(0, 20 - max(0, HR - 55) * 0.7)
  deep:   min(20, deep/total * 100)            optimum >= 20%
  REM:    min(15, rem/total * 75)              optimum >= 20%
  awake:  max(0, 10 - awakenings * 1.2)        optimum 0-2x
```

Zdroje: Walker (2017) *Why We Sleep*, Hirshkowitz et al. (2015) NSF, Oura Ring metodologie (2023).

### 4.5 Vizualizace dat

**overview.png - 9 histogramů:**
- *Sleep Score* - distribuce skóre, patrné dvě oblasti odpovídající érám
- *Sleep Score po érách* - éra 0 (modrá) vs éra 1 (červená)
- *Sleep Score v čase* - 2017-2026, patrné zlepšení po přechodu na watchOS 9
- *Délka spánku* - histogram s optimem 7.5h
- *Noční HR* - červené = reálná data, šedé = imputované hodnoty (pokud existují)
- *Deep Sleep / REM Sleep* - pouze data éry 1, distribuce kolem 80-100 min

**correlation.png - korelační matice:**

| Dvojice | Korelace | Interpretace |
|---|---|---|
| sleep_score vs sleep_total_min | -0.56 | Delší spánek != lepší (optimum 7.5h) |
| sleep_score vs hr_night_avg | -0.68 | Nižší klidová HR = lepší regenerace |
| sleep_score vs sleep_rem_min | +0.52 | Více REM = lepší skóre |
| sleep_score vs data_era | +0.60 | Éra 1 má více informací |

### 4.6 Zdokumentované limitace

| Limitace | Popis |
|---|---|
| Chybějící rok 2021 | 445 dní bez dat - výměna zařízení |
| `sleep_awakenings = 0` v éře 0 | Starší zařízení probuzení neměřila |
| Mezera 2020 | ~246 dní bez hodinek |

---

## 5. Plánovaný ML model

*(Tato část bude doplněna po dokončení NB02)*

Plánuje se srovnání těchto algoritmů na chronologickém splitu 80/20:
- Ridge Regression (baseline)
- Random Forest
- Gradient Boosting Regressor

Metriky: MAE, RMSE, R². Výsledný model bude exportován jako `sleep_model.pkl`.

---

## 6. Instalace a spuštění

### Požadavky

- Python 3.10+
- pip

### Instalace závislostí

```bash
pip install pandas numpy matplotlib scikit-learn
```

### Spuštění

```bash
python src/transformation.py
```

Skript načte data z `data/data.zip`, zpracuje je a uloží `health_daily_v5.csv` a grafy do `docs/`.

---

## 7. Konfigurace

| Proměnná | Výchozí | Soubor | Popis |
|---|---|---|---|
| `MIN_DAYS` | `1500` | `transformation.py` | Minimální počet dní po čistění |
| `STAGES_ERA_START` | `2022-12-01` | `transformation.py` | Datum přechodu na éru 1 |

---

## 8. Chybové stavy

| Chyba | Příčina | Řešení |
|---|---|---|
| `FileNotFoundError: data.zip` | Chybí archiv se daty | Zkontrolujte složku `data/` |
| `ValueError: Nepodarilo se nacist: ...` | Nepodporovaný formát CSV | Zkontrolujte oddělovač |
| `KeyError: ts_start` | Neočekávané názvy sloupců | Zkontrolujte verzi exportu |

---

## 9. Knihovny třetích stran

| Knihovna | Účel | Licence |
|---|---|---|
| `pandas` | Načtení a zpracování CSV | BSD |
| `numpy` | Numerické výpočty | BSD |
| `matplotlib` | Vizualizace | PSF |
| `scikit-learn` | ML modely (plánováno) | BSD |

---

## 10. Licence

Zdrojový kód je autorské dílo Martina Šilara vytvořené jako školní práce na SPŠE Ječná. Data pochází z osobního Apple Health exportu autora. Projekt není určen pro komerční použití.

---

## 11. Verze a známé problémy

### Historie verzí

| Verze | Popis |
|---|---|
| v1 | Načtení CSV souborů, základní statistiky |
| v2 | Agregace na denní záznamy |
| v3 | Čistění dat, filtrování outlierů, příznak data_era |
| v4 | Vzorec sleep_score |
| v4.1 | Deduplikace Watch vs iPhone pro kroky |
| v4.2 | Oprava double countingu Apple Health 2023 (`merge_intervals`) |
| v4.3 | Greedy odstranění bloků s chybějícím HR |
| v4.4 | HR stacked bar vizualizace, korelační matice |
| v5 | Orientační evaluace datasetu, finální export |

### Známé problémy

| ID | Popis | Závažnost |
|---|---|---|
| B01 | `sleep_awakenings = 0` pro éru 0 - starší zařízení data nesbírala | Nízká |
| B02 | Rok 2021 chybí v datasetu (445 dní) - výměna zařízení | Nízká |
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)