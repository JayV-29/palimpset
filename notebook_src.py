# %% [markdown]
# # PALIMPSEST — Identified Counterfactual Explanations for Load Forecasting
#
# **One notebook, every experiment.** This notebook reproduces the full PALIMPSEST pipeline:
#
# | § | Stage | What it does |
# |---|---|---|
# | 1 | Setup & config | Paths, Kaggle detection, internet detection, experiment constants |
# | 2 | Data ingestion | EIA-930 hourly demand (CAISO), NOAA ISD hourly weather, CAISO Flex Alert dates, Low Carbon London (Kaggle mirror) |
# | 3 | Synthetic fallback | Same schemas with a **known** thermal/behavioral ground truth — used automatically when downloads are unavailable |
# | 4 | Thermal layer | Cooling/heating degree-hours with fitted thermal-mass lag and activation temperatures (nonlinear least squares) |
# | 5 | Behavioral layer | Appeal (Flex Alert) and occupancy (COVID lockdown) coefficients **identified from matched natural-experiment contrasts only** |
# | 6 | Residual forecaster | LightGBM quantile models on calendar, lagged load, weather and layer 1–2 outputs |
# | 7 | Counterfactual engine | Abduction–action–prediction; kNN plausibility projection; decomposed `thermal + behavioral + residual` delta; analog spread |
# | 8 | Eval A: accuracy | MAPE / pinball / coverage on 2023–2024 vs plain LightGBM, persistence, similar-day |
# | 9 | Eval B: Flex Alert holdout | **Central figure** — identified vs. full-series ablation vs. flag-as-feature on held-out alert days |
# | 10 | Eval C: COVID holdout | "Same weather, 2020 occupancy" vs observed |
# | 11 | Eval D: Low Carbon London | Randomized ToU contrast transferred across household splits |
# | 12 | Analog agreement | kNN historical-day spread as calibration diagnostic |
# | 13 | Interactive view | Pick a day, pose a weather/appeal counterfactual, see decomposition + analogs |
#
# **Modes.** With internet enabled the notebook downloads free public data (EIA-930 six-month files, NOAA
# global-hourly ISD CSVs). Attach the Kaggle dataset *"Smart meters in London"* (`jeanmidev/smart-meters-in-london`)
# for the Low Carbon London experiment. If any source is unavailable, the affected experiment falls back to
# synthetic data with a known ground truth so that the pipeline and every evaluation still run end-to-end,
# and the output clearly says which mode was used.
#
# CPU only; total run time is minutes (real mode is dominated by download time).

# %% [markdown]
# ### The technologies behind the data
#
# <table style="border:none">
# <tr>
# <td style="text-align:center;border:none"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Electrical_power_tower.jpg/960px-Electrical_power_tower.jpg" width="330"><br><sub><b>Bulk transmission.</b> CAISO schedules up to ~50 GW across lines like these; EIA-930 reports the hourly balancing-authority demand we forecast.<br><i>Dennis Schroeder / NREL, public domain, Wikimedia Commons</i></sub></td>
# <td style="text-align:center;border:none"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/ERCOTOperator_2.jpg/960px-ERCOTOperator_2.jpg" width="330"><br><sub><b>Grid operator at a system-operator control desk (ERCOT).</b> The load curve on the wall is the forecast this work explains; the operator's question is "what if tomorrow is 5 °F hotter — and how much of that can an appeal move?"<br><i>Dpysh w, CC BY 3.0, Wikimedia Commons</i></sub></td>
# <td style="text-align:center;border:none"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/Condenser_unit_for_central_air_conditioning.JPG/960px-Condenser_unit_for_central_air_conditioning.JPG" width="330"><br><sub><b>Central air-conditioning condenser.</b> The thermal layer: cooling load follows lagged temperature through building thermal mass. This is what a Flex Alert asks people to turn down.<br><i>H Padleckas, CC BY-SA 3.0, Wikimedia Commons</i></sub></td>
# </tr>
# <tr>
# <td style="text-align:center;border:none"><img src="https://upload.wikimedia.org/wikipedia/commons/2/2d/ASOS_Weather_station_being_installed.jpg" width="330"><br><sub><b>ASOS automated surface observing station.</b> NOAA's Integrated Surface Database (ISD) carries these hourly temperature, dew-point, wind and ceiling records for LAX, Sacramento, San Jose and Fresno.<br><i>Aviation Expert I, CC BY-SA 4.0, Wikimedia Commons</i></sub></td>
# <td style="text-align:center;border:none"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Itron_OpenWay_Electricity_Meter_with_Two-Way_Communications.JPG/960px-Itron_OpenWay_Electricity_Meter_with_Two-Way_Communications.JPG" width="330"><br><sub><b>Two-way smart meter.</b> The Low Carbon London trial metered ~5,500 households at half-hourly resolution and randomized ~1,100 of them onto a dynamic time-of-use tariff — our cleanest behavioral contrast.<br><i>Dwight Burdette, CC BY 3.0, Wikimedia Commons</i></sub></td>
# <td style="text-align:center;border:none;vertical-align:top"><sub><b>Natural experiments used for identification</b><br><br>
# • <b>CAISO Flex Alerts</b> — public conservation appeals (typically 4–9 pm) on 43 dated afternoons 2016–2022<br><br>
# • <b>Low Carbon London dToU trial</b> — randomized High/Normal/Low price signals, 2013<br><br>
# • <b>COVID-19 stay-at-home order</b> — occupancy regime change from 19 Mar 2020, matched year-over-year on weather<br><br>
# Each shifts behavior while weather is held fixed by design or by matching.</sub></td>
# </tr>
# </table>

# %%
# ---------------------------------------------------------------------------
# 0. PIPELINE DIAGRAM
# ---------------------------------------------------------------------------
import matplotlib.pyplot as _plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def _box(ax, x, y, w, h, title, body, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h - 0.09, title, ha="center", va="top", fontsize=10, weight="bold", color="#222222")
    ax.text(x + w / 2, y + h / 2 - 0.05, body, ha="center", va="center", fontsize=8, color="#333333", linespacing=1.4)

def _arrow(ax, p, q, text=None):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=1.3, color="#555555", shrinkA=2, shrinkB=2))
    if text: ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.05, text, ha="center", fontsize=7.5, color="#555555")

fig, ax = _plt.subplots(figsize=(14, 6.2)); ax.set_xlim(0, 14); ax.set_ylim(0, 6.2); ax.axis("off")
# inputs
_box(ax, 0.3, 4.3, 2.6, 1.5, "Hourly load", "EIA-930 (CAISO)\n2016–2024", "#fff3e8", "#eb6834")
_box(ax, 0.3, 2.45, 2.6, 1.5, "Hourly weather", "NOAA ISD, 4 stations\nT · dew · wind · cloud", "#eaf2fc", "#2a78d6")
_box(ax, 0.3, 0.6, 2.6, 1.5, "Natural experiments", "Flex Alert dates · COVID order\nLow Carbon London dToU", "#e8f7f1", "#1baf7a")
# layers
_box(ax, 4.0, 4.3, 3.0, 1.5, "1 · Thermal layer", "CDH/HDH with thermal-mass lag\nNLS, physically constrained\nfit on event-free hours only", "#eaf2fc", "#2a78d6")
_box(ax, 4.0, 2.45, 3.0, 1.5, "2 · Behavioral layer", "appeal & occupancy coefficients\nmatched difference-in-differences\nnever fitted on the full series", "#e8f7f1", "#1baf7a")
_box(ax, 4.0, 0.6, 3.0, 1.5, "3 · Residual forecaster", "LightGBM quantiles (q10/q50/q90)\ncalendar + lags + layers 1–2\nconformalized band", "#fff8e6", "#eda100")
# engine
_box(ax, 8.1, 2.0, 2.7, 3.0, "Counterfactual engine", "abduction → action → prediction\n\nkNN plausibility projection\nΔ = thermal + behavioral\n   + residual (unexplained)\n\nanalog-day spread", "#f4f4f4", "#666666")
# evaluations
_box(ax, 11.6, 4.5, 2.2, 1.3, "Eval A", "accuracy vs\nplain LGBM, analog,\npersistence", "#ffffff", "#999999")
_box(ax, 11.6, 3.0, 2.2, 1.3, "Eval B · central", "held-out Flex Alert\ndays: implied vs\nobserved DiD", "#ffffff", "#eb6834")
_box(ax, 11.6, 1.5, 2.2, 1.3, "Eval C / D", "COVID holdout ·\nLCL ToU transfer", "#ffffff", "#999999")
_box(ax, 11.6, 0.1, 2.2, 1.2, "Ablation", "behavioral layer on\nfull series", "#ffffff", "#999999")
for y in (5.05, 3.2, 1.35):
    _arrow(ax, (2.9, y), (4.0, y))
_arrow(ax, (2.9, 3.2), (4.0, 5.0)); _arrow(ax, (2.9, 3.2), (4.0, 1.4))
_arrow(ax, (7.0, 5.05), (8.1, 4.4)); _arrow(ax, (7.0, 3.2), (8.1, 3.5)); _arrow(ax, (7.0, 1.35), (8.1, 2.6))
_arrow(ax, (5.5, 4.3), (5.5, 3.95), "R = (L − base − thermal)/(base + thermal)")
_arrow(ax, (5.5, 2.45), (5.5, 2.1))
for y in (5.15, 3.65, 2.15, 0.7):
    _arrow(ax, (10.8, 3.5), (11.6, y))
ax.text(7, 6.05, "PALIMPSEST — identified counterfactual explanations for load forecasting", ha="center", fontsize=12, weight="bold")
_plt.tight_layout(); _plt.show()

# %%
# ---------------------------------------------------------------------------
# 1. SETUP & CONFIG
# ---------------------------------------------------------------------------
import os, sys, io, json, math, glob, hashlib, warnings, time
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.optimize import least_squares
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)
np.random.seed(0)

ON_KAGGLE = Path("/kaggle").exists()
WORK = Path("/kaggle/working") if ON_KAGGLE else Path("./palimpsest_work")
CACHE = WORK / "cache"
OUT = WORK / "outputs"
for p in (CACHE, OUT):
    p.mkdir(parents=True, exist_ok=True)

# --- Chart style: fixed-order categorical palette (validated CVD-safe), thin marks, recessive axes.
PAL = {
    "identified": "#2a78d6",   # slot 1 blue
    "ablation":   "#eb6834",   # slot 2 orange
    "flagfeat":   "#1baf7a",   # slot 3 aqua
    "plain":      "#eda100",   # slot 4 yellow
    "truth":      "#e87ba4",   # slot 5 magenta (synthetic ground truth only)
    "observed":   "#333333",   # ink
    "seq":        ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"],
}
matplotlib.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.edgecolor": "#999999", "lines.linewidth": 2, "font.size": 10,
})

CFG = dict(
    BA="CISO",                       # EIA-930 balancing authority code
    YEARS=list(range(2016, 2025)),   # EIA-930 begins July 2015
    TZ="America/Los_Angeles",
    # NOAA ISD stations (USAF+WBAN) and population-style weights for a CAISO composite temperature
    STATIONS={"72295023174": ("Los Angeles Intl", 0.45),
              "72483023232": ("Sacramento Exec", 0.20),
              "72494523293": ("San Jose Intl", 0.20),
              "72389093193": ("Fresno Yosemite", 0.15)},
    HOLDOUT_START="2023-01-01",      # accuracy holdout
    FLEX_HOLDOUT_AFTER_YEAR=2021,    # Flex Alert days after this year are interventional holdouts
    ALERT_HOURS=list(range(13, 24)), # hours the appeal coefficient is allowed to be non-zero (pre-cool .. rebound)
    ALERT_WINDOW=list(range(16, 21)),# the stated 4–9 pm Flex Alert window, used for scalar effect summaries
    COVID_START="2020-03-19",        # California stay-at-home order
    COVID_ID_END="2020-04-30",       # lockdown days used to identify the occupancy coefficient
    COVID_HOLDOUT_END="2020-06-15",  # lockdown days held out ("same weather, 2020 occupancy" test)
    MATCH_K=5, MATCH_DOY_WINDOW=45,
    LGB_TREES=400, LGB_LR=0.05, LGB_LEAVES=31,
    KNN_ANALOGS=10,
    LCL_MAX_BLOCKS=int(os.environ.get("LCL_MAX_BLOCKS", "40")),   # Kaggle mirror has 112 hhblock files
    FORCE_SYNTHETIC=os.environ.get("PALIMPSEST_FORCE_SYNTHETIC", "0") == "1",
    SEED=0,
)
print("Kaggle:", ON_KAGGLE, "| work dir:", WORK.resolve())

# %%
# ---------------------------------------------------------------------------
# 1b. UTILITIES
# ---------------------------------------------------------------------------
def have_internet(timeout=6):
    if CFG["FORCE_SYNTHETIC"]:
        return False
    try:
        import requests
        requests.head("https://www.eia.gov", timeout=timeout)
        return True
    except Exception:
        return False

def download(url, dest, retries=2, timeout=180):
    """Download url to dest (cached). Returns dest or None."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    import requests
    for i in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "palimpsest-notebook"})
            if r.status_code == 200 and len(r.content) > 1000:
                dest.write_bytes(r.content)
                return dest
            print(f"  [{r.status_code}] {url}")
            if r.status_code == 404:
                return None
        except Exception as e:
            print(f"  download error ({i}): {e}")
        time.sleep(2)
    return None

try:
    import holidays as _hol
    _US = _hol.US(years=range(2013, 2027))
    def is_us_holiday(d): return d in _US
except Exception:
    def _nth_weekday(y, m, wd, n):
        d = pd.Timestamp(y, m, 1)
        off = (wd - d.dayofweek) % 7
        return d + pd.Timedelta(days=off + 7 * (n - 1))
    def _last_weekday(y, m, wd):
        d = pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
        return d - pd.Timedelta(days=(d.dayofweek - wd) % 7)
    _cache = {}
    def is_us_holiday(d):
        d = pd.Timestamp(d).normalize(); y = d.year
        if y not in _cache:
            _cache[y] = {pd.Timestamp(y, 1, 1), pd.Timestamp(y, 7, 4), pd.Timestamp(y, 12, 25),
                         pd.Timestamp(y, 11, 11), _nth_weekday(y, 1, 0, 3), _nth_weekday(y, 2, 0, 3),
                         _last_weekday(y, 5, 0), _nth_weekday(y, 9, 0, 1), _nth_weekday(y, 11, 3, 4)}
        return d in _cache[y]

def add_calendar(df):
    idx = df.index
    df["hour"] = idx.hour; df["dow"] = idx.dayofweek; df["month"] = idx.month
    df["doy"] = idx.dayofyear; df["year"] = idx.year
    df["date"] = idx.normalize()
    hol = pd.Series(idx.normalize().unique()).map(is_us_holiday)
    holmap = dict(zip(idx.normalize().unique(), hol.values))
    df["is_holiday"] = df["date"].map(holmap).astype(int)
    # daytype: 0 weekday, 1 saturday, 2 sunday/holiday
    df["daytype"] = np.where(df["is_holiday"] == 1, 2, np.where(df["dow"] == 6, 2, np.where(df["dow"] == 5, 1, 0)))
    return df

def bootstrap_ci(x, stat=np.mean, B=2000, q=(0.1, 0.9), seed=0):
    x = np.asarray(x); rng = np.random.default_rng(seed)
    if len(x) == 0: return (np.nan, np.nan)
    s = [stat(x[rng.integers(0, len(x), len(x))], axis=0) for _ in range(B)]
    return np.quantile(np.array(s), q[0], axis=0), np.quantile(np.array(s), q[1], axis=0)

def mape(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float); m = np.isfinite(y) & np.isfinite(p) & (y != 0)
    return 100 * np.mean(np.abs((y[m] - p[m]) / y[m]))

def pinball(y, p, q):
    y = np.asarray(y, float); p = np.asarray(p, float); d = y - p
    return np.nanmean(np.maximum(q * d, (q - 1) * d))

RESULTS = {}   # everything reported at the end is collected here

# %% [markdown]
# ## 2. Data ingestion (free public sources)
#
# * **EIA-930** six-month balance files: `https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_{year}_{Jan_Jun|Jul_Dec}.csv`
# * **NOAA ISD** global-hourly CSVs: `https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{station}.csv`
# * **CAISO Flex Alerts**: dated public conservation appeals, compiled below from CAISO/Flex Alert public notices.
#   Edit `FLEX_ALERT_DATES` if you have the authoritative archive; the notebook treats the list as data.
# * **Low Carbon London**: Kaggle dataset `jeanmidev/smart-meters-in-london` (attach as input); optionally a
#   `Tariffs.csv` (TariffDateTime, Tariff ∈ {High, Normal, Low}) from the UK Data Service release.

# %%
FLEX_ALERT_DATES = """
2016-06-20 2016-07-26 2016-07-27 2016-07-28
2017-06-20 2017-06-21 2017-06-22 2017-08-28 2017-08-29 2017-08-30 2017-08-31 2017-09-01
2018-07-24 2018-07-25
2020-08-14 2020-08-15 2020-08-16 2020-08-17 2020-08-18 2020-08-19 2020-09-05 2020-09-06 2020-09-07 2020-10-01 2020-10-15
2021-06-17 2021-06-18 2021-07-09 2021-07-10 2021-07-12 2021-07-28 2021-08-17 2021-09-08
2022-08-31 2022-09-01 2022-09-02 2022-09-03 2022-09-04 2022-09-05 2022-09-06 2022-09-07 2022-09-08 2022-09-09
""".split()
FLEX_ALERT_DATES = sorted(set(pd.to_datetime(FLEX_ALERT_DATES)))
print(f"{len(FLEX_ALERT_DATES)} Flex Alert days listed, "
      f"{sum(d.year > CFG['FLEX_HOLDOUT_AFTER_YEAR'] for d in FLEX_ALERT_DATES)} of them in the holdout (> {CFG['FLEX_HOLDOUT_AFTER_YEAR']})")

def load_eia930(ba, years):
    frames = []
    for y in years:
        for half, tag in ((0, "Jan_Jun"), (1, "Jul_Dec")):
            if y == 2015 and half == 0:
                continue
            url = f"https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_{y}_{tag}.csv"
            p = download(url, CACHE / f"eia930_{y}_{tag}.csv")
            if p is None:
                print("  missing", url); continue
            head = pd.read_csv(p, nrows=2)
            cols = list(head.columns)
            ba_col = next(c for c in cols if "Balancing Authority" in c)
            utc_col = next(c for c in cols if "UTC Time" in c)
            dem_adj = [c for c in cols if c.startswith("Demand (MW) (Adjusted)")]
            dem_col = dem_adj[0] if dem_adj else next(c for c in cols if c.startswith("Demand (MW)"))
            df = pd.read_csv(p, usecols=[ba_col, utc_col, dem_col], thousands=",", low_memory=False)
            df = df[df[ba_col] == ba]
            ts = pd.to_datetime(df[utc_col], errors="coerce", utc=True)
            # EIA stamps hour *end*; shift to hour start and convert to local wall time
            ts = (ts - pd.Timedelta(hours=1)).dt.tz_convert(CFG["TZ"]).dt.tz_localize(None)
            frames.append(pd.DataFrame({"load": pd.to_numeric(df[dem_col], errors="coerce").values}, index=ts))
            print(f"  EIA-930 {y} {tag}: {len(df)} rows for {ba}")
    if not frames:
        return None
    s = pd.concat(frames).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    s = s[s.index.notna()]
    full = pd.date_range(s.index.min().normalize(), s.index.max().normalize() + pd.Timedelta(hours=23), freq="h")
    s = s.reindex(full)
    s.loc[(s["load"] <= 0) | (s["load"] > 5 * s["load"].median()), "load"] = np.nan
    s["load"] = s["load"].interpolate(limit=6)
    return s

def _isd_tenths(col, missing=9999):
    v = pd.to_numeric(col.astype(str).str.split(",").str[0], errors="coerce")
    v[np.abs(v) >= missing] = np.nan
    return v / 10.0

def load_isd(station, years):
    frames = []
    for y in years:
        p = download(f"https://www.ncei.noaa.gov/data/global-hourly/access/{y}/{station}.csv", CACHE / f"isd_{station}_{y}.csv")
        if p is None:
            continue
        df = pd.read_csv(p, usecols=["DATE", "TMP", "DEW", "WND", "CIG"], low_memory=False)
        out = pd.DataFrame({
            "T": _isd_tenths(df["TMP"]),
            "dew": _isd_tenths(df["DEW"]),
        })
        wnd = pd.to_numeric(df["WND"].astype(str).str.split(",").str[3], errors="coerce"); wnd[wnd >= 9999] = np.nan
        out["wind"] = wnd / 10.0
        cig = pd.to_numeric(df["CIG"].astype(str).str.split(",").str[0], errors="coerce"); cig[cig >= 99999] = np.nan
        out["cloud"] = 1.0 - np.clip(cig / 22000.0, 0, 1)   # ceiling-height proxy for cloud cover (22000 m = unlimited)
        ts = pd.to_datetime(df["DATE"], errors="coerce", utc=True).dt.tz_convert(CFG["TZ"]).dt.tz_localize(None)
        out.index = ts.dt.floor("h")
        frames.append(out[out.index.notna()])
    if not frames:
        return None
    w = pd.concat(frames).groupby(level=0).mean().sort_index()
    return w

def load_weather_composite(stations, years, index):
    parts, weights = [], []
    for sid, (name, wgt) in stations.items():
        w = load_isd(sid, years)
        if w is None:
            print("  no weather for", name); continue
        w = w.reindex(index).interpolate(limit=12)
        print(f"  ISD {name}: {w['T'].notna().mean():.1%} hours with temperature")
        parts.append(w); weights.append(wgt)
    if not parts:
        return None
    W = np.array(weights)
    comp = {}
    for c in ["T", "dew", "wind", "cloud"]:
        M = np.column_stack([p[c].values for p in parts])
        mask = np.isfinite(M)
        num = np.nansum(M * W, axis=1); den = (mask * W).sum(axis=1)
        comp[c] = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    out = pd.DataFrame(comp, index=index)
    return out.interpolate(limit=24).bfill().ffill()

# %%
# ---------------------------------------------------------------------------
# 3. SYNTHETIC FALLBACK with known ground truth
# ---------------------------------------------------------------------------
TRUE = {}   # populated only in synthetic mode

def make_synthetic_ba(years, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31 23:00", freq="h")
    n = len(idx); nd = n // 24
    doy = idx.dayofyear.values; hr = idx.hour.values; day_i = np.arange(n) // 24
    season = -np.cos(2 * np.pi * (doy - 15) / 365.0); season_d = season[::24]
    a = np.zeros(nd)
    for d in range(1, nd):
        a[d] = 0.8 * a[d - 1] + rng.normal(0, 3.0)
    T = 16 + 9 * season + (5 + 2 * season) * (-np.cos(2 * np.pi * (hr - 4) / 24)) + a[day_i] + rng.normal(0, 0.8, n)
    dew = T - 6 - 4 * season - np.abs(rng.normal(0, 3, n)) - 0.4 * a[day_i]
    cloudd = np.clip(0.5 - 0.3 * season_d + rng.normal(0, 0.25, nd), 0, 1)
    cloud = np.clip(cloudd[day_i] + rng.normal(0, 0.08, n), 0, 1)
    wind = np.abs(rng.normal(3.0, 1.5, n)) + 1.5 * cloud
    df = pd.DataFrame({"T": T, "dew": dew, "wind": wind, "cloud": cloud}, index=idx)
    add_calendar(df)
    # true thermal term. Deliberately richer than the model's functional form: latent (humidity) cooling load
    # and a mild saturation at extreme heat, because real buildings have physics the parametric term omits.
    th = dict(alpha=0.85, Tc=20.5, Th=12.0, bc=900.0, bc2=25.0, bh=500.0, b_hum=14.0, b_sat=-0.35)
    Tl = lfilter([1 - th["alpha"]], [1, -th["alpha"]], T, zi=[T[0] * th["alpha"]])[0]
    CDH = np.maximum(Tl - th["Tc"], 0); HDH = np.maximum(th["Th"] - Tl, 0)
    thermal = (th["bc"] * CDH + th["bc2"] * CDH ** 2 + th["bh"] * HDH
               + th["b_hum"] * CDH * np.maximum(dew - 10, 0) + th["b_sat"] * CDH ** 3)
    # base profile
    h = np.arange(24)
    shape = 0.55 * np.exp(-((h - 9) / 3.5) ** 2) + np.exp(-((h - 18.5) / 3.0) ** 2) - 0.35 * np.exp(-((h - 3.5) / 3) ** 2)
    prof = 20000 + 5000 * shape
    dtf = np.array([1.0, 0.92, 0.88])
    base = prof[hr] * dtf[df["daytype"].values] * (1 + 0.012 * (df["year"].values - years[0]))
    # true behavioral effects
    covid_eff = np.zeros((3, 24))
    covid_eff[0, 7:18] = -0.10; covid_eff[0, 18:23] = -0.03; covid_eff[0, :7] = 0.02; covid_eff[0, 23] = 0.02
    covid_eff[1, 8:20] = -0.05; covid_eff[2, 8:20] = -0.04
    alert_eff = np.zeros(24)
    alert_eff[15] = -0.01; alert_eff[16] = -0.03; alert_eff[17:19] = -0.045; alert_eff[19] = -0.04
    alert_eff[20] = -0.03; alert_eff[21] = -0.01; alert_eff[22:24] = 0.01
    covid = ((df["date"] >= CFG["COVID_START"]) & (df["date"] <= CFG["COVID_HOLDOUT_END"])).astype(int).values
    # alert days = hottest summer afternoons each year (deliberately confounded with temperature)
    daily = df.groupby("date").agg(Tmax=("T", "max"), month=("month", "first"), daytype=("daytype", "first"))
    alert_days = []
    for y in years:
        cand = daily[(daily.index.year == y) & (daily.month.between(6, 9)) & (daily.daytype == 0)]
        nsel = {2019: 2}.get(y, 6)
        alert_days += list(cand.Tmax.nlargest(nsel).index)
    alert = df["date"].isin(alert_days).astype(int).values
    # noise: AR(1) hourly + daily effect
    e = np.zeros(n); e[0] = rng.normal(0, 350)
    eps = rng.normal(0, 350 * math.sqrt(1 - 0.9 ** 2), n)
    for t in range(1, n):
        e[t] = 0.9 * e[t - 1] + eps[t]
    e += rng.normal(0, 300, nd)[day_i]
    occ = 1 + covid_eff[df["daytype"].values, hr] * covid
    app = 1 + alert_eff[hr] * alert
    load = (base * occ + thermal) * app + e
    df["load"] = load; df["alert"] = alert; df["covid"] = covid
    TRUE.update(dict(thermal=th, alert_eff=alert_eff, covid_eff=covid_eff, alert_days=sorted(alert_days)))
    return df

def make_synthetic_lcl(seed=1):
    """Daily household-mean half-hourly profiles by group for a 2013-like ToU trial."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2013-01-01", "2013-12-31", freq="D")
    hh = np.arange(48)
    prof = 0.15 + 0.08 * np.exp(-((hh - 15) / 4) ** 2) + 0.25 * np.exp(-((hh - 37) / 5) ** 2)
    season = -np.cos(2 * np.pi * (np.asarray(days.dayofyear) - 15) / 365.0)
    tmean = 10 + 7 * season + rng.normal(0, 3, len(days))
    heat = np.maximum(14 - tmean, 0)[:, None] * 0.012 * (1 + 0.5 * (hh > 30))[None, :]
    high_days = set(rng.choice(len(days), 40, replace=False)); low_days = set(rng.choice(len(days), 40, replace=False))
    tariff = np.full((len(days), 48), "Normal", dtype=object)
    tou_eff = np.zeros((len(days), 48))
    for d in range(len(days)):
        if d in high_days:
            tariff[d, 32:39] = "High"; tou_eff[d, 32:39] = -0.14; tou_eff[d, 39:46] = 0.05
        if d in low_days:
            tariff[d, 0:12] = "Low"; tou_eff[d, 0:12] = 0.10
    rows = []
    groups = {"Std": 800, "ToU_A": 200, "ToU_B": 200}
    for g, nh in groups.items():
        eff = tou_eff if g.startswith("ToU") else 0
        mean = (prof[None, :] + heat) * (1 + eff) * (1 + 0.03 * rng.normal(size=(len(days), 1)))
        noise = rng.normal(0, 0.012, mean.shape) / math.sqrt(nh / 200)
        M = mean + noise
        for d in range(len(days)):
            rows.append([days[d], g, nh, tmean[d]] + list(M[d]))
    lcl = pd.DataFrame(rows, columns=["day", "group", "n", "tmean"] + [f"hh_{i}" for i in range(48)])
    tariffs = pd.DataFrame(tariff, index=days, columns=[f"hh_{i}" for i in range(48)])
    TRUE["lcl_tou_eff"] = {"High": -0.14, "Low": 0.10, "post-High": 0.05}
    return lcl, tariffs

# %%
# ---------------------------------------------------------------------------
# 3b. ASSEMBLE THE BALANCING-AUTHORITY DATASET
# ---------------------------------------------------------------------------
DATA_MODE = "synthetic"
df = None
if have_internet():
    print("Internet available — downloading EIA-930 and NOAA ISD (cached in", CACHE, ")")
    try:
        L = load_eia930(CFG["BA"], CFG["YEARS"])
        if L is not None and L["load"].notna().mean() > 0.9:
            W = load_weather_composite(CFG["STATIONS"], CFG["YEARS"], L.index)
            if W is not None and W["T"].notna().mean() > 0.95:
                df = L.join(W)
                add_calendar(df)
                df["alert"] = df["date"].isin(FLEX_ALERT_DATES).astype(int)
                df["covid"] = ((df["date"] >= CFG["COVID_START"]) & (df["date"] <= CFG["COVID_HOLDOUT_END"])).astype(int)
                DATA_MODE = "real"
    except Exception as e:
        print("Real-data ingestion failed:", repr(e))
if df is None:
    print("Using SYNTHETIC balancing-authority data with known ground truth.")
    df = make_synthetic_ba(CFG["YEARS"], CFG["SEED"])
    df["alert"] = df["alert"].astype(int)
df = df.dropna(subset=["load"]).copy()
df["T"] = df["T"].interpolate().bfill().ffill()
for c in ["dew", "wind", "cloud"]:
    df[c] = df[c].interpolate().bfill().ffill()
RESULTS["data_mode"] = DATA_MODE
print(f"DATA_MODE = {DATA_MODE} | {len(df):,} hourly rows {df.index.min()} → {df.index.max()} | "
      f"{int(df.groupby('date')['alert'].max().sum())} alert days")
df[["load", "T", "dew", "wind", "cloud"]].describe().T

# %%
# lagged-load features (all lags ≥ 24 h so a day-ahead forecast never sees same-day load)
df["lag24"] = df["load"].shift(24)
df["lag168"] = df["load"].shift(168)
df["roll7"] = df["load"].shift(24).rolling(168, min_periods=100).mean()
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24); df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25); df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)

HOLDOUT_START = pd.Timestamp(CFG["HOLDOUT_START"])
covid_id = (df["date"] >= CFG["COVID_START"]) & (df["date"] <= CFG["COVID_ID_END"])
covid_ho = (df["date"] > CFG["COVID_ID_END"]) & (df["date"] <= CFG["COVID_HOLDOUT_END"])
alert_ho = (df["alert"] == 1) & (df["year"] > CFG["FLEX_HOLDOUT_AFTER_YEAR"])
df["is_train"] = ((df.index < HOLDOUT_START) & ~covid_ho & ~alert_ho).astype(int)
# rows allowed in the *thermal/baseline* fit: training rows with no behavioral event
df["is_clean"] = ((df["is_train"] == 1) & (df["alert"] == 0) & (df["covid"] == 0)).astype(int)
print("train rows:", int(df.is_train.sum()), "| clean rows for thermal fit:", int(df.is_clean.sum()),
      "| holdout rows:", int((df.index >= HOLDOUT_START).sum()))

# %% [markdown]
# ## 4. Thermal layer
#
# $$T^{lag}_t = \alpha T^{lag}_{t-1} + (1-\alpha) T_t,\qquad
# \text{CDH}_t = (T^{lag}_t - T_c)_+,\quad \text{HDH}_t = (T_h - T^{lag}_t)_+$$
# $$\text{thermal}_t = b_c\,\text{CDH}_t + b_{c2}\,\text{CDH}_t^2 + b_h\,\text{HDH}_t$$
#
# Six parameters ($\alpha, T_c, T_h, b_c, b_{c2}, b_h$), all physically constrained (non-negative coefficients,
# activation temperatures in plausible ranges). The calendar baseline (hour × day-type profile × annual factor) is
# solved in closed form inside the residual so the two are fitted jointly. The fit uses **only** rows with no
# behavioral event (no Flex Alert, no lockdown) — this is the first half of the identification strategy.

# %%
PNAMES = ["alpha", "Tc", "Th", "bc", "bc2", "bh"]

def thermal_terms(T, p):
    alpha = p[0]
    Tl = lfilter([1 - alpha], [1, -alpha], T, zi=[T[0] * alpha])[0]
    return Tl, np.maximum(Tl - p[1], 0), np.maximum(p[2] - Tl, 0)

def thermal_predict(T, p):
    Tl, CDH, HDH = thermal_terms(T, p)
    return p[3] * CDH + p[4] * CDH ** 2 + p[5] * HDH, Tl, CDH, HDH

class Baseline:
    """hour × daytype profile × annual factor (unseen years use the last fitted factor)."""
    def __init__(self, prof, yf):
        self.prof, self.yf = prof, yf; self.last = max(yf)
    def predict(self, hour, daytype, year):
        yrs = np.asarray(year)
        f = np.array([self.yf.get(y, self.yf[self.last]) for y in np.unique(yrs)])
        fmap = dict(zip(np.unique(yrs), f))
        return self.prof[np.asarray(hour) * 3 + np.asarray(daytype)] * np.vectorize(fmap.get)(yrs)

def fit_baseline(y, hour, daytype, year, mask):
    cell = hour * 3 + daytype
    cnt = np.bincount(cell[mask], minlength=72); s = np.bincount(cell[mask], weights=y[mask], minlength=72)
    prof = s / np.maximum(cnt, 1)
    ratio = y / np.maximum(prof[cell], 1e-6)
    yf = pd.Series(ratio[mask]).groupby(year[mask]).mean().to_dict()
    return Baseline(prof, yf)

def fit_thermal(df, mask, p0=None, max_rows=60000, seed=0):
    T = df["T"].values.astype(float); load = df["load"].values.astype(float)
    hour = df["hour"].values; dt = df["daytype"].values; year = df["year"].values
    mask = np.asarray(mask, bool) & np.isfinite(load)
    idx = np.where(mask)[0]
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(idx, size=min(len(idx), max_rows), replace=False))
    scale = np.nanstd(load[idx])
    mean_load = np.nanmean(load[idx])
    if p0 is None:
        p0 = [0.8, 20.0, 12.0, 0.02 * mean_load, 0.0005 * mean_load, 0.01 * mean_load]
    lb = [0.0, 14.0, 4.0, 0.0, 0.0, 0.0]; ub = [0.985, 27.0, 18.0, 0.2 * mean_load, 0.02 * mean_load, 0.2 * mean_load]
    def resid(p):
        th, *_ = thermal_predict(T, p)
        y = load - th
        b = fit_baseline(y, hour, dt, year, mask)
        return (y[sub] - b.predict(hour[sub], dt[sub], year[sub])) / scale
    sol = least_squares(resid, p0, bounds=(lb, ub), x_scale="jac", max_nfev=300, loss="soft_l1", f_scale=2.0)
    p = sol.x
    th, *_ = thermal_predict(T, p)
    base = fit_baseline(load - th, hour, dt, year, mask)
    return p, base

t0 = time.time()
THERMAL_P, BASELINE = fit_thermal(df, df["is_clean"].values == 1)
print(f"thermal fit in {time.time()-t0:.1f}s:", {k: round(float(v), 3) for k, v in zip(PNAMES, THERMAL_P)})
if TRUE:
    print("ground truth:                ", TRUE["thermal"])
RESULTS["thermal_params"] = dict(zip(PNAMES, map(float, THERMAL_P)))

# %%
def structural(frame, p, base, behav):
    """Layers 1–2 on any hourly frame with columns T, hour, daytype, year, alert, covid.
    behav = {'alert': (24,), 'covid': (3,24)} multiplicative coefficients on (baseline + thermal), i.e. in
    units of *fraction of load* — the same units as an observed effect or a counterfactual delta."""
    th, Tl, CDH, HDH = thermal_predict(frame["T"].values.astype(float), p)
    b = base.predict(frame["hour"].values, frame["daytype"].values, frame["year"].values)
    beh = (b + th) * (behav["alert"][frame["hour"].values] * frame["alert"].values
               + behav["covid"][frame["daytype"].values, frame["hour"].values] * frame["covid"].values)
    return pd.DataFrame({"base": b, "thermal": th, "behav": beh, "struct": b + th + beh,
                         "Tlag": Tl, "CDH": CDH, "HDH": HDH}, index=frame.index)

ZERO_BEHAV = {"alert": np.zeros(24), "covid": np.zeros((3, 24))}
S0 = structural(df, THERMAL_P, BASELINE, ZERO_BEHAV)
df["R"] = (df["load"] - S0["base"] - S0["thermal"]) / (S0["base"] + S0["thermal"])   # normalized residual after layers 0–1
df["Tlag"] = S0["Tlag"]
print("normalized residual on clean rows: mean %.4f  sd %.4f" % (df.loc[df.is_clean == 1, "R"].mean(), df.loc[df.is_clean == 1, "R"].std()))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
clean = df[df.is_clean == 1]
bins = pd.cut(clean["Tlag"], np.arange(-5, 45, 1))
m = (clean["load"] - S0.loc[clean.index, "base"]).groupby(bins).mean()
ax[0].plot([b.mid for b in m.index], m.values, color=PAL["observed"], label="observed load − baseline")
tt = np.arange(-5, 45, 0.5)
ax[0].plot(tt, thermal_predict(tt, THERMAL_P)[0], color=PAL["identified"], label="fitted thermal term")
ax[0].set_xlabel("lagged temperature (°C)"); ax[0].set_ylabel("MW"); ax[0].legend(frameon=False); ax[0].set_title("Thermal layer")
hp = clean.groupby("hour")["R"].std()
ax[1].bar(hp.index, 100 * hp.values, color=PAL["seq"][3], width=0.7)
ax[1].set_xlabel("hour"); ax[1].set_ylabel("sd of normalized residual (%)"); ax[1].set_title("Residual after thermal + calendar")
plt.tight_layout(); plt.savefig(OUT / "fig_thermal.png"); plt.show()

# %% [markdown]
# ## 5. Behavioral layer — identified from natural experiments
#
# The normalized residual $R_t = (L_t - \text{base}_t - \text{thermal}_t)/(\text{base}_t + \text{thermal}_t)$ is what
# remains after the physics and the calendar, expressed as a fraction of load so that a coefficient, an observed
# effect and a counterfactual delta all share one unit. Behavioral coefficients are estimated **only** from
# contrasts that shift behavior while holding weather fixed:
#
# * **Flex Alerts** — each training alert day is matched to its $k$ nearest non-alert days (same day-type, within ±45
#   days of the calendar, nearest in daily max/mean/lagged-afternoon temperature and dew point). The per-hour
#   difference in $R$ is a difference-in-differences estimate of the appeal effect.
# * **COVID lockdown** — lockdown days in the identification window are matched to 2018–2019 days the same way,
#   giving an occupancy shift per (day-type, hour).
#
# The **ablation** ("naive") fits the same coefficients on the full series: the thermal term is refitted including
# event days and the alert effect is the residual on alert days minus *all* other summer days — no weather matching.

# %%
daily = df.groupby("date").agg(
    Tmax=("T", "max"), Tmean=("T", "mean"), dew_mean=("dew", "mean"),
    Tlag_pm=("Tlag", lambda s: s.iloc[13:21].mean() if len(s) == 24 else s.mean()),
    daytype=("daytype", "first"), doy=("doy", "first"), year=("year", "first"), month=("month", "first"),
    is_holiday=("is_holiday", "max"), alert=("alert", "max"), covid=("covid", "max"),
    is_train=("is_train", "min"), nh=("load", "size"))
daily = daily[daily.nh == 24]
R_piv = df.pivot_table(index="date", columns="hour", values="R").reindex(daily.index)
MATCH_FEATS = ["Tmax", "Tmean", "Tlag_pm", "dew_mean"]

def match_days(daily, event_dates, pool_mask, k=None, doy_window=None):
    k = k or CFG["MATCH_K"]; doy_window = doy_window or CFG["MATCH_DOY_WINDOW"]
    pool = daily[pool_mask]; sd = daily[MATCH_FEATS].std()
    out = {}
    for d in event_dates:
        if d not in daily.index: continue
        row = daily.loc[d]
        cand = pool[pool.daytype == row.daytype]
        dd = (cand.doy - row.doy).abs(); dd = np.minimum(dd, 366 - dd)
        cand = cand[(dd <= doy_window) & (cand.index != d)]
        if len(cand) < k: continue
        dist = (((cand[MATCH_FEATS] - row[MATCH_FEATS]) / sd) ** 2).sum(axis=1)
        out[d] = dist.nsmallest(k).index.tolist()
    return out

def did_matrix(R_piv, matches):
    rows, dates = [], []
    for d, ctr in matches.items():
        rows.append(R_piv.loc[d].values - R_piv.loc[ctr].mean(axis=0).values); dates.append(d)
    return pd.DataFrame(rows, index=dates, columns=range(24))

# --- identified Flex Alert coefficient
train_alert_days = daily[(daily.alert == 1) & (daily.year <= CFG["FLEX_HOLDOUT_AFTER_YEAR"])].index
pool_train = (daily.is_train == 1) & (daily.alert == 0) & (daily.covid == 0) & (daily.is_holiday == 0)
m_alert = match_days(daily, train_alert_days, pool_train)
E_alert = did_matrix(R_piv, m_alert)
beta_alert = np.zeros(24)
beta_alert[CFG["ALERT_HOURS"]] = E_alert.mean(axis=0).values[CFG["ALERT_HOURS"]]
lo, hi = bootstrap_ci(E_alert.values)
print(f"Flex Alert: {len(E_alert)} training alert days matched. Window-average effect "
      f"{100*beta_alert[CFG['ALERT_WINDOW']].mean():+.2f}%  (80% CI {100*lo[CFG['ALERT_WINDOW']].mean():+.2f} .. {100*hi[CFG['ALERT_WINDOW']].mean():+.2f})")

# --- identified COVID occupancy coefficient
covid_id_days = daily[(daily.covid == 1) & (daily.index <= CFG["COVID_ID_END"])].index
pool_pre = (daily.year.isin([2018, 2019])) & (daily.alert == 0)
m_covid = match_days(daily, covid_id_days, pool_pre, k=CFG["MATCH_K"], doy_window=60)
E_covid = did_matrix(R_piv, m_covid)
beta_covid = np.zeros((3, 24))
for dt in range(3):
    ds = [d for d in E_covid.index if daily.loc[d, "daytype"] == dt]
    if ds: beta_covid[dt] = E_covid.loc[ds].mean(axis=0).values
print(f"COVID: {len(E_covid)} lockdown days matched; weekday daytime (9–16h) occupancy shift {100*beta_covid[0, 9:17].mean():+.2f}%")
BEHAV_ID = {"alert": beta_alert, "covid": beta_covid}

# --- ABLATION: full-series fit (thermal includes event days; unmatched contrasts)
THERMAL_P_NAIVE, BASELINE_NAIVE = fit_thermal(df, df["is_train"].values == 1, p0=THERMAL_P)
S_naive = structural(df, THERMAL_P_NAIVE, BASELINE_NAIVE, ZERO_BEHAV)
df["R_naive"] = (df["load"] - S_naive["base"] - S_naive["thermal"]) / (S_naive["base"] + S_naive["thermal"])
Rn_piv = df.pivot_table(index="date", columns="hour", values="R_naive").reindex(daily.index)
summer_ctrl = daily[(daily.is_train == 1) & (daily.alert == 0) & (daily.month.between(6, 10)) & (daily.daytype == 0) & (daily.covid == 0)].index
beta_alert_naive = np.zeros(24)
beta_alert_naive[CFG["ALERT_HOURS"]] = (Rn_piv.loc[train_alert_days].mean() - Rn_piv.loc[summer_ctrl].mean()).values[CFG["ALERT_HOURS"]]
beta_covid_naive = np.zeros((3, 24))
for dt in range(3):
    ev = daily[(daily.index.isin(covid_id_days)) & (daily.daytype == dt)].index
    ctr = daily[(daily.year.isin([2018, 2019])) & (daily.month.isin([3, 4])) & (daily.daytype == dt)].index
    if len(ev): beta_covid_naive[dt] = (Rn_piv.loc[ev].mean() - Rn_piv.loc[ctr].mean()).values
BEHAV_NAIVE = {"alert": beta_alert_naive, "covid": beta_covid_naive}
print(f"ablation window-average alert effect {100*beta_alert_naive[CFG['ALERT_WINDOW']].mean():+.2f}%")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
h = np.arange(24)
ax[0].fill_between(h, 100 * lo, 100 * hi, color=PAL["identified"], alpha=0.15, linewidth=0)
ax[0].plot(h, 100 * E_alert.mean(axis=0).values, color=PAL["identified"], label="identified (matched DiD)")
ax[0].plot(h, 100 * (Rn_piv.loc[train_alert_days].mean() - Rn_piv.loc[summer_ctrl].mean()).values, color=PAL["ablation"], label="ablation (full series)")
if TRUE:
    ax[0].plot(h, 100 * TRUE["alert_eff"], color=PAL["truth"], linestyle="--", label="ground truth")
ax[0].axhline(0, color="#999999", linewidth=0.8); ax[0].set_xlabel("hour"); ax[0].set_ylabel("effect on load (%)")
ax[0].set_title("Flex Alert appeal effect (training days)"); ax[0].legend(frameon=False, fontsize=8)
ax[1].plot(h, 100 * beta_covid[0], color=PAL["identified"], label="identified, weekday")
ax[1].plot(h, 100 * beta_covid_naive[0], color=PAL["ablation"], label="ablation, weekday")
if TRUE: ax[1].plot(h, 100 * TRUE["covid_eff"][0], color=PAL["truth"], linestyle="--", label="ground truth (on baseline)")
ax[1].axhline(0, color="#999999", linewidth=0.8); ax[1].set_xlabel("hour"); ax[1].set_title("Lockdown occupancy effect"); ax[1].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "fig_behavioral.png"); plt.show()
RESULTS["beta_alert_identified"] = beta_alert.round(4).tolist()
RESULTS["beta_alert_ablation"] = beta_alert_naive.round(4).tolist()

# %% [markdown]
# ## 6. Residual forecaster (LightGBM) and baselines
#
# * **PALIMPSEST (identified)**: quantile LightGBM on the residual $L - \text{struct}$ with layers 1–2 as inputs.
# * **Ablation**: same, with the full-series thermal/behavioral parameters.
# * **Flag-as-feature**: plain LightGBM on load with `alert`/`covid` as ordinary features (the common practice).
# * **Plain LightGBM**: no structure, no flags.  **Persistence**: same hour last week.  **Similar-day**: kNN analog.

# %%
F_CAL = ["hour", "dow", "doy", "month", "daytype", "is_holiday", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
F_WX = ["T", "dew", "wind", "cloud"]
F_LAG = ["lag24", "lag168", "roll7"]
F_BASE = F_CAL + F_WX + F_LAG
F_STRUCT = F_BASE + ["Tlag", "CDH", "HDH", "base", "thermal", "behav", "struct"]
F_FLAG = F_BASE + ["alert", "covid"]
QS = [0.1, 0.5, 0.9]

def lgb_params(q):
    return dict(objective="quantile", alpha=q, n_estimators=CFG["LGB_TREES"], learning_rate=CFG["LGB_LR"],
                num_leaves=CFG["LGB_LEAVES"], min_child_samples=50, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, verbose=-1, random_state=CFG["SEED"])

class Forecaster:
    """Structured (residual) or direct LightGBM quantile forecaster."""
    def __init__(self, name, feats, thermal_p=None, base=None, behav=None, structured=True):
        self.name, self.feats, self.p, self.base, self.behav, self.structured = name, feats, thermal_p, base, behav, structured
        self.models = {}
    def frame(self, frame):
        X = frame.copy()
        if self.structured:
            S = structural(frame, self.p, self.base, self.behav)
            for c in S.columns: X[c] = S[c].values
        return X
    def fit(self, frame, calib_days=365):
        """Fit quantile models on all but the last `calib_days` of the frame, then conformalize the
        q10/q90 band on those days (CQR) so the nominal 80% interval is honest out of sample."""
        X = self.frame(frame)
        y = frame["load"].values - (X["struct"].values if self.structured else 0)
        ok = np.isfinite(y) & np.isfinite(X[self.feats].values).all(axis=1)
        cut = frame.index.max() - pd.Timedelta(days=calib_days)
        fit_m = ok & (frame.index <= cut); cal_m = ok & (frame.index > cut)
        for q in QS:
            self.models[q] = lgb.LGBMRegressor(**lgb_params(q)).fit(X.loc[fit_m, self.feats], y[fit_m])
        lo = self.models[0.1].predict(X.loc[cal_m, self.feats]); hi = self.models[0.9].predict(X.loc[cal_m, self.feats])
        score = np.maximum(lo - y[cal_m], y[cal_m] - hi)
        self.cqr = float(np.quantile(score, 0.8 * (1 + 1 / len(score)))) if cal_m.sum() > 50 else 0.0
        return self
    def predict(self, frame, components=False):
        X = self.frame(frame)
        off = X["struct"].values if self.structured else 0
        out = pd.DataFrame({f"q{int(q*100)}": off + self.models[q].predict(X[self.feats]) for q in QS}, index=frame.index)
        out["q10"] -= self.cqr; out["q90"] += self.cqr
        if components:
            if self.structured:
                out["thermal"] = X["thermal"].values; out["behav"] = X["behav"].values; out["base"] = X["base"].values
                out["resid"] = out["q50"] - X["struct"].values
            else:
                out["thermal"] = 0.0; out["behav"] = 0.0; out["base"] = 0.0; out["resid"] = out["q50"]
        return out

train = df[(df.is_train == 1)].dropna(subset=F_LAG)
t0 = time.time()
M_ID = Forecaster("PALIMPSEST (identified)", F_STRUCT, THERMAL_P, BASELINE, BEHAV_ID).fit(train)
M_AB = Forecaster("Ablation (full-series)", F_STRUCT, THERMAL_P_NAIVE, BASELINE_NAIVE, BEHAV_NAIVE).fit(train)
M_FLAG = Forecaster("Flag-as-feature LGBM", F_FLAG, structured=False).fit(train)
M_PLAIN = Forecaster("Plain LGBM", F_BASE, structured=False).fit(train)
print(f"4 forecasters × 3 quantiles trained in {time.time()-t0:.0f}s")

# similar-day (analog) baseline: kNN on the daily temperature profile + calendar
T_piv = df.pivot_table(index="date", columns="hour", values="T").reindex(daily.index)
L_piv = df.pivot_table(index="date", columns="hour", values="load").reindex(daily.index)
def analog_matrix(dates):
    d = daily.loc[dates]
    return np.column_stack([T_piv.loc[dates].values / 5.0, d.daytype.values[:, None] * 10,
                            np.sin(2 * np.pi * d.doy.values / 365)[:, None] * 3, np.cos(2 * np.pi * d.doy.values / 365)[:, None] * 3])
an_train_days = daily[(daily.is_train == 1) & (daily.alert == 0) & (daily.covid == 0)].index
AN_NN = NearestNeighbors(n_neighbors=CFG["KNN_ANALOGS"]).fit(analog_matrix(an_train_days))
def analog_forecast(dates, k=None):
    k = k or CFG["KNN_ANALOGS"]
    dist, ind = AN_NN.kneighbors(analog_matrix(dates), n_neighbors=k)
    preds, spreads = [], []
    for i, d in enumerate(dates):
        nd = an_train_days[ind[i]]
        yf = np.array([BASELINE.yf.get(daily.loc[d, "year"], BASELINE.yf[BASELINE.last]) / BASELINE.yf.get(y, BASELINE.yf[BASELINE.last]) for y in daily.loc[nd, "year"]])
        prof = L_piv.loc[nd].values * yf[:, None]
        preds.append(prof.mean(axis=0)); spreads.append(prof.std(axis=0))
    return np.array(preds), np.array(spreads), dist

# %% [markdown]
# ## 7. Counterfactual engine
#
# **Abduction** — infer the day's residual noise $u = L^{obs} - \hat L(x)$.
# **Action** — change the queried variable (temperature shift, appeal on/off, occupancy regime).
# **Prediction** — propagate through all three layers; report
# $\Delta = \Delta_\text{thermal} + \Delta_\text{behavioral} + \Delta_\text{residual}$.
#
# **Plausibility** — the queried weather state is projected onto the historical joint distribution of
# (temperature, dew point, wind, cloud | hour, season) via kNN; the nearest-neighbour distance is compared to the
# training distribution and flagged as *extrapolation* beyond its 99th percentile.

# %%
class Plausibility:
    def __init__(self, hist, k=20):
        self.cols = ["T", "dew", "wind", "cloud"]
        Z = hist[self.cols + ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]].dropna()
        self.mu = Z[self.cols].mean(); self.sd = Z[self.cols].std()
        self.X = self._embed(Z); self.vals = Z[self.cols].values
        self.nn = NearestNeighbors(n_neighbors=k + 1).fit(self.X)
        sample = np.random.default_rng(0).choice(len(self.X), min(5000, len(self.X)), replace=False)
        d, _ = self.nn.kneighbors(self.X[sample], n_neighbors=2)
        self.thresh = np.quantile(d[:, 1], 0.99)
        self.k = k
    def _embed(self, Z):
        W = (Z[self.cols] - self.mu) / self.sd
        return np.column_stack([W.values, 2 * Z[["hour_sin", "hour_cos", "doy_sin", "doy_cos"]].values])
    def project(self, frame):
        """Return frame with dew/wind/cloud replaced by neighbour means around the queried T; plus NN distance."""
        Z = self._embed(frame)
        d, ind = self.nn.kneighbors(Z, n_neighbors=self.k)
        out = frame.copy()
        nbr = self.vals[ind]                                    # (n, k, 4)
        for j, c in enumerate(self.cols[1:], start=1):
            out[c] = nbr[:, :, j].mean(axis=1)
        out["nn_dist"] = d[:, 0]
        out["extrapolation"] = (d[:, 0] > self.thresh).astype(int)
        return out

PLAUS = Plausibility(df[df.is_train == 1])

class CounterfactualEngine:
    def __init__(self, model, plaus, history, context_days=21):
        self.m, self.plaus, self.hist, self.ctx = model, plaus, history, context_days
    def window(self, date):
        date = pd.Timestamp(date).normalize()
        w = self.hist.loc[date - pd.Timedelta(days=self.ctx): date + pd.Timedelta(hours=23)].copy()
        return w, w.index.normalize() == date
    def run(self, date, dT=0.0, alert=None, covid=None, project=True, dDew=0.0):
        w, today = self.window(date)
        fact = self.m.predict(w, components=True).loc[today]
        obs = w.loc[today, "load"].values
        u = obs - fact["q50"].values                             # abduction
        cf = w.copy()
        cf.loc[today, "T"] = cf.loc[today, "T"] + dT
        cf.loc[today, "dew"] = cf.loc[today, "dew"] + dDew
        info = {"extrapolation": 0, "nn_dist": np.nan}
        if project and (dT != 0 or dDew != 0):
            proj = self.plaus.project(cf.loc[today])
            for c in ["dew", "wind", "cloud"]: cf.loc[today, c] = proj[c].values
            info = {"extrapolation": int(proj["extrapolation"].max()), "nn_dist": float(proj["nn_dist"].mean()),
                    "nn_thresh": float(self.plaus.thresh)}
        if alert is not None: cf.loc[today, "alert"] = int(alert)
        if covid is not None: cf.loc[today, "covid"] = int(covid)
        cfp = self.m.predict(cf, components=True).loc[today]
        out = pd.DataFrame({
            "observed": obs, "factual_q50": fact["q50"].values,
            "cf_q10": cfp["q10"].values + u, "cf_q50": cfp["q50"].values + u, "cf_q90": cfp["q90"].values + u,
            "d_thermal": cfp["thermal"].values - fact["thermal"].values,
            "d_behavioral": cfp["behav"].values - fact["behav"].values,
            "d_residual": cfp["resid"].values - fact["resid"].values,
        }, index=fact.index)
        out["d_total"] = out["d_thermal"] + out["d_behavioral"] + out["d_residual"]
        info.update(query=dict(date=str(pd.Timestamp(date).date()), dT=dT, dDew=dDew, alert=alert, covid=covid),
                    cf_weather=cf.loc[today, ["T", "dew", "wind", "cloud"]])
        return out, info
    def analogs(self, cf_weather, date, k=None):
        """Nearest historical days to the counterfactual weather state (same day-type)."""
        k = k or CFG["KNN_ANALOGS"]
        date = pd.Timestamp(date).normalize()
        d = daily.loc[date]
        q = np.concatenate([cf_weather["T"].values / 5.0, [d.daytype * 10, 3 * np.sin(2 * np.pi * d.doy / 365), 3 * np.cos(2 * np.pi * d.doy / 365)]])
        dist, ind = AN_NN.kneighbors(q[None, :], n_neighbors=k)
        nd = an_train_days[ind[0]]
        yf = np.array([BASELINE.yf.get(d.year, BASELINE.yf[BASELINE.last]) / BASELINE.yf.get(y, BASELINE.yf[BASELINE.last]) for y in daily.loc[nd, "year"]])
        prof = L_piv.loc[nd].values * yf[:, None]
        return pd.DataFrame(prof.T, columns=[str(x.date()) for x in nd], index=range(24)), dist[0]

ENGINE = CounterfactualEngine(M_ID, PLAUS, df)
ENGINE_AB = CounterfactualEngine(M_AB, PLAUS, df)
ENGINE_FLAG = CounterfactualEngine(M_FLAG, PLAUS, df)

# demo: a hot holdout weekday, +3 °C and an appeal
demo_day = daily[(daily.index >= HOLDOUT_START) & (daily.daytype == 0)].Tmax.idxmax()
cf_tab, cf_info = ENGINE.run(demo_day, dT=3.0, alert=1)
print(f"Demo counterfactual on {demo_day.date()}: +3 °C with a Flex Alert | extrapolation={cf_info['extrapolation']}")
print(cf_tab[["observed", "cf_q50", "d_thermal", "d_behavioral", "d_residual", "d_total"]].iloc[14:22].round(0))

# %% [markdown]
# ## 8. Evaluation A — forecast accuracy on held-out 2023–2024 (sanity baseline)

# %%
hold = df[(df.index >= HOLDOUT_START)].dropna(subset=F_LAG)
acc = []
preds = {}
for M in (M_ID, M_AB, M_FLAG, M_PLAIN):
    P = M.predict(hold); preds[M.name] = P
    acc.append(dict(model=M.name, MAPE=mape(hold["load"], P["q50"]),
                    pinball10=pinball(hold["load"], P["q10"], 0.1), pinball50=pinball(hold["load"], P["q50"], 0.5),
                    pinball90=pinball(hold["load"], P["q90"], 0.9),
                    coverage80=100 * np.mean((hold["load"] >= P["q10"]) & (hold["load"] <= P["q90"]))))
acc.append(dict(model="Persistence (lag 168h)", MAPE=mape(hold["load"], hold["lag168"]), pinball10=np.nan, pinball50=pinball(hold["load"], hold["lag168"], 0.5), pinball90=np.nan, coverage80=np.nan))
hd = daily.index[(daily.index >= HOLDOUT_START)]
hd = hd[hd.isin(hold["date"].unique())]
ap, asp, _ = analog_forecast(hd)
an_series = pd.Series(ap.ravel(), index=pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in hd for h in range(24)])).reindex(hold.index)
acc.append(dict(model="Similar-day (kNN analog)", MAPE=mape(hold["load"], an_series), pinball10=np.nan, pinball50=pinball(hold["load"], an_series, 0.5), pinball90=np.nan, coverage80=np.nan))
ACC = pd.DataFrame(acc).set_index("model").round(3)
RESULTS["accuracy_holdout"] = ACC.reset_index().to_dict("records")
print(f"Holdout {hold.index.min().date()} → {hold.index.max().date()}, {len(hold):,} hours")
ACC

# %% [markdown]
# ## 9. Evaluation B — interventional holdout on Flex Alert days (the central test)
#
# For every Flex Alert day after the holdout year (never seen by any layer), we ask each model for the
# counterfactual **"same day, no alert"** and compare the implied alert effect with the observed
# difference-in-differences against matched non-alert days. The ablation fitted its behavioral layer on the full
# series; if it misattributes thermal load as behavioral, its implied effect will be biased on these days.

# %%
ho_alert_days = daily[(daily.alert == 1) & (daily.year > CFG["FLEX_HOLDOUT_AFTER_YEAR"])].index
pool_all = (daily.alert == 0) & (daily.covid == 0) & (daily.is_holiday == 0)
m_ho = match_days(daily, ho_alert_days, pool_all)
E_ho = did_matrix(R_piv, m_ho)
W = CFG["ALERT_WINDOW"]

def implied_effect(engine, d):
    """Model's implied alert effect (%): (forecast with alert − forecast without) / forecast without."""
    with_, _ = engine.run(d, alert=1, project=False); wo, _ = engine.run(d, alert=0, project=False)
    return 100 * ((with_["cf_q50"] - wo["cf_q50"]) / wo["cf_q50"]).values

rows = []
for d in E_ho.index:
    r = dict(date=d, observed=100 * E_ho.loc[d, W].mean())
    for name, eng in (("identified", ENGINE), ("ablation", ENGINE_AB), ("flagfeat", ENGINE_FLAG)):
        r[name] = implied_effect(eng, d)[W].mean()
    r["plain"] = 0.0
    if TRUE:
        r["truth"] = 100 * TRUE["alert_eff"][W].mean()
    rows.append(r)
HO = pd.DataFrame(rows).set_index("date")
models_b = ["identified", "ablation", "flagfeat", "plain"]
summary = []
for mname in models_b:
    err = HO[mname] - HO["observed"]
    lo_, hi_ = bootstrap_ci(np.abs(err.values))
    summary.append(dict(model=mname, n_days=len(HO), bias_pp=err.mean(), MAE_pp=np.abs(err).mean(), MAE_lo=lo_, MAE_hi=hi_,
                        RMSE_pp=np.sqrt((err ** 2).mean())))
# forecast interval coverage on held-out alert days (forecast made with the alert flag on, as an operator would)
for i, (mname, M) in enumerate(zip(models_b, (M_ID, M_AB, M_FLAG, M_PLAIN))):
    sub = df[df["date"].isin(E_ho.index)].dropna(subset=F_LAG)
    P = M.predict(sub); win = sub["hour"].isin(W)
    summary[i]["window_MAPE"] = mape(sub.loc[win, "load"], P.loc[win, "q50"])
    summary[i]["window_coverage80"] = 100 * np.mean((sub.loc[win, "load"] >= P.loc[win, "q10"]) & (sub.loc[win, "load"] <= P.loc[win, "q90"]))
SUMB = pd.DataFrame(summary).set_index("model").round(3)
RESULTS["flex_alert_holdout"] = SUMB.reset_index().to_dict("records")
RESULTS["flex_alert_holdout_days"] = HO.round(3).reset_index().assign(date=lambda x: x.date.astype(str)).to_dict("records")
print(f"{len(HO)} held-out Flex Alert days; observed window effect mean {HO.observed.mean():+.2f} pp "
      f"(80% CI {bootstrap_ci(HO.observed.values)[0]:+.2f} .. {bootstrap_ci(HO.observed.values)[1]:+.2f})")
SUMB

# %%
# ---- CENTRAL FIGURE
labels = {"identified": "PALIMPSEST\n(identified)", "ablation": "Ablation\n(full series)", "flagfeat": "Flag-as-\nfeature", "plain": "Plain\nLGBM"}
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw=dict(width_ratios=[1.1, 1, 1.2]))
# (a) MAE with bootstrap CI
for i, mname in enumerate(models_b):
    r = SUMB.loc[mname]
    ax[0].bar(i, r.MAE_pp, color=PAL[mname], width=0.6)
    ax[0].errorbar(i, r.MAE_pp, yerr=[[r.MAE_pp - r.MAE_lo], [r.MAE_hi - r.MAE_pp]], color=PAL["observed"], capsize=3, linewidth=1)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels([labels[m] for m in models_b], fontsize=8)
ax[0].set_ylabel("|implied − observed alert effect| (pp of load)"); ax[0].set_title("(a) Error on held-out alert days")
# (b) per-day observed vs implied
lim = [min(HO[models_b + ["observed"]].min().min(), -1) - 0.5, max(HO[models_b + ["observed"]].max().max(), 1) + 0.5]
ax[1].plot(lim, lim, color="#bbbbbb", linewidth=1)
for mname in ["identified", "ablation", "flagfeat"]:
    ax[1].scatter(HO["observed"], HO[mname], color=PAL[mname], s=28, label=labels[mname].replace("\n", " "), edgecolor="white", linewidth=0.8)
ax[1].set_xlabel("observed DiD effect (pp)"); ax[1].set_ylabel("model-implied effect (pp)"); ax[1].set_title("(b) Per-day, 4–9 pm window")
ax[1].legend(frameon=False, fontsize=8)
# (c) hourly profile
h = np.arange(24)
lo_h, hi_h = bootstrap_ci(E_ho.values)
ax[2].fill_between(h, 100 * lo_h, 100 * hi_h, color="#dddddd", linewidth=0, label="observed DiD 80% CI")
ax[2].plot(h, 100 * E_ho.mean(axis=0).values, color=PAL["observed"], label="observed DiD (holdout)")
prof = {n: np.mean([implied_effect(e, d) for d in E_ho.index], axis=0) for n, e in (("identified", ENGINE), ("ablation", ENGINE_AB), ("flagfeat", ENGINE_FLAG))}
for n, v in prof.items(): ax[2].plot(h, v, color=PAL[n], label=labels[n].replace("\n", " "))
if TRUE:
    ax[2].plot(h, 100 * TRUE["alert_eff"], color=PAL["truth"], linestyle="--", label="ground truth")
ax[2].axhline(0, color="#999999", linewidth=0.8); ax[2].set_xlabel("hour"); ax[2].set_ylabel("effect (%)"); ax[2].set_title("(c) Hourly appeal effect")
ax[2].legend(frameon=False, fontsize=7, ncol=2)
plt.suptitle(f"Interventional holdout: {len(HO)} Flex Alert days after {CFG['FLEX_HOLDOUT_AFTER_YEAR']}  [{DATA_MODE} data]", y=1.02)
plt.tight_layout(); plt.savefig(OUT / "fig_central_flex_alert_holdout.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 10. Evaluation C — COVID lockdown holdout ("same weather, 2020 occupancy")
#
# The occupancy coefficient was identified on 19 Mar – 30 Apr 2020. Here every model forecasts 1 May – 15 Jun 2020,
# which no layer has seen. The reference "no occupancy term" shows how far a weather-only structural model misses.

# %%
cho = df[covid_ho].dropna(subset=F_LAG)
rows = []
for M in (M_ID, M_AB, M_FLAG, M_PLAIN):
    P = M.predict(cho)
    rows.append(dict(model=M.name, MAPE=mape(cho["load"], P["q50"]), bias_pct=100 * np.mean((P["q50"] - cho["load"]) / cho["load"]),
                     coverage80=100 * np.mean((cho["load"] >= P["q10"]) & (cho["load"] <= P["q90"]))))
# structural-only comparisons (no residual forecaster) isolate the occupancy term
S_id = structural(cho, THERMAL_P, BASELINE, BEHAV_ID); S_no = structural(cho, THERMAL_P, BASELINE, ZERO_BEHAV)
rows.append(dict(model="Structural only, identified occupancy", MAPE=mape(cho["load"], S_id["struct"]), bias_pct=100 * np.mean((S_id["struct"] - cho["load"]) / cho["load"]), coverage80=np.nan))
rows.append(dict(model="Structural only, no occupancy term", MAPE=mape(cho["load"], S_no["struct"]), bias_pct=100 * np.mean((S_no["struct"] - cho["load"]) / cho["load"]), coverage80=np.nan))
SUMC = pd.DataFrame(rows).set_index("model").round(3)
RESULTS["covid_holdout"] = SUMC.reset_index().to_dict("records")

# weekday hourly profile: observed vs counterfactuals
wk = cho[cho.daytype == 0]
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(range(24), wk.groupby("hour")["load"].mean(), color=PAL["observed"], label="observed 2020")
ax.plot(range(24), pd.Series(S_no.loc[wk.index, "struct"].values, index=wk.index).groupby(wk["hour"]).mean(), color=PAL["plain"], label="structural, pre-COVID occupancy")
ax.plot(range(24), pd.Series(S_id.loc[wk.index, "struct"].values, index=wk.index).groupby(wk["hour"]).mean(), color=PAL["identified"], label="structural, identified 2020 occupancy")
ax.plot(range(24), M_ID.predict(wk)["q50"].groupby(wk["hour"]).mean(), color=PAL["identified"], linestyle=":", label="PALIMPSEST full forecast")
ax.set_xlabel("hour"); ax.set_ylabel("MW"); ax.set_title("Held-out lockdown weekdays, 1 May – 15 Jun 2020"); ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "fig_covid_holdout.png"); plt.show()
SUMC

# %% [markdown]
# ## 11. Evaluation D — Low Carbon London randomized ToU trial
#
# ToU households are split in two halves (A, B) by a hash of their ID. The price-response profile is identified
# from the **A vs. Std** randomized contrast (same days ⇒ same weather), then used to produce the counterfactual
# "half B under ToU pricing", which is compared with what half B actually did. If a `Tariffs` file is available the
# effect is estimated per tariff state (High / Normal / Low); otherwise a per-slot average is used.

# %%
def load_lcl_kaggle():
    roots = glob.glob("/kaggle/input/**/informations_households.csv", recursive=True)
    if not roots:
        return None, None
    root = Path(roots[0]).parent
    info = pd.read_csv(roots[0])
    grp = info.set_index("LCLid")["stdorToU"].to_dict()
    def split(hid):
        if grp.get(hid) != "ToU": return "Std"
        return "ToU_A" if int(hashlib.md5(hid.encode()).hexdigest(), 16) % 2 == 0 else "ToU_B"
    blocks = sorted(glob.glob(str(root / "**/hhblock_dataset/**/block_*.csv"), recursive=True))
    if not blocks:
        return None, None
    blocks = blocks[:CFG["LCL_MAX_BLOCKS"]]
    print(f"LCL: reading {len(blocks)} hhblock files from {root}")
    hh_cols = [f"hh_{i}" for i in range(48)]
    acc = []
    for b in blocks:
        d = pd.read_csv(b)
        d["group"] = d["LCLid"].map(split); d["day"] = pd.to_datetime(d["day"])
        g = d.groupby(["day", "group"])[hh_cols].agg(["sum", "count"])
        s = g.xs("sum", axis=1, level=1); c = g.xs("count", axis=1, level=1)
        s["n"] = c["hh_0"]; acc.append(s)
    A = pd.concat(acc).groupby(level=[0, 1]).sum()
    n = A.pop("n")
    lcl = A.div(n, axis=0); lcl["n"] = n
    lcl = lcl.reset_index()
    wx = glob.glob(str(root / "**/weather_daily_darksky.csv"), recursive=True)
    if wx:
        w = pd.read_csv(wx[0]); w["day"] = pd.to_datetime(w["time"]).dt.normalize()
        w["tmean"] = (w["temperatureMax"] + w["temperatureMin"]) / 2
        lcl = lcl.merge(w[["day", "tmean"]].drop_duplicates("day"), on="day", how="left")
    else:
        lcl["tmean"] = np.nan
    lcl = lcl[(lcl.day >= "2013-01-01") & (lcl.day <= "2013-12-31") & (lcl.n >= 20)]
    tariffs = None
    tf = glob.glob("/kaggle/input/**/*ariff*.csv", recursive=True)
    if tf:
        t = pd.read_csv(tf[0]); tcol = [c for c in t.columns if "Date" in c][0]; vcol = [c for c in t.columns if c != tcol][0]
        t["ts"] = pd.to_datetime(t[tcol]); t["day"] = t["ts"].dt.normalize(); t["slot"] = (t["ts"].dt.hour * 2 + t["ts"].dt.minute // 30)
        tariffs = t.pivot_table(index="day", columns="slot", values=vcol, aggfunc="first")
        tariffs.columns = [f"hh_{i}" for i in tariffs.columns]
    return lcl, tariffs

LCL_MODE = "synthetic"
lcl, tariffs = (None, None)
if ON_KAGGLE and not CFG["FORCE_SYNTHETIC"]:
    try:
        lcl, tariffs = load_lcl_kaggle()
        if lcl is not None: LCL_MODE = "real"
    except Exception as e:
        print("LCL load failed:", repr(e))
if lcl is None:
    lcl, tariffs = make_synthetic_lcl()
RESULTS["lcl_mode"] = LCL_MODE
hh_cols = [f"hh_{i}" for i in range(48)]
piv = {g: lcl[lcl.group == g].set_index("day")[hh_cols] for g in ["Std", "ToU_A", "ToU_B"]}
days = piv["Std"].index.intersection(piv["ToU_A"].index).intersection(piv["ToU_B"].index)
Std, A, B = (piv[g].loc[days] for g in ["Std", "ToU_A", "ToU_B"])
print(f"LCL mode = {LCL_MODE}: {len(days)} common days, tariffs {'available' if tariffs is not None else 'not available'}")

# identification on A vs Std (randomized, same days => weather held fixed)
rel_A = (A - Std) / Std
if tariffs is not None:
    Tm = tariffs.reindex(days)[hh_cols].fillna("Normal")
    eff = {s: rel_A.values[(Tm.values == s)].mean() for s in ["High", "Normal", "Low"] if (Tm.values == s).any()}
    # post-High rebound slots: the 7 slots after a High block
    post = np.zeros_like(Tm.values, dtype=bool)
    hi = (Tm.values == "High")
    for j in range(1, 8): post[:, j:] |= hi[:, :-j] & ~hi[:, j:]
    post &= ~hi
    if post.any(): eff["post-High"] = rel_A.values[post].mean()
    pred_rel = np.full(Tm.shape, eff.get("Normal", 0.0))
    for s in ["High", "Low"]:
        if s in eff: pred_rel[Tm.values == s] = eff[s]
    if "post-High" in eff: pred_rel[post] = eff["post-High"]
    print("identified ToU effects by tariff state:", {k: f"{100*v:+.1f}%" for k, v in eff.items()})
    if TRUE.get("lcl_tou_eff"): print("ground truth:", {k: f"{100*v:+.1f}%" for k, v in TRUE["lcl_tou_eff"].items()})
else:
    slot_eff = rel_A.mean(axis=0).values
    pred_rel = np.tile(slot_eff, (len(days), 1))
    print("per-slot ToU effect, peak (16–19h) mean: %+.2f%%" % (100 * slot_eff[32:38].mean()))
B_cf = Std * (1 + pred_rel)                       # counterfactual: half B under ToU pricing
B_naive = Std                                     # no behavioral term: "B behaves like control"
def rel_mae(obs, pred): return 100 * np.nanmean(np.abs((obs.values - pred.values) / obs.values))
SUMD = pd.DataFrame([
    dict(model="Counterfactual with identified ToU response", MAPE_all=rel_mae(B, B_cf), MAPE_peak=rel_mae(B.iloc[:, 32:38], B_cf.iloc[:, 32:38])),
    dict(model="No behavioral term (B ≈ control)", MAPE_all=rel_mae(B, B_naive), MAPE_peak=rel_mae(B.iloc[:, 32:38], B_naive.iloc[:, 32:38])),
]).set_index("model").round(3)
# does the identified price response depend on weather? (it should not, beyond the heating baseline)
if lcl["tmean"].notna().any():
    tm = lcl[lcl.group == "Std"].set_index("day").loc[days, "tmean"]
    terc = pd.qcut(tm, 3, labels=["cold", "mid", "warm"])
    by_t = pd.DataFrame({t: 100 * rel_A.loc[terc.index[terc == t]].mean(axis=0).values for t in ["cold", "mid", "warm"]}, index=range(48))
else:
    by_t = None
RESULTS["lcl"] = SUMD.reset_index().to_dict("records")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
x = np.arange(48) / 2
ax[0].plot(x, 100 * rel_A.mean(axis=0).values, color=PAL["identified"], label="ToU half A vs Std (identification)")
ax[0].plot(x, 100 * ((B - Std) / Std).mean(axis=0).values, color=PAL["observed"], label="ToU half B vs Std (observed holdout)")
ax[0].plot(x, 100 * ((B_cf - Std) / Std).mean(axis=0).values, color=PAL["identified"], linestyle=":", label="half B counterfactual")
ax[0].axhline(0, color="#999999", linewidth=0.8); ax[0].set_xlabel("hour of day"); ax[0].set_ylabel("relative to control (%)")
ax[0].set_title("Low Carbon London ToU response"); ax[0].legend(frameon=False, fontsize=8)
if by_t is not None:
    for i, t in enumerate(["cold", "mid", "warm"]):
        ax[1].plot(x, by_t[t].values, color=PAL["seq"][1 + 2 * i], label=f"{t} tercile")
    ax[1].axhline(0, color="#999999", linewidth=0.8); ax[1].set_xlabel("hour of day"); ax[1].set_title("Price response by temperature tercile"); ax[1].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "fig_lcl.png"); plt.show()
SUMD

# %% [markdown]
# ## 12. Analog agreement — calibration diagnostic for counterfactuals
#
# For each counterfactual we find the $k$ nearest real historical days to the *counterfactual* weather state and
# report their spread. A counterfactual whose prediction sits inside the analog envelope is corroborated by history;
# one far outside it (or flagged as extrapolation) should be read with caution.

# %%
rows = []
sample_days = daily[(daily.index >= HOLDOUT_START) & (daily.daytype == 0)].sample(min(40, int((daily.index >= HOLDOUT_START).sum())), random_state=0).index
for d in sample_days:
    for dT in (-3, 0, 3, 6):
        tab, info = ENGINE.run(d, dT=dT)
        an, dist = ENGINE.analogs(info["cf_weather"], d)
        mu, sd = an.mean(axis=1).values, an.std(axis=1).values
        z = (tab["cf_q50"].values - mu) / np.maximum(sd, 1)
        rows.append(dict(date=d, dT=dT, extrapolation=info["extrapolation"], nn_dist=info["nn_dist"],
                         analog_spread_pct=100 * np.mean(sd / mu), inside_envelope=100 * np.mean(np.abs(z) <= 1.0),
                         analog_dist=dist.mean()))
AN = pd.DataFrame(rows)
SUMA = AN.groupby("dT")[["extrapolation", "nn_dist", "analog_spread_pct", "inside_envelope", "analog_dist"]].mean().round(3)
RESULTS["analog_agreement"] = SUMA.reset_index().to_dict("records")
SUMA

# %% [markdown]
# ## 13. Interactive view
#
# Choose a day, pose a weather (ΔT) and/or appeal counterfactual, and see the projected plausible input, the
# decomposed delta and the analog spread. Uses `ipywidgets` when available (Kaggle editor); otherwise renders one
# static example.

# %%
def explain(date, dT=0.0, alert=None, project=True, show=True):
    tab, info = ENGINE.run(date, dT=dT, alert=alert, project=project)
    an, dist = ENGINE.analogs(info["cf_weather"], date)
    tot = tab[["d_thermal", "d_behavioral", "d_residual"]].sum()
    if show:
        print(f"{pd.Timestamp(date).date()}  |  query: ΔT={dT:+.1f} °C, alert={alert}  |  "
              f"{'EXTRAPOLATION — outside historical weather support' if info['extrapolation'] else 'within historical support'}"
              + (f" (NN dist {info['nn_dist']:.2f} vs threshold {info.get('nn_thresh', float('nan')):.2f})" if project and dT else ""))
        print("Daily energy delta (MWh): thermal %+.0f | behavioral %+.0f | residual (unexplained) %+.0f | total %+.0f"
              % (tot.d_thermal, tot.d_behavioral, tot.d_residual, tot.sum()))
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
        h = np.arange(24)
        ax[0].plot(h, tab["observed"], color=PAL["observed"], label="observed")
        ax[0].fill_between(h, tab["cf_q10"], tab["cf_q90"], color=PAL["identified"], alpha=0.15, linewidth=0)
        ax[0].plot(h, tab["cf_q50"], color=PAL["identified"], label="counterfactual (q10–q90)")
        for i, c in enumerate(an.columns[:CFG["KNN_ANALOGS"]]):
            ax[0].plot(h, an[c], color="#bbbbbb", linewidth=0.7, label="analog days" if i == 0 else None)
        ax[0].set_title("Load (MW)"); ax[0].legend(frameon=False, fontsize=8); ax[0].set_xlabel("hour")
        ax[1].bar(h, tab["d_thermal"], color=PAL["ablation"], label="thermal", width=0.8)
        ax[1].bar(h, tab["d_behavioral"], bottom=np.where(tab["d_behavioral"] * tab["d_thermal"] >= 0, tab["d_thermal"], 0), color=PAL["identified"], label="behavioral", width=0.8)
        ax[1].plot(h, tab["d_total"], color=PAL["observed"], linewidth=1.2, label="total (incl. residual)")
        ax[1].axhline(0, color="#999999", linewidth=0.8); ax[1].set_title("Decomposed delta (MW)"); ax[1].legend(frameon=False, fontsize=8); ax[1].set_xlabel("hour")
        w = info["cf_weather"]
        ax[2].plot(h, ENGINE.hist.loc[w.index, "T"], color=PAL["observed"], label="observed T")
        ax[2].plot(h, w["T"].values, color=PAL["identified"], label="counterfactual T")
        ax[2].plot(h, ENGINE.hist.loc[w.index, "dew"], color=PAL["observed"], linestyle=":", label="observed dew")
        ax[2].plot(h, w["dew"].values, color=PAL["identified"], linestyle=":", label="projected dew")
        ax[2].set_title("Plausible input projection (°C)"); ax[2].legend(frameon=False, fontsize=8); ax[2].set_xlabel("hour")
        plt.tight_layout(); plt.show()
    return tab, info, an

try:
    import ipywidgets as widgets
    from IPython.display import display
    opts = [str(d.date()) for d in daily.index[daily.index >= HOLDOUT_START]]
    ui = widgets.interactive(lambda date, dT, alert, project: explain(date, dT, {"as observed": None, "on": 1, "off": 0}[alert], project),
                             date=widgets.Dropdown(options=opts, value=str(demo_day.date()), description="day"),
                             dT=widgets.FloatSlider(min=-8, max=8, step=0.5, value=3.0, description="ΔT (°C)"),
                             alert=widgets.Dropdown(options=["as observed", "on", "off"], value="on", description="appeal"),
                             project=widgets.Checkbox(value=True, description="project to plausible weather"))
    display(ui)
except Exception as e:
    print("ipywidgets unavailable — static example:", repr(e))
    _ = explain(demo_day, dT=3.0, alert=1)

# %% [markdown]
# ## 14. Summary of all experiments

# %%
print(f"DATA MODE: balancing authority = {DATA_MODE}, Low Carbon London = {LCL_MODE}")
print("\n[A] Forecast accuracy, holdout %s→" % CFG["HOLDOUT_START"]); print(ACC.to_string())
print("\n[B] Flex Alert interventional holdout (%d days) — implied vs observed appeal effect, 4–9 pm window" % len(HO)); print(SUMB.to_string())
print("\n[C] COVID lockdown holdout (1 May – 15 Jun 2020)"); print(SUMC.to_string())
print("\n[D] Low Carbon London: counterfactual for held-out ToU households"); print(SUMD.to_string())
print("\n[E] Analog agreement by ΔT"); print(SUMA.to_string())
if TRUE:
    print("\nSynthetic ground truth — thermal:", TRUE["thermal"])
    print("  window-average true alert effect (%% of load): %+.2f" % HO["truth"].mean())
with open(OUT / "results.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nSaved:", sorted(p.name for p in OUT.iterdir()))
