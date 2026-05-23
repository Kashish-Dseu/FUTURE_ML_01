import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler as SS

DATA_PATH = os.path.dirname(os.path.abspath(__file__))

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Store Sales Forecasting",
    layout="wide",
    initial_sidebar_state="expanded",
)

# COLOR THEME
NAVY      = "#0d1b3e"
NAVY2     = "#1a3272"
NAVY_CARD = "#1a2f5e"
BORDER    = "#2a4070"
BLUE      = "#1d4ed8"
BLUE_L    = "#3b82f6"
ORANGE    = "#f97316"
TEAL      = "#0d9488"
PURPLE    = "#7c3aed"
PURPLE_L  = "#c084fc"
GREEN     = "#10b981"
RED       = "#ef4444"
GOLD      = "#f5c842"
GRAY_TXT  = "#9fb3d8" 
DARK_TXT  = "#111827"
BG_CHART  = "rgba(0,0,0,0)" 
GRID      = "#2a4070"        
CHART_TXT = "#e2e8f0"        

# CSS STYLES
st.markdown(f"""
<style>
/* ── global ── */
html, body, [class*="css"] {{
    font-family: 'Segoe UI', Arial, sans-serif;
}}
.main .block-container {{
    padding: 0.6rem 1.8rem 1rem;
    max-width: 100%;
}}

/* ── sidebar ── */
section[data-testid="stSidebar"] {{
    background: {NAVY} !important;
    min-width: 220px !important;
    max-width: 230px !important;
}}
section[data-testid="stSidebar"] > div {{
    padding-top: 0 !important;
}}
section[data-testid="stSidebar"] * {{
    color: #dce6f7 !important;
}}
section[data-testid="stSidebar"] .stSelectbox > div > div {{
    background: {NAVY_CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 6px;
    color: white !important;
}}
section[data-testid="stSidebar"] input {{
    background: {NAVY_CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 6px;
    color: white !important;
}}
section[data-testid="stSidebar"] .stSlider div[data-baseweb="slider"] > div {{
    background: {BORDER} !important;
}}
section[data-testid="stSidebar"] hr {{
    border-color: {BORDER} !important;
    margin: 8px 0;
}}
section[data-testid="stSidebar"] label {{
    color: #9fb3d8 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
}}

/* ── KPI cards ── */
.kpi-wrap {{
    background: #1a2f5e;
    border-radius: 14px;
    padding: 20px 14px 16px;
    display: flex;
    flex-direction: column;     /* Stacks elements vertically */
    align-items: center;        /* Centers everything horizontally */
    text-align: center;         /* Centers the inner multi-line text blocks */
    gap: 10px;                  /* Clean spacing between the icon and text block */
    box-shadow: 0 2px 10px rgba(0,0,0,0.25);
    height: 100%;
    border-top: 3px solid transparent;
}}
.kpi-icon {{
    width: 48px; height: 48px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; flex-shrink: 0;
}}
.kpi-lbl  {{ font-size:10.5px; font-weight:700; color:#9fb3d8;
             text-transform:uppercase; letter-spacing:.5px; line-height:1.2; }}
.kpi-sub  {{ font-size:9.5px; color:#9ca3af; margin-top:2px; }}
.kpi-val  {{ font-size:23px; font-weight:800; color:white;
             line-height:1.1; margin:6px 0 4px; }} /* Cleaned margins for vertical symmetry */
.delta-up   {{ font-size:11px; color:#10b981; font-weight:600; }}
.delta-down {{ font-size:11px; color:#ef4444; font-weight:600; }}
.delta-flat {{ font-size:11px; color:#9fb3d8; font-weight:600; }}

/* ── section header ── */
.shdr {{
    font-size:12.5px; font-weight:800; color:{GOLD};
    text-transform:uppercase; letter-spacing:.6px;
    border-bottom:2.5px solid {BORDER};
    padding-bottom:5px; margin-bottom:12px;
    margin-top: 10px;
}}

/* ── insights bar ── */
.ins-bar {{
    background: linear-gradient(90deg, {NAVY} 0%, {NAVY2} 100%);
    border-radius: 12px;
    padding: 14px 22px;
    display: flex; align-items: flex-start; gap: 15px;
    margin-top: 15px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.22);
}}
.ins-hdr  {{ color:{GOLD}; font-weight:800; font-size:12px;
             text-transform:uppercase; letter-spacing:1px; margin-bottom:5px; }}
.ins-grid {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:4px 28px; }}
.ins-item {{ color:#dce6f7; font-size:11.5px; line-height:1.75; }}

/* ── misc ── */
#MainMenu {{visibility:hidden;}} footer {{visibility:hidden;}}
.stDeployButton {{display:none;}}
div[data-testid="stVerticalBlock"] > div {{gap:0.7rem;}}
</style>
""", unsafe_allow_html=True)


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_all():
    train_gz    = os.path.join(DATA_PATH, "train.csv.gz")
    train_plain = os.path.join(DATA_PATH, "train.csv")
    if os.path.exists(train_gz):
        train_path = train_gz
    elif os.path.exists(train_plain):
        train_path = train_plain
    else:
        st.error(
            "⚠️ Required data file not found: `train.csv` or `train.csv.gz`\n\n"
            "Please add the training data file to the same folder as `Task1.py`."
        )
        st.stop()

    stores_path  = os.path.join(DATA_PATH, "stores.csv")
    oil_path     = os.path.join(DATA_PATH, "oil.csv")
    hol_path     = os.path.join(DATA_PATH, "holidays_events.csv")

    for p in [stores_path, oil_path, hol_path]:
        if not os.path.exists(p):
            st.error(
                f"⚠️ Required data file not found: `{os.path.basename(p)}`\n\n"
                "Please make sure all required CSV files are present in the same "
                "folder as `Task1.py`:\n"
                "- `train.csv.gz` (or `train.csv`)\n- `stores.csv`\n"
                "- `oil.csv`\n- `holidays_events.csv`"
            )
            st.stop()

    df = pd.read_csv(
        train_path, parse_dates=["date"],
        dtype={"store_nbr": "int8", "onpromotion": "int8", "sales": "float32"},
    )
    stores = pd.read_csv(stores_path)
    oil    = pd.read_csv(oil_path,    parse_dates=["date"])
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()
    hol    = pd.read_csv(hol_path,    parse_dates=["date"])

    holiday_set = set(hol["date"].dt.date)
    df = df.merge(stores, on="store_nbr", how="left")
    df = df.merge(oil,    on="date",      how="left")
    df["dcoilwtico"]  = df["dcoilwtico"].ffill().bfill()
    df["is_holiday"]  = df["date"].dt.date.isin(holiday_set).map(
                            {True: "Holiday", False: "Non-Holiday"})
    df["day_of_week"]  = df["date"].dt.day_name()
    df["week_day_num"] = df["date"].dt.dayofweek
    df["month"]        = df["date"].dt.month
    df["month_name"]   = df["date"].dt.strftime("%b")
    df["year"]         = df["date"].dt.year
    df["quarter"]      = df["date"].dt.quarter
    df["yw"]           = df["date"].dt.to_period("W").dt.start_time
    return df, oil, hol, stores


df_all, oil_raw, hol_raw, stores_meta = load_all()


# ── FORECAST MODEL ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_forecast(df_key_hash: str, _daily: pd.DataFrame, horizon: int = 30):
    d = _daily.sort_values("date").copy()
    d["t"]      = np.arange(len(d))
    d["month"]  = d["date"].dt.month
    d["dow"]    = d["date"].dt.dayofweek
    d["sin_m"]  = np.sin(2 * np.pi * d["month"] / 12)
    d["cos_m"]  = np.cos(2 * np.pi * d["month"] / 12)
    d["sin_d"]  = np.sin(2 * np.pi * d["dow"] / 7)
    d["lag7"]   = d["sales"].shift(7)
    d["lag14"]  = d["sales"].shift(14)
    d["lag30"]  = d["sales"].shift(30)
    d["roll7"]  = d["sales"].shift(1).rolling(7).mean()
    d["roll30"] = d["sales"].shift(1).rolling(30).mean()
    d = d.dropna()

    F = ["t","month","dow","sin_m","cos_m","sin_d",
         "lag7","lag14","lag30","roll7","roll30"]
    model = GradientBoostingRegressor(
        n_estimators=120, max_depth=4,
        learning_rate=0.04, subsample=0.85, random_state=42
    )
    model.fit(d[F], d["sales"])

    hist   = list(d["sales"].values)
    last_d = d["date"].max()
    last_t = int(d["t"].max())
    f_dates, f_vals = [], []

    for i in range(1, horizon + 1):
        fd  = last_d + timedelta(days=i)
        ti  = last_t + i
        mo  = fd.month;  dw = fd.weekday()
        l7  = hist[-7]  if len(hist) >= 7  else hist[-1]
        l14 = hist[-14] if len(hist) >= 14 else hist[-1]
        l30 = hist[-30] if len(hist) >= 30 else hist[-1]
        r7  = np.mean(hist[-7:])
        r30 = np.mean(hist[-30:]) if len(hist) >= 30 else np.mean(hist)
        row = [[ti, mo, dw,
                np.sin(2*np.pi*mo/12), np.cos(2*np.pi*mo/12),
                np.sin(2*np.pi*dw/7), l7, l14, l30, r7, r30]]
        pred = float(np.clip(model.predict(row)[0], 0, None))
        hist.append(pred)
        f_dates.append(fd)
        f_vals.append(pred)

    fc = pd.DataFrame({"date": f_dates, "sales": f_vals})
    return d[["date","sales"]], fc


# ── TODAY'S FORECAST ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⚡ Computing today's forecast…")
def forecast_today(cache_key: str, _df_full: pd.DataFrame, target_date: date):
    today    = pd.Timestamp(target_date)
    t_origin = _df_full["date"].min()
    t_today  = (today - t_origin).days
    mo_t     = today.month
    dw_t     = today.weekday()
    sin_m_t  = np.sin(2*np.pi*mo_t/12); cos_m_t = np.cos(2*np.pi*mo_t/12)
    sin_d_t  = np.sin(2*np.pi*dw_t/7);  cos_d_t = np.cos(2*np.pi*dw_t/7)
    X_today  = np.array([[t_today, sin_m_t, cos_m_t, sin_d_t, cos_d_t]])

    results = []
    for (store, fam), grp in _df_full.groupby(["store_nbr", "family"]):
        daily = (grp.groupby("date")["sales"]
                 .sum().reset_index().sort_values("date"))
        if len(daily) > 180:
            daily = daily.tail(180).reset_index(drop=True)
        if len(daily) < 14:
            continue
        t_idx  = (daily["date"] - t_origin).dt.days.values
        month  = daily["date"].dt.month.values
        dow    = daily["date"].dt.dayofweek.values
        sin_m  = np.sin(2*np.pi*month/12); cos_m = np.cos(2*np.pi*month/12)
        sin_d  = np.sin(2*np.pi*dow/7);    cos_d = np.cos(2*np.pi*dow/7)
        X  = np.column_stack([t_idx, sin_m, cos_m, sin_d, cos_d])
        y  = daily["sales"].values
        sc = SS()
        Xs = sc.fit_transform(X)
        Xt = sc.transform(X_today)
        mdl = Ridge(alpha=10.0)
        mdl.fit(Xs, y)
        pred      = float(np.clip(mdl.predict(Xt)[0], 0, None))
        last7_avg = float(daily["sales"].tail(7).mean())
        vs_last7  = (pred - last7_avg) / (last7_avg + 1e-9) * 100
        meta = _df_full[_df_full["store_nbr"] == store][
                   ["city","state","type"]].iloc[0]
        results.append({
            "store_nbr":    store,
            "family":       fam,
            "city":         meta["city"],
            "state":        meta["state"],
            "store_type":   meta["type"],
            "pred_sales":   pred,
            "last7_avg":    last7_avg,
            "vs_last7_pct": vs_last7,
        })
    return pd.DataFrame(results)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="background:{NAVY_CARD};margin:-1rem -1rem 1.2rem;
                padding:13px 18px;border-bottom:1px solid {BORDER};">
        <span style="color:{GOLD};font-size:13px;margin-right:4px;">▼</span>
        <span style="color:white;font-size:13.5px;font-weight:700;
              text-transform:uppercase;letter-spacing:1px;">FILTERS</span>
    </div>""", unsafe_allow_html=True)

    st.markdown(f'<p style="color:#9fb3d8;font-size:11px;font-weight:600;'
                f'margin-bottom:3px;">Date</p>', unsafe_allow_html=True)
    min_d = df_all["date"].min().date()
    max_d = df_all["date"].max().date()
    dc1, dc2 = st.columns(2)
    with dc1:
        d_from = st.date_input("From", value=min_d, min_value=min_d,
                               max_value=max_d, key="df",
                               label_visibility="collapsed")
    with dc2:
        d_to   = st.date_input("To", value=max_d, min_value=min_d,
                               max_value=max_d, key="dt",
                               label_visibility="collapsed")
    st.markdown("---")
    store_opts = ["All"] + [f"Store {s}" for s in
                            sorted(df_all["store_nbr"].unique())]
    store_sel  = st.selectbox("Store Number", store_opts)
    st.markdown("---")
    fam_opts  = ["All"] + sorted(df_all["family"].unique())
    fam_sel   = st.selectbox("Product Family", fam_opts)
    st.markdown("---")
    hol_sel   = st.selectbox("Holiday", ["All", "Holiday", "Non-Holiday"])
    st.markdown("---")
    city_opts  = ["All"] + sorted(df_all["city"].dropna().unique())
    city_sel   = st.selectbox("City", city_opts)
    st.markdown("---")
    horizon    = st.slider("Forecast Horizon (days)", 7, 60, 30, 7)
    st.markdown(f"""
    <div style="background:{NAVY_CARD};border-radius:8px;padding:11px 12px;
                margin-top:14px;">
      <div style="color:{GOLD};font-weight:700;font-size:11.5px;
                  letter-spacing:.5px;margin-bottom:5px;">ℹ ABOUT DATA</div>
      <div style="color:#9fb3d8;font-size:11px;line-height:1.65;">
        Kaggle <em>Store Sales – Time Series Forecasting</em> competition dataset.<br><br>
        Covers <strong style="color:white">3 years</strong> of daily sales across
        <strong style="color:white">10 stores</strong> and
        <strong style="color:white">10 product families</strong> in Ecuador,
        enriched with oil prices and public holiday flags.
      </div>
    </div>""", unsafe_allow_html=True)


# ── APPLY FILTERS ─────────────────────────────────────────────────────────────
df = df_all.copy()
df = df[(df["date"].dt.date >= d_from) & (df["date"].dt.date <= d_to)]
if store_sel != "All":
    nbr = int(store_sel.replace("Store ", ""))
    df  = df[df["store_nbr"] == nbr]
if fam_sel   != "All": df = df[df["family"]     == fam_sel]
if hol_sel   != "All": df = df[df["is_holiday"] == hol_sel]
if city_sel  != "All": df = df[df["city"]       == city_sel]
daily_sales = df.groupby("date")["sales"].sum().reset_index()


# ── KPI CALCULATIONS ──────────────────────────────────────────────────────────
total_sales  = df["sales"].sum()
avg_daily    = daily_sales["sales"].mean() if len(daily_sales) > 0 else 0
total_stores = df["store_nbr"].nunique()
promo_pct    = (df["onpromotion"].sum() / (len(df) + 1e-9) * 100)
mid = d_from + (d_to - d_from) / 2
cur = df[df["date"].dt.date >= mid]["sales"].sum()
prv = df[df["date"].dt.date <  mid]["sales"].sum()
growth_pct   = (cur - prv) / (prv + 1e-9) * 100

fc_key = f"{d_from}_{d_to}_{store_sel}_{fam_sel}_{hol_sel}_{city_sel}_{horizon}"
hist_fc, fc_df = run_forecast(fc_key, daily_sales, horizon)
fc_total  = fc_df["sales"].sum()
fc_delta  = ((fc_df["sales"].mean() -
              hist_fc["sales"].tail(horizon).mean()) /
             (hist_fc["sales"].tail(horizon).mean() + 1e-9) * 100)


def fmt(v):
    if   v >= 1e9: return f"${v/1e9:.2f}bn"
    elif v >= 1e6: return f"${v/1e6:.2f}M"
    elif v >= 1e3: return f"${v/1e3:.1f}K"
    else:           return f"${v:.0f}"


TODAY     = date.today()
today_str = TODAY.strftime("%A, %d %b %Y")
today_cache_key = f"today_{TODAY.isoformat()}"
today_df = forecast_today(today_cache_key, df_all, TODAY)

if store_sel != "All":
    _nbr     = int(store_sel.replace("Store ", ""))
    today_df = today_df[today_df["store_nbr"] == _nbr]
if fam_sel  != "All":
    today_df = today_df[today_df["family"] == fam_sel]
if city_sel != "All":
    today_df = today_df[today_df["city"] == city_sel]

today_total      = today_df["pred_sales"].sum()    if len(today_df) > 0 else 0
today_vs_last7   = today_df["vs_last7_pct"].mean() if len(today_df) > 0 else 0
today_best_store = int(today_df.groupby("store_nbr")["pred_sales"]
                       .sum().idxmax()) if len(today_df) > 0 else 0
today_best_fam   = (today_df.groupby("family")["pred_sales"]
                    .sum().idxmax()) if len(today_df) > 0 else "N/A"
now_str = datetime.now().strftime("%d %b %Y  %I:%M %p")


# ── HEADER BAR ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(90deg,{NAVY} 0%,{NAVY2} 100%);
            padding:12px 24px;border-radius:10px;margin-bottom:13px;
            display:flex;align-items:center;justify-content:space-between;
            box-shadow:0 4px 14px rgba(0,0,0,0.25);">
  <div style="display:flex;align-items:center;gap:14px;">
    <div style="width:40px;height:40px;background:#1a4baa;border-radius:8px;
                display:flex;align-items:center;justify-content:center;
                font-size:20px;">🛒</div>
    <span style="color:white;font-size:19px;font-weight:800;
                 letter-spacing:1.6px;text-transform:uppercase;">
      STORE SALES FORECASTING DASHBOARD</span>
  </div>
  <div style="display:flex;align-items:center;gap:9px;">
    <span style="font-size:18px;">📅</span>
    <div>
      <div style="color:#9fb3d8;font-size:10px;font-weight:500;">
        Data Last Refreshed:</div>
      <div style="color:white;font-weight:700;font-size:12.5px;">{now_str}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── KPI CARDS ─────────────────────────────────────────────────────────────────
def kpi_card(icon, bg_color, border_color, label, sublabel,
             value, delta_txt, delta_type="up"):
    cls = {"up": "delta-up", "down": "delta-down", "flat": "delta-flat"}[delta_type]
    sub = f'<div class="kpi-sub">{sublabel}</div>' if sublabel else ""
    return f"""
    <div class="kpi-wrap" style="border-top-color:{border_color};">
      <div class="kpi-icon" style="background:{bg_color}; color:{DARK_TXT};">{icon}</div>
      <div>
        <div class="kpi-lbl">{label}</div>{sub}
        <div class="kpi-val">{value}</div>
        <div class="{cls}">{delta_txt}</div>
      </div>
    </div>"""

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.markdown(kpi_card(
        "🛒","#dbeafe",BLUE,"Total Sales",None,
        fmt(total_sales),
        f"↑ {abs(growth_pct):.1f}% vs Last Period" if growth_pct >= 0
        else f"↓ {abs(growth_pct):.1f}% vs Last Period",
        "up" if growth_pct >= 0 else "down"
    ), unsafe_allow_html=True)
with k2:
    st.markdown(kpi_card(
        "📈","#d1fae5",GREEN,"Avg Daily Sales",None,
        fmt(avg_daily),
        f"↑ {abs(growth_pct/2):.1f}% vs Last Period","up"
    ), unsafe_allow_html=True)
with k3:
    st.markdown(kpi_card(
        "🔮","#ede9fe",PURPLE,
        "Forecast Sales",f"(Next {horizon} Days)",
        fmt(fc_total),
        f"↑ {abs(fc_delta):.1f}% vs Prior Period" if fc_delta >= 0
        else f"↓ {abs(fc_delta):.1f}% vs Prior Period",
        "up" if fc_delta >= 0 else "down"
    ), unsafe_allow_html=True)
with k4:
    st.markdown(kpi_card(
        "﹪","#fed7aa",ORANGE,"Growth %",None,
        f"{abs(growth_pct):.1f}%",
        ("↑" if growth_pct >= 0 else "↓") + " vs Last Period",
        "up" if growth_pct >= 0 else "down"
    ), unsafe_allow_html=True)
with k5:
    st.markdown(kpi_card(
        "🏪","#ccfbf1",TEAL,"Total Stores",None,
        str(total_stores),
        f"{promo_pct:.1f}% Items on Promo","flat"
    ), unsafe_allow_html=True)
with k6:
    st.markdown(kpi_card(
        "⚡","#fef9c3",GOLD,"Today's Forecast",TODAY.strftime("%d %b %Y"),
        fmt(today_total),
        (f"↑ {abs(today_vs_last7):.1f}% vs 7-day avg" if today_vs_last7 >= 0
         else f"↓ {abs(today_vs_last7):.1f}% vs 7-day avg"),
        "up" if today_vs_last7 >= 0 else "down"
    ), unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


# ── ROW 2: Trend & Forecast (Wide Vertical Extension) ───────────────────────
st.markdown('<div class="shdr">SALES TREND &amp; FORECAST</div>',
            unsafe_allow_html=True)
hw = (hist_fc.set_index("date")["sales"]
      .resample("W").sum().reset_index())
fw = (fc_df.set_index("date")["sales"]
      .resample("W").sum().reset_index())
fw["lo"] = fw["sales"] * 0.90
fw["hi"] = fw["sales"] * 1.10

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=hw["date"], y=hw["sales"],
    name="Actual Sales", mode="lines",
    line=dict(color=BLUE_L, width=2.2),
    fill="tozeroy", fillcolor="rgba(59,130,246,0.07)",
    hovertemplate="%{x|%b %d %Y}<br>Sales: $%{y:,.0f}<extra></extra>",
))
fig1.add_trace(go.Scatter(
    x=pd.concat([fw["date"], fw["date"][::-1]]),
    y=pd.concat([fw["hi"],   fw["lo"][::-1]]),
    fill="toself", fillcolor="rgba(249,115,22,0.12)",
    line=dict(color="rgba(0,0,0,0)"),
    name="Confidence Band", hoverinfo="skip",
))
fig1.add_trace(go.Scatter(
    x=fw["date"], y=fw["sales"],
    name="Forecast Sales", mode="lines",
    line=dict(color=ORANGE, width=2.5),
    hovertemplate="%{x|%b %d %Y}<br>Forecast: $%{y:,.0f}<extra></extra>",
))
fig1.add_vline(
    x=hw["date"].max(), line_dash="dash",
    line_color="#9ca3af", line_width=1.5,
)
fig1.update_layout(
    height=320, paper_bgcolor=BG_CHART, plot_bgcolor=BG_CHART,
    margin=dict(l=0, r=10, t=6, b=0),
    font=dict(color=CHART_TXT),
    legend=dict(orientation="h", x=0, y=1.12,
                font=dict(size=10.5, color=CHART_TXT), bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(showgrid=False, tickfont=dict(size=10, color=CHART_TXT)),
    yaxis=dict(tickformat="$,.0f", gridcolor=GRID,
               tickfont=dict(size=10, color=CHART_TXT), zeroline=False),
    hovermode="x unified",
)
st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})


# ── ROW 3: Core Analytical Breakdown Grid (2x2 Re-layout) ───────────────────
grid_r1_c1, grid_r1_c2 = st.columns(2)

with grid_r1_c1:
    st.markdown('<div class="shdr">SALES BY STORE (TOP 10)</div>',
                unsafe_allow_html=True)
    sb = (df.groupby("store_nbr")["sales"].sum()
          .sort_values(ascending=False).head(10).reset_index())
    sb["lbl"] = "Store " + sb["store_nbr"].astype(str)
    fig2 = go.Figure(go.Bar(
        y=sb["lbl"][::-1], x=sb["sales"][::-1],
        orientation="h",
        marker=dict(
            color=sb["sales"][::-1].values,
            colorscale=[[0,"#3b82f6"],[1,BLUE_L]],
            showscale=False,
        ),
        text=[fmt(v) for v in sb["sales"][::-1]],
        textposition="outside",
        textfont=dict(size=9.5, color=CHART_TXT),
        hovertemplate="%{y}<br>Sales: $%{x:,.0f}<extra></extra>",
    ))
    fig2.update_layout(
        height=280, paper_bgcolor=BG_CHART, plot_bgcolor=BG_CHART,
        margin=dict(l=0, r=68, t=6, b=0),
        font=dict(color=CHART_TXT),
        xaxis=dict(showgrid=False, showticklabels=False, showline=False, zeroline=False),
        yaxis=dict(tickfont=dict(size=10, color=CHART_TXT), showgrid=False),
        bargap=0.25,
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

with grid_r1_c2:
    st.markdown('<div class="shdr">SALES BY PRODUCT FAMILY</div>',
                unsafe_allow_html=True)
    fam_df = (df.groupby("family")["sales"].sum()
              .sort_values(ascending=False).reset_index())
    fig3 = go.Figure(go.Bar(
        y=fam_df["family"][::-1], x=fam_df["sales"][::-1],
        orientation="h",
        marker=dict(
            color=fam_df["sales"][::-1].values,
            colorscale=[[0,"#a78bfa"],[1,PURPLE]],
        ),
        text=[fmt(v) for v in fam_df["sales"][::-1]],
        textposition="outside",
        textfont=dict(size=9, color=CHART_TXT),
        hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>",
    ))
    fig3.update_layout(
        height=280, paper_bgcolor=BG_CHART, plot_bgcolor=BG_CHART,
        margin=dict(l=0, r=72, t=4, b=0),
        font=dict(color=CHART_TXT),
        xaxis=dict(showgrid=False, showticklabels=False, showline=False),
        yaxis=dict(tickfont=dict(size=9.5, color=CHART_TXT), showgrid=False),
        bargap=0.25,
    )
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})


grid_r2_c1, grid_r2_c2 = st.columns(2)

with grid_r2_c1:
    st.markdown('<div class="shdr">SALES BY DAY OF WEEK</div>',
                unsafe_allow_html=True)
    DOW_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow_df = (df.groupby("day_of_week")["sales"]
              .sum().reindex(DOW_ORDER).reset_index())
    fig4 = go.Figure(go.Bar(
        x=dow_df["day_of_week"], y=dow_df["sales"],
        marker=dict(
            color=dow_df["sales"].values,
            colorscale=[[0,"#5eead4"],[1,TEAL]],
        ),
        text=[fmt(v) if pd.notna(v) else "" for v in dow_df["sales"]],
        textposition="outside",
        textfont=dict(size=9, color=CHART_TXT),
        hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig4.update_layout(
        height=280, paper_bgcolor=BG_CHART, plot_bgcolor=BG_CHART,
        margin=dict(l=0, r=10, t=4, b=0),
        font=dict(color=CHART_TXT),
        xaxis=dict(tickangle=0, tickfont=dict(size=10, color=CHART_TXT), showgrid=False),
        yaxis=dict(tickformat="$,.0f", gridcolor=GRID,
                   tickfont=dict(size=9, color=CHART_TXT), showline=False),
        bargap=0.3,
    )
    st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

with grid_r2_c2:
    st.markdown('<div class="shdr">SALES VS OIL PRICE</div>',
                unsafe_allow_html=True)
    oil_join = (daily_sales
                .merge(oil_raw[["date","dcoilwtico"]], on="date", how="left")
                .set_index("date")
                .resample("W")
                .agg({"sales":"sum","dcoilwtico":"mean"})
                .dropna()
                .reset_index())
    fig6 = make_subplots(specs=[[{"secondary_y": True}]])
    fig6.add_trace(go.Scatter(
        x=oil_join["date"], y=oil_join["sales"],
        name="Sales", mode="lines",
        line=dict(color=BLUE_L, width=2.1),
        hovertemplate="%{x|%b %Y}<br>Sales: $%{y:,.0f}<extra></extra>",
    ), secondary_y=False)
    fig6.add_trace(go.Scatter(
        x=oil_join["date"], y=oil_join["dcoilwtico"],
        name="Oil (WTI)", mode="lines",
        line=dict(color=ORANGE, width=2.1),
        hovertemplate="%{x|%b %Y}<br>Oil: $%{y:.2f}<extra></extra>",
    ), secondary_y=True)
    fig6.update_layout(
        height=280, paper_bgcolor=BG_CHART, plot_bgcolor=BG_CHART,
        margin=dict(l=0, r=10, t=20, b=0),
        font=dict(color=CHART_TXT),
        legend=dict(orientation="h", x=0, y=1.14,
                    font=dict(size=10, color=CHART_TXT), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, tickfont=dict(size=9.5, color=CHART_TXT)),
        hovermode="x unified",
    )
    fig6.update_yaxes(tickformat="$,.0f", gridcolor=GRID,
                      tickfont=dict(size=9, color=CHART_TXT), secondary_y=False)
    fig6.update_yaxes(tickprefix="$", tickformat=".0f",
                      tickfont=dict(size=9, color=CHART_TXT),
                      gridcolor="rgba(0,0,0,0)", secondary_y=True)
    st.plotly_chart(fig6, use_container_width=True, config={"displayModeBar": False})


# ── ROW 4: Context Dynamics (Holiday Share & Heatmap Side-by-Side) ───────────
context_c1, context_c2 = st.columns([1.1, 1.9])

with context_c1:
    st.markdown('<div class="shdr">SALES BY HOLIDAY SHARE</div>',
                unsafe_allow_html=True)
    hg        = df.groupby("is_holiday")["sales"].sum().reset_index()
    total_hg  = hg["sales"].sum()
    hol_share = (hg.loc[hg["is_holiday"]=="Holiday","sales"].sum() / (total_hg + 1e-9) * 100)
    fig5 = go.Figure(go.Pie(
        labels=hg["is_holiday"], values=hg["sales"],
        hole=0.58,
        marker=dict(colors=[ORANGE, BLUE_L], line=dict(color=NAVY_CARD, width=2)),
        textinfo="percent+label",
        textfont=dict(size=10.5, color=CHART_TXT),
        insidetextorientation="radial",
        hovertemplate="%{label}<br>$%{value:,.0f} (%{percent})<extra></extra>",
    ))
    fig5.update_layout(
        height=240, paper_bgcolor=BG_CHART,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        font=dict(color=CHART_TXT),
        annotations=[dict(
            text=f"<b>{hol_share:.0f}%</b><br>Holiday",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color=CHART_TXT),
        )],
    )
    st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})

with context_c2:
    st.markdown('<div class="shdr">MONTHLY SALES HEATMAP</div>',
                unsafe_allow_html=True)
    hm       = df.groupby(["year","month"])["sales"].sum().reset_index()
    hm_pivot = hm.pivot(index="year", columns="month", values="sales").fillna(0)
    months_lbl = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    col_labels  = [months_lbl[m-1] for m in hm_pivot.columns]
    fig7 = go.Figure(go.Heatmap(
        z=hm_pivot.values, x=col_labels,
        y=[str(y) for y in hm_pivot.index],
        colorscale=[[0,NAVY_CARD],[0.5,BLUE_L],[1,GOLD]],
        text=[[fmt(v) for v in row] for row in hm_pivot.values],
        texttemplate="%{text}", textfont=dict(size=9.5, color=CHART_TXT),
        hovertemplate="Year %{y}, %{x}<br>Sales: $%{z:,.0f}<extra></extra>",
        showscale=False,
    ))
    fig7.update_layout(
        height=240, paper_bgcolor=BG_CHART, plot_bgcolor=BG_CHART,
        margin=dict(l=0, r=0, t=10, b=0),
        font=dict(color=CHART_TXT),
        xaxis=dict(tickfont=dict(size=10, color=CHART_TXT), side="bottom"),
        yaxis=dict(tickfont=dict(size=10, color=CHART_TXT)),
    )
    st.plotly_chart(fig7, use_container_width=True, config={"displayModeBar": False})


# ── ROW 5: Operational Performance (Promo Impact & Store Type Structure) ────
op_c1, op_c2 = st.columns(2)

with op_c1:
    st.markdown('<div class="shdr">PROMOTION IMPACT BY PRODUCT FAMILY</div>',
                unsafe_allow_html=True)
    promo_df = (df.groupby(["family","onpromotion"])["sales"]
                .mean().reset_index())
    promo_df["promo_label"] = promo_df["onpromotion"].map({0: "No Promo", 1: "On Promo"})
    
    # Sort by total average sales to keep the horizontal chart structured cleanly
    family_order = promo_df.groupby("family")["sales"].sum().sort_values(ascending=True).index
    
    fig8 = go.Figure()
    for lbl, color in [("No Promo", "#9ca3af"), ("On Promo", GREEN)]:
        sub = promo_df[promo_df["promo_label"] == lbl]
        # Align data to the unified sorted family order
        sub = sub.set_index("family").reindex(family_order).reset_index()
        
        fig8.add_trace(go.Bar(
            name=lbl, 
            y=sub["family"],   # Family on Y-axis
            x=sub["sales"],    # Sales on X-axis
            orientation="h",   # Horizontal bars
            marker_color=color,
            text=[fmt(v) if v > 0 else "" for v in sub["sales"]],
            textposition="outside",
            textfont=dict(size=8.5, color=CHART_TXT),
        ))
    fig8.update_layout(
        height=380, paper_bgcolor=BG_CHART, plot_bgcolor=BG_CHART,
        margin=dict(l=10, r=60, t=4, b=20), # Rebalanced margins for left text reading space
        font=dict(color=CHART_TXT),
        barmode="group", bargap=0.2, bargroupgap=0.05,
        legend=dict(orientation="h", x=0, y=1.12,
                    font=dict(size=10, color=CHART_TXT), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, showticklabels=False, showline=False, zeroline=False), # Clean up X grid lines
        yaxis=dict(tickfont=dict(size=10, color=CHART_TXT), showgrid=False),
    )
    st.plotly_chart(fig8, use_container_width=True, config={"displayModeBar": False})

with op_c2:
    st.markdown('<div class="shdr">SALES BY STORE TYPE &amp; CITY</div>',
                unsafe_allow_html=True)
    sc = (df.groupby(["city","type"])["sales"].sum()
          .reset_index().sort_values("sales", ascending=False))
    fig9 = px.bar(
        sc, x="city", y="sales", color="type",
        color_discrete_sequence=[BLUE_L, TEAL, PURPLE, ORANGE],
        labels={"sales":"Sales","city":"City","type":"Store Type"},
    )
    fig9.update_traces(
        texttemplate=None,
        hovertemplate="%{x} (Type %{legendgroup})<br>$%{y:,.0f}<extra></extra>",
    )
    fig9.update_layout(
        height=340, paper_bgcolor=BG_CHART, plot_bgcolor=BG_CHART,
        # CRITICAL FIX: b=100 allocates 100 pixels of crisp, empty space at the bottom 
        # for the tilted labels so they never run outside the container or clip.
        margin=dict(l=40, r=10, t=10, b=100), 
        font=dict(color=CHART_TXT),
        legend=dict(title="Type", orientation="h", x=0, y=1.15,
                    font=dict(size=10, color=CHART_TXT), bgcolor="rgba(0,0,0,0)"),
        # Elegant -45 degree angle combined with clean text size prevents overlap
        xaxis=dict(
            tickangle=-45, 
            tickfont=dict(size=10, color=CHART_TXT), 
            showgrid=False
        ),
        yaxis=dict(tickformat="$,.0f", gridcolor=GRID, tickfont=dict(size=9, color=CHART_TXT)),
        bargap=0.25,
    )
    # Ensure it expands beautifully to fill its Streamlit column structure
    st.plotly_chart(fig9, use_container_width=True, config={"displayModeBar": False})


# ── TODAY'S SALES FORECAST SECTION ───────────────────────────────────────────
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
today_dir = "📈" if today_vs_last7 >= 0 else "📉"
today_cls = GREEN if today_vs_last7 >= 0 else RED
vs_label  = (f"↑ {abs(today_vs_last7):.1f}% vs 7-day avg" if today_vs_last7 >= 0 else f"↓ {abs(today_vs_last7):.1f}% vs 7-day avg")

st.markdown(f"""
<div style="background:linear-gradient(135deg,#1e3a5f 0%,#0d3b5e 50%,#163060 100%);
            border-radius:14px;padding:18px 28px;margin-bottom:15px;
            border:1.5px solid {GOLD};
            box-shadow:0 6px 20px rgba(0,0,0,0.3); ">
  <div style="display:flex;align-items:center;justify-content:space-between;
              flex-wrap:wrap;gap:14px;">
    <div style="display:flex;align-items:center;gap:14px;">
      <div style="width:52px;height:52px;border-radius:50%;
                  background:linear-gradient(135deg,{GOLD},{ORANGE});
                  display:flex;align-items:center;justify-content:center;
                  font-size:24px;box-shadow:0 0 18px rgba(245,200,66,.45); color:{DARK_TXT}">⚡</div>
      <div>
        <div style="color:{GOLD};font-size:11px;font-weight:700;
                    text-transform:uppercase;letter-spacing:1px;">
          Today's Predicted Sales</div>
        <div style="color:#9fb3d8;font-size:11px;">{today_str}</div>
      </div>
    </div>
    <div style="text-align:center;">
      <div style="color:white;font-size:36px;font-weight:900;
                  line-height:1;letter-spacing:-1px;">{fmt(today_total)}</div>
      <div style="color:{today_cls};font-size:12px;font-weight:700;
                  margin-top:4px;">{today_dir} {vs_label}</div>
    </div>
    <div style="display:flex;gap:32px;">
      <div style="text-align:center;">
        <div style="color:#9fb3d8;font-size:10px;text-transform:uppercase;
                    letter-spacing:.5px;">Top Store</div>
        <div style="color:white;font-size:18px;font-weight:800;">
          Store {today_best_store}</div>
      </div>
      <div style="text-align:center;">
        <div style="color:#9fb3d8;font-size:10px;text-transform:uppercase;
                    letter-spacing:.5px;">Top Family</div>
        <div style="color:white;font-size:18px;font-weight:800;">
          {today_best_fam}</div>
      </div>
      <div style="text-align:center;">
        <div style="color:#9fb3d8;font-size:10px;text-transform:uppercase;
                    letter-spacing:.5px;">Stores Active</div>
        <div style="color:white;font-size:18px;font-weight:800;">
          {today_df["store_nbr"].nunique() if len(today_df) > 0 else 0}</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Spreading Today's charts horizontally into balanced pairs or full vertical paths
t_row1_c1, t_row1_c2 = st.columns(2)

with t_row1_c1:
    st.markdown('<div class="shdr">TODAY — BY STORE</div>', unsafe_allow_html=True)
    if len(today_df) > 0:
        ts = (today_df.groupby("store_nbr")["pred_sales"]
              .sum().sort_values(ascending=True).reset_index())
        ts["label"] = "Store " + ts["store_nbr"].astype(str)
        ts["color"] = [GOLD if s == today_best_store else BLUE_L for s in ts["store_nbr"]]
        fig_ta = go.Figure(go.Bar(
            y=ts["label"], x=ts["pred_sales"],
            orientation="h", marker_color=ts["color"],
            text=[fmt(v) for v in ts["pred_sales"]],
            textposition="outside", textfont=dict(size=9.5, color=CHART_TXT),
            hovertemplate="Store %{y}<br>Today Forecast: $%{x:,.0f}<extra></extra>",
        ))
        fig_ta.update_layout(
            height=280, paper_bgcolor=BG_CHART, plot_bgcolor=BG_CHART,
            margin=dict(l=0, r=70, t=4, b=0), font=dict(color=CHART_TXT),
            xaxis=dict(showgrid=False, showticklabels=False, showline=False),
            yaxis=dict(tickfont=dict(size=10, color=CHART_TXT), showgrid=False),
            bargap=0.25,
        )
        st.plotly_chart(fig_ta, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No data for selected filters.")

with t_row1_c2:
    st.markdown('<div class="shdr">TODAY — BY FAMILY</div>', unsafe_allow_html=True)
    if len(today_df) > 0:
        tf = (today_df.groupby("family")["pred_sales"]
              .sum().sort_values(ascending=True).reset_index())
        tf["color"] = [GOLD if f == today_best_fam else PURPLE for f in tf["family"]]
        fig_tb = go.Figure(go.Bar(
            y=tf["family"], x=tf["pred_sales"],
            orientation="h", marker_color=tf["color"],
            text=[fmt(v) for v in tf["pred_sales"]],
            textposition="outside", textfont=dict(size=9.5, color=CHART_TXT),
            hovertemplate="%{y}<br>Today Forecast: $%{x:,.0f}<extra></extra>",
        ))
        fig_tb.update_layout(
            height=280, paper_bgcolor=BG_CHART, plot_bgcolor=BG_CHART,
            margin=dict(l=0, r=70, t=4, b=0), font=dict(color=CHART_TXT),
            xaxis=dict(showgrid=False, showticklabels=False, showline=False),
            yaxis=dict(tickfont=dict(size=10, color=CHART_TXT), showgrid=False),
            bargap=0.25,
        )
        st.plotly_chart(fig_tb, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No data for selected filters.")


st.markdown('<div class="shdr">TODAY — STORE PERFORMANCE vs 7-DAY AVG</div>',
            unsafe_allow_html=True)
if len(today_df) > 0:
    perf = (today_df.groupby("store_nbr")
            .agg(pred_sales=("pred_sales","sum"),
                 vs_last7_pct=("vs_last7_pct","mean"),
                 city=("city","first"),
                 store_type=("store_type","first"))
            .reset_index()
            .sort_values("pred_sales", ascending=False))
    fig_tc = go.Figure()
    fig_tc.add_trace(go.Bar(
        name="Today's Forecast",
        x=["Store "+str(s) for s in perf["store_nbr"]],
        y=perf["pred_sales"],
        marker_color=[GREEN if v >= 0 else RED for v in perf["vs_last7_pct"]],
        text=[f"{v:+.1f}%" for v in perf["vs_last7_pct"]],
        textposition="outside",
        textfont=dict(size=9.5, color=[GREEN if v >= 0 else RED for v in perf["vs_last7_pct"]]),
        hovertemplate=(
            "%{x}<br>"
            "Forecast: $%{y:,.0f}<br>"
            "vs 7-day avg: %{text}<extra></extra>"
        ),
    ))
    n_stores = len(perf)
    tick_size = 9 if n_stores > 30 else 10
    chart_height = max(300, 260 + max(0, n_stores - 20) * 3)
    fig_tc.update_layout(
        height=chart_height, paper_bgcolor=BG_CHART, plot_bgcolor=BG_CHART,
        margin=dict(l=0, r=10, t=4, b=max(80, 60 + n_stores)),
        font=dict(color=CHART_TXT),
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=tick_size, color=CHART_TXT),
            showgrid=False,
            automargin=True,
        ),
        yaxis=dict(tickformat="$,.0f", gridcolor=GRID, tickfont=dict(size=9, color=CHART_TXT)),
        showlegend=False, bargap=0.35,
    )
    st.plotly_chart(fig_tc, use_container_width=True, config={"displayModeBar": False})

    perf_show = perf[["store_nbr","city","store_type","pred_sales","vs_last7_pct"]].copy()
    perf_show.columns = ["Store","City","Type","Today Forecast","vs 7-Day Avg %"]
    perf_show["Today Forecast"] = perf_show["Today Forecast"].apply(fmt)
    perf_show["vs 7-Day Avg %"] = perf_show["vs 7-Day Avg %"].apply(
        lambda x: f"↑ {x:.1f}%" if x >= 0 else f"↓ {abs(x):.1f}%")
    perf_show["Store"] = "Store " + perf_show["Store"].astype(str)
    st.dataframe(perf_show, use_container_width=True, hide_index=True)
else:
    st.info("No data for selected filters.")

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


# ── INSIGHTS BAR ──────────────────────────────────────────────────────────────
best_fam   = (df.groupby("family")["sales"].sum().idxmax() if len(df) > 0 else "N/A")
best_store = ("Store " + str(int(df.groupby("store_nbr")["sales"].sum().idxmax()))) if len(df) > 0 else "N/A"
hol_sales  = df[df["is_holiday"]=="Holiday"]["sales"].sum()
nhol_sales = df[df["is_holiday"]=="Non-Holiday"]["sales"].sum()
hol_uplift = (hol_sales / (hol_sales + nhol_sales + 1e-9)) * 100
promo_uplift = (df[df["onpromotion"]==1]["sales"].mean() / (df[df["onpromotion"]==0]["sales"].mean() + 1e-9) - 1) * 100

st.markdown(f"""
<div class="ins-bar">
  <div style="font-size:26px;flex-shrink:0;padding-top:2px;">💡</div>
  <div style="flex:1;">
    <div class="ins-hdr">INSIGHTS</div>
    <div class="ins-grid">
      <div class="ins-item">
        • Sales show strong seasonality — peak periods around mid-year
          and year-end. Best family:
          <strong style="color:{GOLD}">{best_fam}</strong>.
      </div>
      <div class="ins-item">
        • Forecast of <strong style="color:{GOLD}">{fmt(fc_total)}</strong>
          over the next {horizon} days
          {"— growth expected." if fc_delta >= 0 else "— slight decline expected."}
          Top performer: <strong style="color:{GOLD}">{best_store}</strong>.
      </div>
      <div class="ins-item">
        • Holidays account for ~<strong style="color:{GOLD}">{hol_uplift:.0f}%</strong>
          of sales. Promotions boost avg daily sales by
          <strong style="color:{GOLD}">{abs(promo_uplift):.0f}%</strong>.
          Plan inventory and staffing accordingly.
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
