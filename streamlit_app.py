"""
CurveTrader — Rates & FX Trading Dashboard
------------------------------------------------
Pulls live government bond yield curves and G10 FX rates, screens for
carry trade opportunities, backtests a simple systematic carry strategy,
and flags yield-curve steepener/flattener signals based on historical
z-scores.

Run locally with: streamlit run streamlit_app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(page_title="CurveTrader", page_icon="📉", layout="wide")

st.markdown("""
    <style>
        .main { background: linear-gradient(160deg, #0a0e1a 0%, #0e1117 100%); }
        h1, h2, h3 { color: #e8eaed; letter-spacing: -0.3px; }
        h2, h3 { color: #60a5fa; }
        [data-testid="stHeaderActionElements"] { display: none; }
        [data-testid="stMetricValue"] { color: #e8eaed; }
        [data-testid="stMetric"] {
            background: linear-gradient(160deg, #171b24 0%, #131722 100%);
            border: 1px solid #2b2f38; border-radius: 10px; padding: 12px 16px;
        }
        .stButton>button {
            background: linear-gradient(135deg, #2563eb, #3b82f6);
            color: white; border: none; border-radius: 8px; font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📉 CurveTrader")
st.caption("Rates & FX dashboard — live yield curves, G10 carry screening, a systematic carry backtest, and curve steepener/flattener signals.")

FX_PAIRS = {
    "EUR": "EURUSD=X", "GBP": "GBPUSD=X", "JPY": "USDJPY=X", "CAD": "USDCAD=X",
    "AUD": "AUDUSD=X", "NZD": "NZDUSD=X", "CHF": "USDCHF=X", "SEK": "USDSEK=X", "NOK": "USDNOK=X",
}
# True if the pair is quoted as USD per 1 unit of foreign currency (affects carry-direction math)
FX_QUOTED_AS_USD_PER_UNIT = {"EUR": True, "GBP": True, "JPY": False, "CAD": False,
                             "AUD": True, "NZD": True, "CHF": False, "SEK": False, "NOK": False}

POLICY_RATE_SERIES = {
    "USD": "DFF", "EUR": "ECBDFR", "GBP": "IRSTCB01GBM156N", "JPY": "IRSTCB01JPM156N",
    "CAD": "IRSTCB01CAM156N", "AUD": "IRSTCB01AUM156N", "NZD": "IRSTCB01NZM156N",
    "CHF": "IRSTCB01CHM156N", "SEK": "IRSTCB01SEM156N", "NOK": "IRSTCB01NOM156N",
}

UST_SERIES = {
    "1M": "DGS1MO", "3M": "DGS3MO", "6M": "DGS6MO", "1Y": "DGS1", "2Y": "DGS2",
    "3Y": "DGS3", "5Y": "DGS5", "7Y": "DGS7", "10Y": "DGS10", "20Y": "DGS20", "30Y": "DGS30",
}


# =======================================================================
# 1. DATA FETCH
# =======================================================================
@st.cache_data(ttl=60 * 60 * 6)
def fetch_us_treasury_curve():
    import pandas_datareader.data as web
    end = date.today()
    start = end - timedelta(days=30)
    curve = {}
    for label, code in UST_SERIES.items():
        try:
            d = web.DataReader(code, "fred", start, end).dropna()
            if not d.empty:
                curve[label] = float(d.iloc[-1].iloc[0])
        except Exception:
            continue
    return curve


@st.cache_data(ttl=60 * 60 * 6)
def fetch_us_treasury_series(tenor_code, years_back=2):
    import pandas_datareader.data as web
    end = date.today()
    start = end - timedelta(days=365 * years_back)
    try:
        d = web.DataReader(tenor_code, "fred", start, end).dropna()
        return d
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60 * 60 * 6)
def fetch_policy_rates():
    import pandas_datareader.data as web
    end = date.today()
    start = end - timedelta(days=800)
    rates = {}
    for ccy, code in POLICY_RATE_SERIES.items():
        try:
            d = web.DataReader(code, "fred", start, end).dropna()
            if not d.empty:
                rates[ccy] = float(d.iloc[-1].iloc[0])
        except Exception:
            continue
    return rates


@st.cache_data(ttl=60 * 15)
def fetch_fx_spot():
    spots = {}
    for ccy, ticker in FX_PAIRS.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                spots[ccy] = float(hist["Close"].iloc[-1])
        except Exception:
            continue
    return spots


@st.cache_data(ttl=60 * 60 * 6)
def fetch_fx_history(ticker, period="3y"):
    try:
        hist = yf.Ticker(ticker).history(period=period)
        return hist["Close"]
    except Exception:
        return pd.Series(dtype=float)


# =======================================================================
# 2. CARRY SCREEN
# =======================================================================
def build_carry_table(policy_rates, fx_spots):
    usd_rate = policy_rates.get("USD")
    if usd_rate is None:
        return pd.DataFrame()

    rows = []
    for ccy in FX_PAIRS:
        foreign_rate = policy_rates.get(ccy)
        spot = fx_spots.get(ccy)
        if foreign_rate is None or spot is None:
            continue
        carry = foreign_rate - usd_rate  # positive = foreign currency pays more than USD
        rows.append({"Currency": ccy, "Policy Rate": foreign_rate, "USD Rate": usd_rate,
                     "Carry (bps)": round(carry * 100, 0), "Spot": spot})
    df = pd.DataFrame(rows).sort_values("Carry (bps)", ascending=False).reset_index(drop=True)
    return df


# =======================================================================
# 3. CARRY BACKTEST (simple: long top-carry currencies, short bottom-carry, monthly rebalance)
# =======================================================================
def run_carry_backtest(policy_rates, top_n=3, years=3):
    """
    Simplified systematic carry backtest: each month, go long the top_n
    highest-carry currencies and short the top_n lowest-carry currencies
    (equal-weighted vs USD), holding for that month. Return = spot return
    + carry accrual (using CURRENT policy rate differential as a proxy
    throughout the backtest window — a simplification, since historical
    daily policy rates for all 9 currencies aren't cleanly available from
    a single free source).
    """
    usd_rate = policy_rates.get("USD", 0)
    carry_by_ccy = {ccy: (policy_rates.get(ccy, 0) - usd_rate) for ccy in FX_PAIRS if ccy in policy_rates}
    if len(carry_by_ccy) < top_n * 2:
        return None

    ranked = sorted(carry_by_ccy.items(), key=lambda x: x[1], reverse=True)
    longs = [c for c, _ in ranked[:top_n]]
    shorts = [c for c, _ in ranked[-top_n:]]

    price_data = {}
    for ccy in set(longs + shorts):
        hist = fetch_fx_history(FX_PAIRS[ccy], period=f"{years}y")
        if not hist.empty:
            price_data[ccy] = hist

    if not price_data:
        return None

    common_index = None
    for s in price_data.values():
        common_index = s.index if common_index is None else common_index.intersection(s.index)
    if common_index is None or len(common_index) < 30:
        return None

    daily_carry = {ccy: carry_by_ccy[ccy] / 252 for ccy in price_data}
    portfolio_daily_returns = pd.Series(0.0, index=common_index)

    for ccy in longs:
        if ccy not in price_data:
            continue
        px = price_data[ccy].reindex(common_index).ffill()
        fx_ret = px.pct_change().fillna(0)
        if not FX_QUOTED_AS_USD_PER_UNIT[ccy]:
            fx_ret = -fx_ret  # invert so positive = foreign currency strengthening vs USD
        portfolio_daily_returns += (fx_ret + daily_carry[ccy]) / top_n

    for ccy in shorts:
        if ccy not in price_data:
            continue
        px = price_data[ccy].reindex(common_index).ffill()
        fx_ret = px.pct_change().fillna(0)
        if not FX_QUOTED_AS_USD_PER_UNIT[ccy]:
            fx_ret = -fx_ret
        portfolio_daily_returns += (-fx_ret - daily_carry[ccy]) / top_n

    cumulative = (1 + portfolio_daily_returns).cumprod() * 100
    ann_return = (1 + portfolio_daily_returns.mean()) ** 252 - 1
    ann_vol = portfolio_daily_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else None

    return {
        "cumulative": cumulative, "longs": longs, "shorts": shorts,
        "ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe,
    }


# =======================================================================
# 4. CURVE STEEPNESS + SIGNAL
# =======================================================================
def curve_steepness_signal(lookback_years=5):
    two_y = fetch_us_treasury_series("DGS2", lookback_years)
    ten_y = fetch_us_treasury_series("DGS10", lookback_years)
    if two_y.empty or ten_y.empty:
        return None

    merged = pd.concat([two_y, ten_y], axis=1, join="inner").dropna()
    merged.columns = ["DGS2", "DGS10"]
    spread = merged["DGS10"] - merged["DGS2"]

    current_spread = spread.iloc[-1]
    mean_spread = spread.mean()
    std_spread = spread.std()
    z_score = (current_spread - mean_spread) / std_spread if std_spread > 0 else 0

    if z_score < -1.0:
        signal = "Curve unusually flat/inverted vs history — potential steepener trade (bet the curve normalizes)"
    elif z_score > 1.0:
        signal = "Curve unusually steep vs history — potential flattener trade (bet the curve compresses)"
    else:
        signal = "Curve within normal historical range — no strong mean-reversion signal"

    return {"spread_series": spread, "current_spread": current_spread, "mean_spread": mean_spread,
            "z_score": z_score, "signal": signal}


# =======================================================================
# 5. CANADIAN GOVERNMENT BOND CURVE (Bank of Canada Valet API)
# =======================================================================
GOC_SERIES = {
    "2Y": "BD.CDN.2YR.DQ.YLD", "3Y": "BD.CDN.3YR.DQ.YLD", "5Y": "BD.CDN.5YR.DQ.YLD",
    "7Y": "BD.CDN.7YR.DQ.YLD", "10Y": "BD.CDN.10YR.DQ.YLD", "Long (30Y)": "BD.CDN.LONG.DQ.YLD",
}


@st.cache_data(ttl=60 * 60 * 6)
def fetch_cad_curve():
    import requests
    curve = {}
    for label, series in GOC_SERIES.items():
        try:
            resp = requests.get(f"https://www.bankofcanada.ca/valet/observations/{series}/json",
                                 params={"recent": 1}, timeout=10)
            resp.raise_for_status()
            obs = resp.json()["observations"][-1]
            curve[label] = float(obs[series]["v"])
        except Exception:
            continue
    return curve


# =======================================================================
# 6. TODAY'S MARKET PULSE (live, for the dynamic Q&A answers)
# =======================================================================
@st.cache_data(ttl=60 * 30)
def fetch_market_pulse():
    try:
        tnx = yf.Ticker("^TNX").history(period="5d")
        spx = yf.Ticker("^GSPC").history(period="5d")
        vix = yf.Ticker("^VIX").history(period="5d")
        tsx = yf.Ticker("^GSPTSE").history(period="5d")
        cad = yf.Ticker("USDCAD=X").history(period="5d")

        def day_change(hist, is_yield=False):
            if len(hist) < 2:
                return None, None
            last, prev = hist["Close"].iloc[-1], hist["Close"].iloc[-2]
            chg = (last - prev) if is_yield else (last / prev - 1)
            return float(last), float(chg)

        ust10_level, ust10_chg = day_change(tnx, is_yield=True)
        spx_level, spx_chg = day_change(spx)
        vix_level, vix_chg = day_change(vix)
        tsx_level, tsx_chg = day_change(tsx)
        cad_level, cad_chg = day_change(cad)

        return {
            "ust10_level": ust10_level / 10 if ust10_level else None,  # ^TNX quoted as yield*10
            "ust10_chg_bps": ust10_chg * 10 if ust10_chg else None,
            "spx_chg_pct": spx_chg * 100 if spx_chg else None,
            "vix_level": vix_level, "vix_chg": vix_chg * 100 if vix_chg else None,
            "tsx_chg_pct": tsx_chg * 100 if tsx_chg else None,
            "cad_level": cad_level, "cad_chg_pct": cad_chg * 100 if cad_chg else None,
            "as_of": tnx.index[-1].strftime("%Y-%m-%d") if not tnx.empty else "unknown",
        }
    except Exception:
        return None


# =======================================================================
# 7. DURATION & CONVEXITY
# =======================================================================
def bond_duration_convexity(face_value, coupon_rate, yield_rate, years_to_maturity, freq=2):
    n = int(years_to_maturity * freq)
    c = coupon_rate * face_value / freq
    y = yield_rate / freq

    price = 0.0
    weighted_time = 0.0
    convexity_sum = 0.0
    for t in range(1, n + 1):
        cf = c + face_value if t == n else c
        pv = cf / (1 + y) ** t
        price += pv
        weighted_time += (t / freq) * pv
        convexity_sum += pv * t * (t + 1)

    macaulay_duration = weighted_time / price
    modified_duration = macaulay_duration / (1 + y)
    convexity = convexity_sum / (price * (1 + y) ** 2 * freq ** 2)

    return {"price": price, "macaulay_duration": macaulay_duration,
            "modified_duration": modified_duration, "convexity": convexity}


def estimate_price_change(modified_duration, convexity, yield_shock_bps):
    dy = yield_shock_bps / 10000
    pct_change = -modified_duration * dy + 0.5 * convexity * dy ** 2
    return pct_change


# =======================================================================
# 8. ROLL-DOWN ANALYSIS
# =======================================================================
def rolldown_analysis(curve, tenor_years_list):
    """For each tenor, estimate 1-year roll-down return: the price gain from
    a bond 'rolling down' the curve to a shorter maturity in 1 year, holding
    the curve shape constant (a standard simplifying assumption)."""
    tenor_map = {2: "2Y", 3: "3Y", 5: "5Y", 7: "7Y", 10: "10Y", 30: "Long (30Y)", 30.0: "Long (30Y)"}
    results = []
    sorted_tenors = sorted(tenor_years_list)
    for tenor in sorted_tenors:
        label = tenor_map.get(tenor)
        if label not in curve:
            continue
        current_yield = curve[label]
        # find the closest shorter tenor for the "rolled to" yield
        shorter_candidates = [t for t in sorted_tenors if t < tenor]
        if not shorter_candidates:
            continue
        rolled_tenor = max(shorter_candidates)
        rolled_label = tenor_map.get(rolled_tenor)
        if rolled_label not in curve:
            continue
        rolled_yield = curve[rolled_label]
        rolldown_bps = (current_yield - rolled_yield) * 100
        # Approximate price gain using modified duration at the rolled tenor
        approx_mod_dur = rolled_tenor * 0.9  # rough approximation for a par bond
        estimated_return = approx_mod_dur * (rolldown_bps / 10000) * 100
        results.append({"Tenor": label, "Yield": current_yield, "Rolls to": rolled_label,
                        "Roll-Down (bps)": round(rolldown_bps, 1), "Est. 1Yr Roll Return (%)": round(estimated_return, 2)})
    return pd.DataFrame(results)


# =======================================================================
# 9. UI
# =======================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Yield Curves", "FX Carry Screener", "Carry Backtest", "Curve Trade Signal",
    "CAD Curve & CAD/USD Spread", "Fixed Income Strategy", "Trading Desk Q&A", "About This Data"
])

# --- TAB 1: YIELD CURVES ---
with tab1:
    st.subheader("U.S. Treasury Yield Curve")
    curve = fetch_us_treasury_curve()
    if not curve:
        st.error("Could not fetch Treasury curve data right now.")
    else:
        tenors = list(curve.keys())
        yields = list(curve.values())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tenors, y=yields, mode="lines+markers",
                                  line=dict(color="#60a5fa", width=3), marker=dict(size=8)))
        fig.update_layout(xaxis_title="Tenor", yaxis_title="Yield (%)",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(color="#e8eaed"))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        if "2Y" in curve and "10Y" in curve:
            c1.metric("2s10s Spread", f"{(curve['10Y']-curve['2Y'])*100:.0f} bps")
        if "5Y" in curve and "30Y" in curve:
            c2.metric("5s30s Spread", f"{(curve['30Y']-curve['5Y'])*100:.0f} bps")
        if "3M" in curve and "10Y" in curve:
            c3.metric("3M10Y Spread", f"{(curve['10Y']-curve['3M'])*100:.0f} bps")

        st.caption("Source: FRED constant-maturity Treasury series, updated daily on business days.")

# --- TAB 2: FX CARRY SCREENER ---
with tab2:
    st.subheader("G10 Carry Screener (vs. USD)")
    st.caption("Carry = foreign policy rate minus USD policy rate. Positive carry means the currency pays more to hold than USD, before spot moves.")

    policy_rates = fetch_policy_rates()
    fx_spots = fetch_fx_spot()
    carry_df = build_carry_table(policy_rates, fx_spots)

    if carry_df.empty:
        st.error("Could not build the carry table right now — try again shortly.")
    else:
        st.dataframe(carry_df, use_container_width=True, hide_index=True)

        fig = go.Figure(data=[go.Bar(x=carry_df["Currency"], y=carry_df["Carry (bps)"],
                                      marker_color=["#4ade80" if v >= 0 else "#f87171" for v in carry_df["Carry (bps)"]])])
        fig.update_layout(yaxis_title="Carry (bps)", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaed"))
        st.plotly_chart(fig, use_container_width=True)

        st.info(f"Highest carry: **{carry_df.iloc[0]['Currency']}** ({carry_df.iloc[0]['Carry (bps)']:.0f} bps) — "
                f"Lowest carry: **{carry_df.iloc[-1]['Currency']}** ({carry_df.iloc[-1]['Carry (bps)']:.0f} bps)")

# --- TAB 3: CARRY BACKTEST ---
with tab3:
    st.subheader("Systematic Carry Trade Backtest")
    st.caption("Goes long the top-N highest-carry currencies and short the bottom-N lowest-carry currencies (equal-weighted vs USD).")

    top_n = st.slider("Number of currencies per side", 1, 4, 3)
    years = st.radio("Lookback period", [1, 3, 5], index=1, horizontal=True)

    if st.button("Run Backtest"):
        with st.spinner("Pulling historical FX data and running backtest..."):
            policy_rates = fetch_policy_rates()
            result = run_carry_backtest(policy_rates, top_n=top_n, years=years)
        st.session_state["_carry_backtest"] = result

    result = st.session_state.get("_carry_backtest")
    if result:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result["cumulative"].index, y=result["cumulative"],
                                  name="Carry Strategy", line=dict(color="#60a5fa", width=2)))
        fig.update_layout(yaxis_title="Growth of $100", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaed"))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Annualized Return", f"{result['ann_return']*100:.2f}%")
        c2.metric("Annualized Volatility", f"{result['ann_vol']*100:.2f}%")
        c3.metric("Sharpe Ratio", f"{result['sharpe']:.2f}" if result["sharpe"] else "N/A")

        st.caption(f"Long: {', '.join(result['longs'])}  |  Short: {', '.join(result['shorts'])}")
        st.warning("This backtest uses CURRENT policy rate differentials applied across the whole historical "
                   "window as a simplification (clean historical daily policy rates for all 9 currencies aren't "
                   "available from a single free source) — treat it as illustrative of the carry-trade mechanic, "
                   "not a precise historical P&L.")

# --- TAB 4: CURVE TRADE SIGNAL ---
with tab4:
    st.subheader("2s10s Curve Steepener / Flattener Signal")
    st.caption("Flags when the current 2s10s spread is unusually wide or narrow relative to its own multi-year history (a simple mean-reversion signal).")

    lookback = st.radio("History window (years)", [2, 5, 10], index=1, horizontal=True)

    if st.button("Generate Signal"):
        with st.spinner("Pulling historical Treasury data..."):
            result = curve_steepness_signal(lookback)
        st.session_state["_curve_signal"] = result

    result = st.session_state.get("_curve_signal")
    if result:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result["spread_series"].index, y=result["spread_series"],
                                  name="2s10s Spread", line=dict(color="#60a5fa", width=2)))
        fig.add_hline(y=result["mean_spread"], line_dash="dash", line_color="#9aa0ab",
                      annotation_text="Historical Mean")
        fig.update_layout(yaxis_title="Spread (%)", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaed"))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Current 2s10s", f"{result['current_spread']*100:.0f} bps")
        c2.metric("Historical Mean", f"{result['mean_spread']*100:.0f} bps")
        c3.metric("Z-Score", f"{result['z_score']:.2f}")

        if result["z_score"] < -1.0:
            st.success(result["signal"])
        elif result["z_score"] > 1.0:
            st.error(result["signal"])
        else:
            st.info(result["signal"])

# --- TAB 5: CAD CURVE & CAD/USD SPREAD ---
with tab5:
    st.subheader("Canadian Government Bond Curve")
    cad_curve = fetch_cad_curve()
    us_curve = fetch_us_treasury_curve()

    if not cad_curve:
        st.error("Could not fetch the Canadian curve right now — try again shortly.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(cad_curve.keys()), y=list(cad_curve.values()),
                                  mode="lines+markers", name="Canada (GoC)",
                                  line=dict(color="#f87171", width=3), marker=dict(size=8)))
        fig.update_layout(xaxis_title="Tenor", yaxis_title="Yield (%)",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(color="#e8eaed"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Source: Bank of Canada Valet API, Government of Canada benchmark bond yields.")

        st.divider()
        st.subheader("Canada vs. U.S. Curve Spread")
        st.caption("Tenor-by-tenor rate differential — the core input for cross-country rates and FX trades (CAD funding vs. USD funding at each maturity).")

        common_tenors = [t for t in ["2Y", "5Y", "10Y"] if t in cad_curve and t in us_curve]
        if common_tenors:
            spread_rows = [{"Tenor": t, "Canada": cad_curve[t], "U.S.": us_curve[t],
                            "Spread (CAD - US, bps)": round((cad_curve[t] - us_curve[t]) * 100, 0)}
                           for t in common_tenors]
            st.dataframe(pd.DataFrame(spread_rows), use_container_width=True, hide_index=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=[r["Tenor"] for r in spread_rows],
                                   y=[r["Spread (CAD - US, bps)"] for r in spread_rows],
                                   marker_color=["#4ade80" if r["Spread (CAD - US, bps)"] >= 0 else "#f87171" for r in spread_rows]))
            fig2.update_layout(yaxis_title="Spread (bps)", paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaed"))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Couldn't match tenors between the two curves right now.")

# --- TAB 6: FIXED INCOME STRATEGY ---
with tab6:
    st.subheader("Duration & Convexity Calculator")
    st.caption("Models a single bond's price sensitivity to interest rate moves — the core risk metric for any fixed income strategy.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        face_value = st.number_input("Face Value ($)", value=1000.0, step=100.0)
    with c2:
        coupon_rate = st.number_input("Coupon Rate (%)", value=4.0, step=0.25) / 100
    with c3:
        yield_rate = st.number_input("Yield to Maturity (%)", value=4.5, step=0.25) / 100
    with c4:
        maturity_years = st.number_input("Years to Maturity", value=10.0, step=0.5)

    dc = bond_duration_convexity(face_value, coupon_rate, yield_rate, maturity_years)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Price", f"${dc['price']:,.2f}")
    m2.metric("Macaulay Duration", f"{dc['macaulay_duration']:.2f} yrs")
    m3.metric("Modified Duration", f"{dc['modified_duration']:.2f}")
    m4.metric("Convexity", f"{dc['convexity']:.2f}")

    st.markdown("**Estimated Price Impact of a Rate Shock**")
    shock_bps = st.slider("Yield shock (bps)", -300, 300, 100, step=25)
    pct_change = estimate_price_change(dc["modified_duration"], dc["convexity"], shock_bps)
    st.metric(f"Estimated Price Change ({shock_bps:+d} bps)", f"{pct_change*100:+.2f}%",
              delta=f"${dc['price']*pct_change:+,.2f}")
    st.caption("Formula: %ΔPrice ≈ −(Modified Duration × Δy) + 0.5 × Convexity × Δy². The convexity term matters most for large yield moves.")

    st.divider()
    st.subheader("Roll-Down Analysis")
    st.caption("Estimates the 1-year return from a bond simply 'rolling down' a positively-sloped curve to a shorter maturity, holding the curve shape constant — a classic fixed income carry strategy independent of any rate view.")

    curve_choice = st.radio("Curve", ["U.S. Treasury", "Canada (GoC)"], horizontal=True, key="rolldown_curve")
    curve_data = us_curve if curve_choice == "U.S. Treasury" else cad_curve
    tenor_map_years = {"2Y": 2, "3Y": 3, "5Y": 5, "7Y": 7, "10Y": 10, "Long (30Y)": 30, "30Y": 30}
    available_tenors = [tenor_map_years[t] for t in curve_data if t in tenor_map_years]

    if len(available_tenors) >= 2:
        rd_df = rolldown_analysis(curve_data, available_tenors)
        st.dataframe(rd_df, use_container_width=True, hide_index=True)
        st.caption("Positive roll-down bps means the bond's yield falls as it rolls to the shorter tenor, which pushes its price up — this is estimated, not a live trade recommendation.")
    else:
        st.info("Not enough tenor data available on this curve right now to run roll-down analysis.")

# --- TAB 7: TRADING DESK Q&A ---
with tab7:
    st.subheader("Trading Desk Q&A")
    st.caption("Answers to common desk questions — live ones pull today's actual data, conceptual ones are frameworks you can apply to any day's numbers.")

    pulse = fetch_market_pulse()
    carry_df_qa = build_carry_table(fetch_policy_rates(), fetch_fx_spot())
    curve_sig_qa = curve_steepness_signal(5)

    with st.expander("Walk me through today's market", expanded=True):
        if pulse:
            st.markdown(f"""
            As of {pulse['as_of']}:
            - **10-Year Treasury yield:** {pulse['ust10_level']:.2f}% ({pulse['ust10_chg_bps']:+.1f} bps on the day)
            - **S&P 500:** {pulse['spx_chg_pct']:+.2f}% on the day
            - **TSX:** {pulse['tsx_chg_pct']:+.2f}% on the day
            - **VIX:** {pulse['vix_level']:.1f} ({pulse['vix_chg']:+.1f}% on the day)
            - **USD/CAD:** {pulse['cad_level']:.4f} ({pulse['cad_chg_pct']:+.2f}% on the day)

            **Read:** {"Yields rose alongside a VIX pickup, consistent with a risk-off or rate-repricing session" if pulse['ust10_chg_bps'] and pulse['ust10_chg_bps'] > 0 and pulse['vix_chg'] and pulse['vix_chg'] > 0 else "Yields and equities moved together today — check whether it's a growth-data-driven session (both up) or a risk-off session (both down)"}.
            """)
        else:
            st.warning("Live market data unavailable right now.")

    with st.expander("Why did the bond market move today?"):
        if pulse and pulse["ust10_chg_bps"] is not None:
            direction = "sold off (yields rose)" if pulse["ust10_chg_bps"] > 0 else "rallied (yields fell)"
            st.markdown(f"""
            The 10-year Treasury {direction} by {abs(pulse['ust10_chg_bps']):.1f} bps today. In practice, bond moves on
            any given day trace back to one of a few drivers: **new economic data** (CPI, payrolls, GDP surprising vs.
            consensus), **Fed/central bank commentary** (a speaker leaning more hawkish or dovish than expected),
            **auction results** (weak demand at a Treasury auction pushes yields up), or **risk sentiment** (a flight
            to safety pulls yields down as investors buy bonds). Today's VIX move ({pulse['vix_chg']:+.1f}%) is the
            quickest tell — a rising VIX alongside falling yields points to risk-off flows; a rising VIX alongside
            rising yields is more unusual and often points to inflation or supply concerns overriding the risk-off bid.
            """)
        else:
            st.warning("Live data unavailable — but the framework: check econ data releases, central bank speakers, auction results, and risk sentiment (VIX) first.")

    with st.expander("Why are Treasury yields rising?"):
        st.markdown("""
        Four standard explanations, roughly in order of how a desk would triage them:
        1. **Growth is stronger than expected** — the market prices in a higher path for future short rates
        2. **Inflation is stickier than expected** — investors demand more yield to compensate for eroding purchasing power
        3. **The Fed is signaling fewer/later rate cuts** (or more hikes) — repricing the expected policy path
        4. **Supply concerns** — heavy Treasury issuance without matching demand pushes yields up to clear the market (this is the "term premium" story that's been more relevant in recent years)

        The fastest way to tell which one is happening: check whether **breakeven inflation** (TIPS-implied) moved with
        nominal yields (inflation story) or **real yields** moved on their own (growth or supply story).
        """)

    with st.expander("How do interest rates affect equities?"):
        st.markdown("""
        Two main channels:
        - **Discount rate channel:** equity valuations are the present value of future cash flows. Higher rates mean a
          higher discount rate, which mechanically lowers the present value of those cash flows — hits long-duration
          growth stocks (most of their value is in distant cash flows) harder than value/short-duration stocks.
        - **Economic channel:** rate moves reflect and influence the growth outlook. Rates rising because growth is
          strong can be *good* for cyclical equities even as it's bad for bond prices — this is why "yields up, stocks
          up" happens on strong-growth days, and "yields up, stocks down" happens on inflation-scare or hawkish-Fed days.

        Rule of thumb a desk uses: **ask why rates are moving before assuming the equity reaction.**
        """)

    with st.expander("Pitch me a stock or bond"):
        if not carry_df_qa.empty and curve_sig_qa:
            top_carry = carry_df_qa.iloc[0]
            st.markdown(f"""
            Here's a live, data-backed rates/FX pitch using today's numbers from this dashboard:

            **Trade idea: Long {top_carry['Currency']} funded in USD**

            - {top_carry['Currency']} currently carries {top_carry['Carry (bps)']:.0f} bps over USD based on policy
              rate differentials — the highest in the G10 set right now
            - Current spot: {top_carry['Spot']:.4f}
            - **Thesis:** as long as {top_carry['Currency']} doesn't depreciate by more than the carry earned, this
              trade is profitable — carry compensates for a stable-to-modestly-weaker currency
            - **Risk:** carry trades unwind sharply during risk-off shocks (the "carry trade unwind" is a well-known
              volatility event) — size accordingly and watch the VIX

            Separately, on rates: the 2s10s curve is currently at a z-score of {curve_sig_qa['z_score']:.2f} relative
            to its 5-year history — {curve_sig_qa['signal'].lower()}
            """)
        else:
            st.warning("Live data unavailable to build a current pitch — but the structure: pick a specific idea, state the thesis in one sentence, name the catalyst, and name the risk.")

    with st.expander("What would happen if the Bank of Canada cut rates?"):
        st.markdown("""
        Standard transmission mechanism:
        - **CAD government bond yields fall**, especially at the short end (2yr moves most, long end moves least — the
          curve typically steepens on a cut since long-end yields reflect longer-run growth/inflation expectations
          more than near-term policy)
        - **CAD weakens vs. USD** — lower Canadian rates make CAD-denominated assets less attractive relative to USD,
          all else equal (this is exactly the carry-trade logic in the FX Carry Screener tab)
        - **Canadian equities, especially rate-sensitive sectors** (REITs, utilities, financials' net interest margin
          effects are mixed) typically react positively to lower discount rates, though the *reason* for the cut
          matters — a cut into weakening growth data can still see equities fall if the growth story deteriorates
          faster than rates help
        - **Mortgage-sensitive Canadian consumers** benefit directly given Canada's shorter mortgage terms vs. the U.S.
        """)

    with st.expander("How would tariffs affect the Canadian dollar?"):
        st.markdown("""
        Tariffs on Canadian exports (especially to the U.S., Canada's dominant trade partner) typically pressure CAD
        through a few channels:
        - **Trade balance channel:** tariffs reduce demand for Canadian exports, weakening the trade balance and
          reducing USD inflows that would otherwise support CAD
        - **Growth channel:** a hit to export-heavy sectors (energy, autos, lumber) slows Canadian growth, which
          typically brings forward BoC rate cut expectations — and lower expected rates weaken CAD further (same
          mechanism as the question above)
        - **Risk sentiment channel:** trade-war escalation tends to be risk-off broadly, and CAD (as a commodity-linked,
          "risk-on" currency) tends to underperform in risk-off environments regardless of the direct trade impact
        - **Commodity channel:** if tariffs specifically hit Canadian oil/energy exports, this compounds the CAD
          weakness since oil prices and CAD are historically positively correlated
        """)

    with st.expander("Explain duration"):
        st.markdown("""
        Duration measures a bond's **price sensitivity to a change in yield** — specifically, the approximate
        percentage price change for a 1% (100 bps) move in yield.

        - **Macaulay Duration:** the weighted-average time until a bondholder receives their cash flows (weighted by
          the present value of each cash flow). Measured in years.
        - **Modified Duration:** Macaulay Duration adjusted for the current yield level — this is the number you
          actually use to estimate price sensitivity: %ΔPrice ≈ −Modified Duration × Δy

        **Intuition:** longer maturity → higher duration → more price sensitivity to rate moves. A 30-year bond moves
        far more per basis point than a 2-year bond. Higher coupons *reduce* duration slightly (more cash flow arrives
        earlier, pulling the weighted-average time closer to today).

        Try it live in the **Duration & Convexity Calculator** on the Fixed Income Strategy tab.
        """)

    with st.expander("Explain convexity"):
        st.markdown("""
        Duration is a *linear* approximation of how a bond's price responds to yield changes — but the real
        relationship between price and yield is curved (convex), not a straight line. Convexity measures **how much
        that curve bends** — i.e., the second-order correction to the duration estimate.

        **Why it matters:** for a given move in yield, a bond price rises *more* than duration alone predicts when
        yields fall, and falls *less* than duration alone predicts when yields rise. This is a genuinely good property
        for a bondholder — it's why convexity is valuable and long-convexity positions (or options) command a premium.

        **The full formula:** %ΔPrice ≈ −(Modified Duration × Δy) + 0.5 × Convexity × Δy². The convexity term is small
        for modest yield moves but matters a lot for large ones — which is exactly what the calculator above shows
        when you push the yield shock slider to the extremes.
        """)

    with st.expander("What's the yield curve telling us?"):
        if curve_sig_qa:
            st.markdown(f"""
            Right now, the 2s10s spread sits at {curve_sig_qa['current_spread']*100:.0f} bps, vs. a 5-year historical
            average of {curve_sig_qa['mean_spread']*100:.0f} bps (z-score: {curve_sig_qa['z_score']:.2f}).

            **General framework for reading the curve:**
            - **Steep curve** (long yields well above short yields): market expects growth/inflation ahead, or expects
              the central bank to cut short-term rates while long-run expectations stay anchored
            - **Flat curve:** market is uncertain, or pricing a stable rate environment
            - **Inverted curve** (short yields above long yields): historically one of the more reliable recession
              signals — it implies the market expects the central bank to cut rates in the future in response to
              weakening growth, pulling expected future short rates (and therefore long yields, which are roughly an
              average of expected future short rates) below today's short rate

            {curve_sig_qa['signal']}
            """)
        else:
            st.warning("Live curve data unavailable right now.")

# --- TAB 8: ABOUT THIS DATA ---
with tab8:
    st.subheader("About the Data in This Dashboard")
    st.markdown("""
    **What's live and real:**
    - U.S. Treasury yield curve — FRED constant-maturity Treasury series (DGS1MO through DGS30)
    - Canadian government bond yields — Bank of Canada Valet API benchmark bond series
    - G10 FX spot rates — Yahoo Finance
    - Central bank policy rates — FRED (sourced from OECD-reported series, updated as each bank changes rates)
    - Today's market pulse (10yr yield, S&P 500, TSX, VIX, USD/CAD) — Yahoo Finance daily data

    **What's simplified or approximated, and why:**
    - **No live interest rate swap curve.** Real SOFR/CORRA swap rates are licensed data (ICE Swap Rate, Bloomberg,
      Refinitiv) — there is no free, reliable, continuously-updated public source for them. This dashboard uses
      government bond yields as the closest free proxy for rates-differential and curve-trading logic. Swap spreads
      (swap rate minus matching-maturity government yield) are real and meaningful in professional trading, but
      require paid data this dashboard doesn't have access to.
    - **The carry backtest uses today's policy rate differentials applied across the whole historical window**, since
      clean historical daily policy rates for all 9 G10 currencies aren't available from one free source. Treat it as
      illustrative of the carry-trade mechanic, not a precise historical P&L.
    - **Roll-down analysis uses an approximated modified duration** (not calculated from each specific benchmark
      bond's actual coupon), since the Bank of Canada and Treasury benchmark series report yields, not full bond
      terms. It's directionally right but not exact.
    """)
