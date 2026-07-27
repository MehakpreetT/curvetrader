"""
SignalR — Personal Bond Derivatives Trading Signaler
------------------------------------------------------------
Combines news sentiment, a rates-aware Monte Carlo simulation, and stress
testing across options, a modeled swap, and a modeled forward — all
tailored to a chosen Treasury bond ETF (TLT, IEF, SHY, TIP) — into a
single composite trading signal.

Options data is REAL and LIVE (Yahoo Finance's actual listed options
chains for these ETFs). Swap and forward analysis use MODELED contracts
built on real government yield curve data, since true swap/forward market
data is licensed and not freely available — this is disclosed in-app.

Run locally with: streamlit run streamlit_app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import date, timedelta

st.set_page_config(page_title="SignalR", page_icon="🎯", layout="wide")

st.markdown("""
    <style>
        .main { background: linear-gradient(160deg, #0a0e1a 0%, #0e1117 100%); }
        h1, h2, h3 { letter-spacing: -0.3px; }
        h2, h3 { color: #60a5fa; }
        [data-testid="stHeaderActionElements"] { display: none; }

        h1 {
            background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 50%, #60a5fa 100%);
            background-size: 200% auto;
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: shine 6s linear infinite;
        }
        @keyframes shine { to { background-position: 200% center; } }

        [data-testid="stMetricValue"] { color: #e8eaed; }
        [data-testid="stMetricLabel"] { color: #9aa0ab; }
        [data-testid="stMetric"] {
            background: linear-gradient(160deg, #171b24 0%, #131722 100%);
            border: 1px solid #2b2f38; border-radius: 10px; padding: 12px 16px;
            transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: #3b82f6;
            box-shadow: 0 4px 16px rgba(59, 130, 246, 0.18);
        }

        .stButton>button {
            background: linear-gradient(135deg, #2563eb, #3b82f6);
            color: white; border: none; border-radius: 8px; font-weight: 600;
            padding: 0.5em 1.5em;
            transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.2s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #1d4ed8, #2563eb);
            transform: translateY(-1px);
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
        }
        .stButton>button:active { transform: translateY(0px); }

        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #171b24; border-radius: 8px 8px 0 0;
            border: 1px solid #2b2f38; border-bottom: none;
            color: #9aa0ab; padding: 8px 14px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1d4ed8 !important; color: white !important;
        }

        [data-testid="stExpander"] {
            background: linear-gradient(160deg, #171b24 0%, #131722 100%);
            border: 1px solid #2b2f38; border-radius: 10px;
        }

        [data-testid="stAppViewContainer"] { animation: fadeIn 0.35s ease-in; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    </style>
""", unsafe_allow_html=True)

st.title("🎯 SignalR")
st.caption("A personal trading signaler for bond derivatives — sentiment analysis, Monte Carlo simulation, and stress testing across options, swaps, and forwards.")

BOND_ETFS = {
    "TLT": {"name": "20+ Year Treasury Bond ETF", "duration": 17.0, "convexity": 4.0, "category": "U.S. Treasury"},
    "IEF": {"name": "7-10 Year Treasury Bond ETF", "duration": 7.5, "convexity": 0.7, "category": "U.S. Treasury"},
    "SHY": {"name": "1-3 Year Treasury Bond ETF", "duration": 1.9, "convexity": 0.05, "category": "U.S. Treasury"},
    "TIP": {"name": "TIPS Bond ETF", "duration": 7.0, "convexity": 0.6, "category": "U.S. Treasury"},
    "LQD": {"name": "Investment-Grade Corporate Bond ETF", "duration": 8.5, "convexity": 0.9, "category": "Corporate"},
    "HYG": {"name": "High-Yield Corporate Bond ETF", "duration": 3.5, "convexity": 0.2, "category": "Corporate"},
}


# =======================================================================
# 1. DATA FETCH
# =======================================================================
@st.cache_data(ttl=60 * 15)
def fetch_underlying_data(ticker):
    t = yf.Ticker(ticker)
    hist = t.history(period="2y")
    if hist.empty:
        return None
    info = t.info
    current_price = float(hist["Close"].iloc[-1])
    daily_returns = hist["Close"].pct_change().dropna()
    daily_vol = daily_returns.std()
    ann_vol = daily_vol * np.sqrt(252)
    return {"ticker": ticker, "price": current_price, "history": hist,
            "daily_vol": daily_vol, "ann_vol": ann_vol, "name": info.get("shortName", ticker)}


@st.cache_data(ttl=60 * 30)
def fetch_news_sentiment(ticker):
    positive_words = {"rally", "growth", "beat", "optimism", "gain", "strong", "resilient", "upgrade",
                      "cut", "ease", "easing", "dovish", "cooling", "soft landing", "stabilize"}
    negative_words = {"recession", "shock", "sell-off", "selloff", "inflation", "hike", "hawkish",
                      "surge", "spike", "crash", "downgrade", "slump", "default", "risk", "volatil"}
    try:
        items = yf.Ticker(ticker).news[:8]
        headlines = []
        score = 0
        for item in items:
            content = item.get("content", item)
            title = content.get("title") or item.get("title", "")
            link = (content.get("canonicalUrl") or {}).get("url") or item.get("link", "")
            publisher = (content.get("provider") or {}).get("displayName", "")
            title_lower = title.lower()
            pos_hits = sum(1 for w in positive_words if w in title_lower)
            neg_hits = sum(1 for w in negative_words if w in title_lower)
            headline_score = pos_hits - neg_hits
            score += headline_score
            headlines.append({"title": title, "link": link, "publisher": publisher, "score": headline_score})
        normalized_score = max(-1.0, min(1.0, score / (len(headlines) * 2))) if headlines else 0.0
        return {"headlines": headlines, "score": normalized_score}
    except Exception:
        return {"headlines": [], "score": 0.0}


@st.cache_data(ttl=60 * 60 * 6)
def fetch_sofr():
    import pandas_datareader.data as web
    end = date.today()
    start = end - timedelta(days=30)
    try:
        d = web.DataReader("SOFR", "fred", start, end).dropna()
        return float(d.iloc[-1].iloc[0]) / 100
    except Exception:
        return 0.045  # fallback assumption, disclosed in UI


GOC_SERIES = {
    "2Y": "BD.CDN.2YR.DQ.YLD", "3Y": "BD.CDN.3YR.DQ.YLD", "5Y": "BD.CDN.5YR.DQ.YLD",
    "7Y": "BD.CDN.7YR.DQ.YLD", "10Y": "BD.CDN.10YR.DQ.YLD", "30Y": "BD.CDN.LONG.DQ.YLD",
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
            curve[label] = float(obs[series]["v"]) / 100  # match fraction format used elsewhere
        except Exception:
            continue
    return curve


@st.cache_data(ttl=60 * 60 * 6)
def fetch_treasury_curve():
    import pandas_datareader.data as web
    series = {"1Y": "DGS1", "2Y": "DGS2", "3Y": "DGS3", "5Y": "DGS5", "7Y": "DGS7",
              "10Y": "DGS10", "20Y": "DGS20", "30Y": "DGS30"}
    end = date.today()
    start = end - timedelta(days=30)
    curve = {}
    for label, code in series.items():
        try:
            d = web.DataReader(code, "fred", start, end).dropna()
            if not d.empty:
                curve[label] = float(d.iloc[-1].iloc[0]) / 100
        except Exception:
            continue
    return curve


# =======================================================================
# 2. MONTE CARLO — rates-aware (simulates yield changes, converts to price
#    via duration/convexity, rather than a generic equity-style price walk)
# =======================================================================
def run_monte_carlo(current_price, ann_vol, duration, convexity, horizon_days, n_sims=5000):
    dt = 1 / 252
    n_steps = horizon_days
    implied_yield_vol = ann_vol / duration  # rough conversion: price vol -> yield vol via duration

    np.random.seed(None)
    yield_changes = np.random.normal(0, implied_yield_vol * np.sqrt(dt), size=(n_sims, n_steps))
    cumulative_dy = yield_changes.sum(axis=1)

    price_changes_pct = -duration * cumulative_dy + 0.5 * convexity * cumulative_dy ** 2
    terminal_prices = current_price * (1 + price_changes_pct)

    var_95 = np.percentile(terminal_prices, 5)
    var_99 = np.percentile(terminal_prices, 1)
    prob_loss = (terminal_prices < current_price).mean()
    expected_price = terminal_prices.mean()

    return {"terminal_prices": terminal_prices, "var_95": var_95, "var_99": var_99,
            "prob_loss": prob_loss, "expected_price": expected_price, "current_price": current_price}


# =======================================================================
# =======================================================================
# 2b. DURATION, CONVEXITY & ROLL-DOWN (Fixed Income Strategy toolkit)
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
    return -modified_duration * dy + 0.5 * convexity * dy ** 2


def rolldown_analysis(curve, tenor_years_list):
    """For each tenor, estimate 1-year roll-down return: the price gain from
    a bond 'rolling down' the curve to a shorter maturity in 1 year, holding
    the curve shape constant (a standard simplifying assumption). Expects
    curve values as fractions (e.g., 0.045), matching fetch_treasury_curve."""
    tenor_map = {1: "1Y", 2: "2Y", 3: "3Y", 5: "5Y", 7: "7Y", 10: "10Y", 20: "20Y", 30: "30Y"}
    results = []
    sorted_tenors = sorted(tenor_years_list)
    for tenor in sorted_tenors:
        label = tenor_map.get(tenor)
        if label not in curve:
            continue
        current_yield = curve[label]
        shorter_candidates = [t for t in sorted_tenors if t < tenor]
        if not shorter_candidates:
            continue
        rolled_tenor = max(shorter_candidates)
        rolled_label = tenor_map.get(rolled_tenor)
        if rolled_label not in curve:
            continue
        rolled_yield = curve[rolled_label]
        rolldown_bps = (current_yield - rolled_yield) * 10000  # fraction -> bps
        approx_mod_dur = rolled_tenor * 0.9
        estimated_return = approx_mod_dur * (rolldown_bps / 10000) * 100
        results.append({"Tenor": label, "Yield": f"{current_yield*100:.2f}%", "Rolls to": rolled_label,
                        "Roll-Down (bps)": round(rolldown_bps, 1), "Est. 1Yr Roll Return (%)": round(estimated_return, 2)})
    return pd.DataFrame(results)


# =======================================================================
# 3. OPTIONS — real live chain + Black-Scholes stress testing
# =======================================================================
@st.cache_data(ttl=60 * 15)
def fetch_option_chain(ticker, expiry):
    t = yf.Ticker(ticker)
    chain = t.option_chain(expiry)
    return chain.calls, chain.puts


def black_scholes(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def stress_test_option(S, K, T, r, sigma, option_type, price_shocks_pct):
    results = []
    for shock in price_shocks_pct:
        shocked_S = S * (1 + shock / 100)
        price = black_scholes(shocked_S, K, T, r, sigma, option_type)
        results.append({"Price Shock": f"{shock:+.0f}%", "Underlying Price": shocked_S, "Option Value": price})
    return pd.DataFrame(results)


# =======================================================================
# 4. MODELED SWAP — fixed-for-floating, valued off the real Treasury curve
# =======================================================================
def price_swap(notional, fixed_rate, tenor_years, curve, payment_freq=2):
    """
    Simplified swap valuation: fixed leg PV using curve-implied discount
    factors (flat-forward interpolated from available tenors), floating
    leg assumed to reprice to par at each reset (standard simplifying
    assumption for a freshly-issued floating leg).
    """
    tenors = sorted(float(t.replace("Y", "")) for t in curve.keys())
    yields = [curve[f"{int(t)}Y"] for t in tenors]

    def interpolated_yield(t):
        if t <= tenors[0]:
            return yields[0]
        if t >= tenors[-1]:
            return yields[-1]
        for i in range(len(tenors) - 1):
            if tenors[i] <= t <= tenors[i + 1]:
                frac = (t - tenors[i]) / (tenors[i + 1] - tenors[i])
                return yields[i] + frac * (yields[i + 1] - yields[i])
        return yields[-1]

    n_payments = int(tenor_years * payment_freq)
    fixed_coupon = notional * fixed_rate / payment_freq

    fixed_leg_pv = 0.0
    for i in range(1, n_payments + 1):
        t = i / payment_freq
        y = interpolated_yield(t)
        df = 1 / (1 + y / payment_freq) ** (payment_freq * t)
        cf = fixed_coupon + (notional if i == n_payments else 0)
        fixed_leg_pv += cf * df

    floating_leg_pv = notional  # reprices to par
    swap_value_to_fixed_receiver = fixed_leg_pv - floating_leg_pv

    return {"fixed_leg_pv": fixed_leg_pv, "floating_leg_pv": floating_leg_pv,
            "swap_value": swap_value_to_fixed_receiver}


def stress_test_swap(notional, fixed_rate, tenor_years, curve, shock_scenarios):
    results = []
    for label, shock_fn in shock_scenarios.items():
        shocked_curve = {k: shock_fn(k, v) for k, v in curve.items()}
        result = price_swap(notional, fixed_rate, tenor_years, shocked_curve)
        results.append({"Scenario": label, "Swap Value (Fixed Receiver)": result["swap_value"]})
    return pd.DataFrame(results)


# =======================================================================
# 5. MODELED FORWARD — bond forward priced off spot + repo/carry
# =======================================================================
def price_bond_forward(spot_price, repo_rate, time_to_delivery_years):
    return spot_price * (1 + repo_rate * time_to_delivery_years)


def stress_test_forward(spot_price, repo_rate, time_to_delivery, price_shocks_pct, rate_shocks_bps):
    results = []
    for p_shock in price_shocks_pct:
        for r_shock in rate_shocks_bps:
            shocked_spot = spot_price * (1 + p_shock / 100)
            shocked_repo = repo_rate + r_shock / 10000
            fwd_price = price_bond_forward(shocked_spot, shocked_repo, time_to_delivery)
            results.append({"Price Shock": f"{p_shock:+.0f}%", "Repo Rate Shock": f"{r_shock:+.0f}bps",
                            "Forward Price": fwd_price})
    return pd.DataFrame(results)


# =======================================================================
# 6. UI
# =======================================================================
ticker = st.selectbox("Choose a Bond ETF", list(BOND_ETFS.keys()),
                       format_func=lambda t: f"{t} — {BOND_ETFS[t]['name']} ({BOND_ETFS[t]['category']})")
etf_info = BOND_ETFS[ticker]

data = fetch_underlying_data(ticker)
if data is None:
    st.error("Couldn't fetch data for this ticker right now.")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Price", f"${data['price']:.2f}")
m2.metric("Annualized Volatility", f"{data['ann_vol']*100:.1f}%")
m3.metric("Assumed Duration", f"{etf_info['duration']:.1f} yrs")
m4.metric("Assumed Convexity", f"{etf_info['convexity']:.2f}")
st.caption("Duration/convexity are standard published approximations for this ETF, not calculated live from its exact holdings.")
if etf_info["category"] == "Corporate":
    st.info("This is a corporate bond ETF — its price and yield reflect **credit risk** (the issuer's ability to pay) on top of interest rate risk, unlike Treasuries. The Monte Carlo simulation and duration/convexity math here only model rate risk, not credit/default risk.")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Sentiment Analysis", "Monte Carlo Simulation", "Options Stress Test",
    "Swap Stress Test", "Forward Stress Test", "Fixed Income Strategy",
    "Canadian Government Bonds", "Composite Signal", "Education"
])

# --- TAB 1: SENTIMENT ---
with tab1:
    st.subheader(f"News Sentiment — {ticker}")
    sentiment = fetch_news_sentiment(ticker)
    st.metric("Sentiment Score", f"{sentiment['score']:+.2f}", help="Range: -1 (bearish) to +1 (bullish), based on keyword scoring of recent headlines.")

    if sentiment["headlines"]:
        for h in sentiment["headlines"]:
            tag = "🟢" if h["score"] > 0 else ("🔴" if h["score"] < 0 else "⚪")
            st.markdown(f"{tag} [{h['title']}]({h['link']}) — *{h['publisher']}*" if h["link"] else f"{tag} {h['title']}")
    else:
        st.info("No recent headlines available for this ticker right now.")

# --- TAB 2: MONTE CARLO ---
with tab2:
    st.subheader("Rates-Aware Monte Carlo Simulation")
    st.caption("Simulates future yield changes (not just a generic price random walk) and converts them to price outcomes using duration/convexity — a bond-appropriate simulation approach.")

    horizon_days = st.slider("Simulation horizon (trading days)", 5, 126, 21)
    n_sims = st.select_slider("Number of simulations", [1000, 2500, 5000, 10000], value=5000)

    if st.button("Run Simulation"):
        with st.spinner("Simulating..."):
            mc_result = run_monte_carlo(data["price"], data["ann_vol"], etf_info["duration"],
                                        etf_info["convexity"], horizon_days, n_sims)
        st.session_state["_mc_result"] = mc_result

    mc_result = st.session_state.get("_mc_result")
    if mc_result:
        fig = go.Figure(data=[go.Histogram(x=mc_result["terminal_prices"], nbinsx=60, marker_color="#60a5fa")])
        fig.add_vline(x=mc_result["current_price"], line_dash="dash", line_color="#e8eaed", annotation_text="Current Price")
        fig.add_vline(x=mc_result["var_95"], line_dash="dot", line_color="#f87171", annotation_text="95% VaR")
        fig.update_layout(xaxis_title="Simulated Price", yaxis_title="Frequency",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaed"))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Expected Price", f"${mc_result['expected_price']:.2f}")
        c2.metric("95% VaR", f"${mc_result['var_95']:.2f}")
        c3.metric("99% VaR", f"${mc_result['var_99']:.2f}")
        c4.metric("P(Loss)", f"{mc_result['prob_loss']*100:.1f}%")

# --- TAB 3: OPTIONS STRESS TEST ---
with tab3:
    st.subheader(f"Live Options Chain — {ticker}")
    st.caption("Real, live listed options data from Yahoo Finance — Treasury ETFs like this one trade actual exchange-listed options.")

    try:
        expiries = yf.Ticker(ticker).options
    except Exception:
        expiries = []

    if not expiries:
        st.error("Couldn't fetch the options chain right now.")
    else:
        expiry = st.selectbox("Expiration", expiries)
        calls, puts = fetch_option_chain(ticker, expiry)

        option_type = st.radio("Option Type", ["call", "put"], horizontal=True)
        chain_df = calls if option_type == "call" else puts

        if chain_df.empty:
            st.warning("No contracts available for this expiry/type.")
        else:
            strike = st.selectbox("Strike", sorted(chain_df["strike"].unique()))
            contract = chain_df[chain_df["strike"] == strike].iloc[0]
            iv = contract.get("impliedVolatility", 0.15) or 0.15

            c1, c2, c3 = st.columns(3)
            c1.metric("Last Price", f"${contract['lastPrice']:.2f}")
            c2.metric("Implied Volatility", f"{iv*100:.1f}%")
            c3.metric("Open Interest", f"{contract.get('openInterest', 0):.0f}")

            T = max((pd.Timestamp(expiry) - pd.Timestamp(date.today())).days / 365, 0.001)
            sofr = fetch_sofr()

            st.divider()
            st.markdown("**Stress Test: Option Value vs. Underlying Price Shock**")
            shocks = list(range(-10, 11, 2))
            stress_df = stress_test_option(data["price"], strike, T, sofr, iv, option_type, shocks)
            fig = go.Figure(data=[go.Scatter(x=stress_df["Price Shock"], y=stress_df["Option Value"],
                                              mode="lines+markers", line=dict(color="#60a5fa", width=3))])
            fig.update_layout(xaxis_title="Underlying Price Shock", yaxis_title="Option Value ($)",
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaed"))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(stress_df, use_container_width=True, hide_index=True)

# --- TAB 4: SWAP STRESS TEST ---
with tab4:
    st.subheader("Modeled Interest Rate Swap — Stress Test")
    st.warning("No free source publishes live swap market rates (ICE Swap Rate / Bloomberg are licensed). "
               "This models a hypothetical swap valued off REAL Treasury curve data — the mechanics are legitimate, "
               "but this is not a live-priced market swap.")

    curve = fetch_treasury_curve()
    if not curve:
        st.error("Couldn't fetch the Treasury curve right now.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            notional = st.number_input("Notional ($)", value=10_000_000.0, max_value=10_000_000.0, step=1_000_000.0)
        with c2:
            fixed_rate = st.number_input("Fixed Rate (%)", value=curve.get("10Y", 0.04) * 100, step=0.1) / 100
        with c3:
            swap_tenor = st.number_input("Tenor (years)", value=10, min_value=1, max_value=30, step=1)

        base_result = price_swap(notional, fixed_rate, swap_tenor, curve)
        st.metric("Swap Value (to fixed receiver)", f"${base_result['swap_value']:,.0f}")

        st.divider()
        st.markdown("**Stress Scenarios**")
        scenarios = {
            "Base Case": lambda k, v: v,
            "+50bps Parallel Shift": lambda k, v: v + 0.005,
            "-50bps Parallel Shift": lambda k, v: v - 0.005,
            "Steepener (+50bps long end)": lambda k, v: v + 0.005 if k in ["10Y", "30Y"] else v,
            "Flattener (-50bps long end)": lambda k, v: v - 0.005 if k in ["10Y", "30Y"] else v,
        }
        stress_df = stress_test_swap(notional, fixed_rate, swap_tenor, curve, scenarios)
        st.dataframe(stress_df, use_container_width=True, hide_index=True)

        fig = go.Figure(data=[go.Bar(x=stress_df["Scenario"], y=stress_df["Swap Value (Fixed Receiver)"],
                                      marker_color=["#4ade80" if v >= 0 else "#f87171" for v in stress_df["Swap Value (Fixed Receiver)"]])])
        fig.update_layout(yaxis_title="Swap Value ($)", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaed"))
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 5: FORWARD STRESS TEST ---
with tab5:
    st.subheader("Modeled Bond Forward — Stress Test")
    st.caption("Forward price modeled off the real current ETF price and SOFR as a repo-rate proxy — standard cost-of-carry forward pricing.")

    sofr = fetch_sofr()
    c1, c2 = st.columns(2)
    with c1:
        delivery_months = st.slider("Months to Delivery", 1, 12, 3)
    with c2:
        st.metric("Repo Rate Proxy (SOFR)", f"{sofr*100:.2f}%")

    time_to_delivery = delivery_months / 12
    base_fwd = price_bond_forward(data["price"], sofr, time_to_delivery)
    st.metric("Forward Price", f"${base_fwd:.2f}")

    st.divider()
    st.markdown("**Stress Test Grid**")
    price_shocks = [-5, -2, 0, 2, 5]
    rate_shocks = [-50, 0, 50]
    fwd_stress_df = stress_test_forward(data["price"], sofr, time_to_delivery, price_shocks, rate_shocks)
    pivot = fwd_stress_df.pivot(index="Price Shock", columns="Repo Rate Shock", values="Forward Price")
    st.dataframe(pivot, use_container_width=True)

# --- TAB 6: FIXED INCOME STRATEGY ---
with tab6:
    st.subheader("Duration & Convexity Calculator")
    st.caption("Models a single bond's price sensitivity to interest rate moves — the core risk metric for any fixed income strategy.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        face_value = st.number_input("Face Value ($)", value=1000.0, max_value=10_000_000.0, step=100.0)
    with c2:
        coupon_rate = st.number_input("Coupon Rate (%)", value=4.0, step=0.25) / 100
    with c3:
        yield_rate = st.number_input("Yield to Maturity (%)", value=4.5, step=0.25) / 100
    with c4:
        maturity_years = st.number_input("Years to Maturity", value=10.0, step=0.5)

    dc = bond_duration_convexity(face_value, coupon_rate, yield_rate, maturity_years)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Price", f"${dc['price']:,.2f}")
    d2.metric("Macaulay Duration", f"{dc['macaulay_duration']:.2f} yrs")
    d3.metric("Modified Duration", f"{dc['modified_duration']:.2f}")
    d4.metric("Convexity", f"{dc['convexity']:.2f}")

    st.markdown("**Estimated Price Impact of a Rate Shock**")
    shock_bps = st.slider("Yield shock (bps)", -300, 300, 100, step=25)
    pct_change = estimate_price_change(dc["modified_duration"], dc["convexity"], shock_bps)
    st.metric(f"Estimated Price Change ({shock_bps:+d} bps)", f"{pct_change*100:+.2f}%",
              delta=f"${dc['price']*pct_change:+,.2f}")
    st.caption("Formula: %ΔPrice ≈ −(Modified Duration × Δy) + 0.5 × Convexity × Δy². The convexity term matters most for large yield moves.")

    st.divider()
    st.subheader("Roll-Down Analysis")
    st.caption("Estimates the 1-year return from a bond simply 'rolling down' a positively-sloped curve to a shorter maturity, holding the curve shape constant — a classic fixed income carry strategy independent of any rate view.")

    curve_for_rolldown = fetch_treasury_curve()
    tenor_map_years = {"1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5, "7Y": 7, "10Y": 10, "20Y": 20, "30Y": 30}
    available_tenors = [tenor_map_years[t] for t in curve_for_rolldown if t in tenor_map_years]

    if len(available_tenors) >= 2:
        rd_df = rolldown_analysis(curve_for_rolldown, available_tenors)
        st.dataframe(rd_df, use_container_width=True, hide_index=True)
        st.caption("Positive roll-down bps means the bond's yield falls as it rolls to the shorter tenor, which pushes its price up — this is estimated, not a live trade recommendation.")
    else:
        st.info("Not enough tenor data available on the Treasury curve right now to run roll-down analysis.")

# --- TAB 7: CANADIAN GOVERNMENT BONDS ---
with tab7:
    st.subheader("Government of Canada Bond Curve")
    st.caption("Live benchmark yields from the Bank of Canada's own Valet API — the same source used for the official policy rate.")

    cad_curve = fetch_cad_curve()
    us_curve_for_cad = fetch_treasury_curve()

    if not cad_curve:
        st.error("Couldn't fetch the Canadian curve right now — try again shortly.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(cad_curve.keys()), y=[v * 100 for v in cad_curve.values()],
                                  mode="lines+markers", name="Canada (GoC)",
                                  line=dict(color="#f87171", width=3), marker=dict(size=8)))
        fig.update_layout(xaxis_title="Tenor", yaxis_title="Yield (%)",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaed"))
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Canada vs. U.S. Spread")
        common_tenors = [t for t in ["2Y", "5Y", "10Y", "30Y"] if t in cad_curve and t in us_curve_for_cad]
        if common_tenors:
            spread_rows = [{"Tenor": t, "Canada": f"{cad_curve[t]*100:.2f}%", "U.S.": f"{us_curve_for_cad[t]*100:.2f}%",
                            "Spread (bps)": round((cad_curve[t] - us_curve_for_cad[t]) * 10000, 0)}
                           for t in common_tenors]
            st.dataframe(pd.DataFrame(spread_rows), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Canadian Roll-Down Analysis")
        tenor_map_years_cad = {"2Y": 2, "3Y": 3, "5Y": 5, "7Y": 7, "10Y": 10, "30Y": 30}
        available_cad_tenors = [tenor_map_years_cad[t] for t in cad_curve if t in tenor_map_years_cad]
        if len(available_cad_tenors) >= 2:
            rd_cad_df = rolldown_analysis(cad_curve, available_cad_tenors)
            st.dataframe(rd_cad_df, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough tenor data available on the Canadian curve right now to run roll-down analysis.")

# --- TAB 8: COMPOSITE SIGNAL ---
with tab8:
    st.subheader(f"Composite Signal — {ticker}")
    st.caption("Combines sentiment, Monte Carlo expected outcome, and stress-test resilience into one view. This is a rules-based aggregation of the other tabs, not independent research.")

    sentiment = fetch_news_sentiment(ticker)
    mc_result = st.session_state.get("_mc_result")

    if not mc_result:
        st.info("Run a Monte Carlo simulation on that tab first to include it in the composite signal.")
    else:
        expected_return = (mc_result["expected_price"] / mc_result["current_price"]) - 1
        signal_score = 0.5 * sentiment["score"] + 0.5 * max(-1, min(1, expected_return * 20))

        if signal_score > 0.3:
            verdict, color = "BULLISH", "success"
        elif signal_score < -0.3:
            verdict, color = "BEARISH", "error"
        else:
            verdict, color = "NEUTRAL", "info"

        c1, c2, c3 = st.columns(3)
        c1.metric("News Sentiment", f"{sentiment['score']:+.2f}")
        c2.metric("MC Expected Return", f"{expected_return*100:+.2f}%")
        c3.metric("Composite Score", f"{signal_score:+.2f}")

        getattr(st, color)(f"**Signal: {verdict}** for {ticker} — driven {'mostly by sentiment' if abs(sentiment['score']) > abs(expected_return*20) else 'mostly by the Monte Carlo projection'}.")
        st.caption("This signal is a simple weighted rule, built for illustration — not investment advice, and not a substitute for real research.")

# --- TAB 9: EDUCATION ---
with tab9:
    st.subheader("How Bonds Work")
    st.markdown("""
    A bond is a loan — you're lending money to a government or company, and they pay you back on a schedule.
    Every bond has four core parts:

    - **Face Value (Par Value):** the amount you get back at maturity — the $1,000 or $10,000,000 you're owed at the end
    - **Coupon Rate:** the fixed interest rate the bond pays, usually semi-annually, calculated on the face value
    - **Maturity:** how many years until the bond repays its face value and stops paying coupons
    - **Yield to Maturity (YTM):** the actual annualized return you'd earn holding the bond to maturity — this is
      *not* the same as the coupon rate once the bond trades above or below its face value

    **Why bond prices and yields move opposite each other:** a bond's coupon payments are fixed once issued. If new
    bonds start being issued at a higher rate (because rates rose), your older, lower-coupon bond becomes less
    attractive — the only way to make it competitive is for its *price* to fall, which raises its effective yield.
    The reverse happens when rates fall: your fixed coupon becomes relatively more attractive, so the price rises.
    This is the single most important relationship in fixed income: **yields up, prices down — yields down, prices up.**
    """)

    st.divider()
    st.subheader("What Each Tab Actually Does")

    with st.expander("Sentiment Analysis"):
        st.markdown("""
        Pulls recent news headlines for the selected Treasury ETF and scores them using rate/macro-relevant
        keywords (dovish/hawkish, cut/hike, recession/growth, etc.). The idea: bond markets react heavily to
        macro narrative and Fed commentary, often before it shows up in price — this is a rough, fast read on
        which way the news flow is leaning, not a replacement for reading the actual articles.
        """)

    with st.expander("Monte Carlo Simulation"):
        st.markdown("""
        Most Monte Carlo tools simulate a stock-style random walk directly on price. This one simulates random
        walks in **yield** instead (since that's the variable that actually moves independently for a bond), then
        converts each simulated yield path into a price outcome using the bond's duration and convexity. Run
        thousands of these paths and you get a full distribution of possible future prices — from which you can
        read off Value at Risk (VaR) and the probability of a loss over your chosen time horizon.
        """)

    with st.expander("Options Stress Test"):
        st.markdown("""
        Treasury ETFs have real, exchange-listed options. This tab pulls the actual live chain (real strikes,
        expiries, and implied volatility), then reprices your chosen contract under a range of underlying price
        shocks using the Black-Scholes model — showing exactly how much an option's value would change if the ETF
        moved up or down by a given amount before expiry.
        """)

    with st.expander("Swap Stress Test"):
        st.markdown("""
        Models a plain-vanilla fixed-for-floating interest rate swap: you receive a fixed rate and pay a floating
        rate (or vice versa) on a notional amount. The fixed leg is valued by discounting its cash flows using the
        real live Treasury curve; the floating leg is assumed to reprice to par (a standard simplification). The
        stress scenarios show how the swap's mark-to-market value would change if the curve shifted up, down, or
        changed shape (steepened/flattened) — exactly the kind of scenario a rates desk runs before putting on a
        position. Note: this models the *mechanics* of a swap correctly, but real swap market rates are licensed
        data this dashboard doesn't have access to — see the disclosure on that tab.
        """)

    with st.expander("Forward Stress Test"):
        st.markdown("""
        A forward contract locks in a price today for a bond delivered later. Its fair price is simply the
        current spot price adjusted for the cost of carrying that position until delivery (the repo rate, here
        proxied by SOFR). This tab shows how that forward price would change under different combinations of
        underlying price moves and repo rate moves.
        """)

    with st.expander("Fixed Income Strategy (Duration, Convexity, Roll-Down)"):
        st.markdown("""
        - **Duration** answers: "how much does this bond's price move for a 1% change in yield?" Longer maturity
          generally means higher duration means more price sensitivity.
        - **Convexity** is the correction to that estimate — the price/yield relationship is actually curved, not
          a straight line, and convexity captures how much it bends. It matters most for large yield moves.
        - **Roll-down** is a strategy, not just a risk metric: if the yield curve is upward-sloping, a bond's yield
          naturally falls as it "ages" toward a shorter maturity each year — and falling yield means rising price.
          This tab estimates that return using the live Treasury curve, independent of any view on where rates
          are headed.
        """)

    with st.expander("Composite Signal"):
        st.markdown("""
        A simple weighted combination of the sentiment score and the Monte Carlo's expected return, producing a
        single Bullish/Neutral/Bearish read. It's deliberately transparent about being a basic rule (not a
        sophisticated model) — the value is in seeing how the individual pieces roll up into one view, not in
        treating the output as investment advice.
        """)
