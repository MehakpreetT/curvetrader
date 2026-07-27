# SignalR

A personal trading signaler for bond derivatives — combines news sentiment, a rates-aware Monte Carlo simulation, and stress testing across options, swaps, and forwards, all tailored to real Treasury, TIPS, and corporate bond ETFs.

Built with Python and Streamlit.

---

## What it does

Pick a bond ETF (Treasury, TIPS, or corporate), and SignalR runs it through six layers of analysis:

- **Sentiment Analysis** — scores recent news headlines using rate/macro-relevant keywords (dovish/hawkish, cut/hike, recession/growth)
- **Monte Carlo Simulation** — unlike a generic equity-style price walk, this simulates future *yield* changes and converts them to price outcomes via duration/convexity — the way a bond actually moves
- **Options Stress Test** — pulls the ETF's **real, live listed options chain** (Treasury ETFs like TLT genuinely trade exchange-listed options) and stress-tests a chosen contract's value across underlying price shocks using Black-Scholes
- **Swap Stress Test** — models a fixed-for-floating interest rate swap valued off the real live Treasury curve, stress-tested under parallel shifts and steepener/flattener scenarios
- **Forward Stress Test** — models a bond forward priced off the real ETF price and SOFR as a repo-rate proxy
- **Fixed Income Strategy** — a duration & convexity calculator for any bond you define, plus roll-down analysis on the live Treasury or Canadian government curve
- **Canadian Government Bonds** — live Government of Canada benchmark yields, a Canada-vs-U.S. spread comparison, and Canadian-specific roll-down analysis
- **Composite Signal** — combines sentiment and the Monte Carlo's expected return into a single Bullish/Neutral/Bearish call
- **Education** — explains how bonds work (face value, coupon, maturity, yield, and why price and yield move opposite each other) and what each tab is actually doing under the hood

## Covered instruments

| Ticker | What it is | Category |
|---|---|---|
| TLT | 20+ Year Treasury Bond ETF | U.S. Treasury |
| IEF | 7-10 Year Treasury Bond ETF | U.S. Treasury |
| SHY | 1-3 Year Treasury Bond ETF | U.S. Treasury |
| TIP | TIPS Bond ETF | U.S. Treasury (inflation-protected) |
| LQD | Investment-Grade Corporate Bond ETF | Corporate |
| HYG | High-Yield Corporate Bond ETF | Corporate |

Corporate bond ETFs carry credit risk on top of interest rate risk — the app flags this explicitly when you select one, since the duration/convexity math here only models rate risk.

## Methodology notes

- **Duration and convexity** shown per ETF are standard published approximations, not calculated live from each fund's exact current holdings.
- **The Monte Carlo simulation** models yield changes with a mean-reverting random walk calibrated to the ETF's historical volatility, then converts to price via the duration/convexity approximation — this is a simplification of real term-structure models (e.g., Vasicek, CIR), built for illustration.
- **Options data is real and live**, pulled directly from Yahoo Finance's actual options chains.
- **Swap and forward contracts are modeled, not live-priced.** True interest rate swap market data (ICE Swap Rate, Bloomberg, Refinitiv) is licensed and not freely available anywhere — this dashboard values a hypothetical swap using real government bond curve data instead, which is mechanically sound but not a live market quote. This is disclosed directly in the app.
- **All dollar inputs are capped at $10,000,000.**

## Tech stack

- **Frontend/framework:** Streamlit, custom dark theme
- **Data:** yfinance (prices, options chains, news), pandas-datareader + FRED (Treasury curve, SOFR), Bank of Canada Valet API (Canadian government bond yields)
- **Analysis:** pandas, numpy, scipy (Black-Scholes via `norm.cdf`)
- **Visualization:** Plotly

## Running locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploying

1. Push this repo to GitHub
2. Deploy on [Streamlit Community Cloud](https://share.streamlit.io), pointing at `streamlit_app.py`
3. Streamlit Cloud auto-redeploys on every push to the connected branch

No login/account system — the app is stateless and works immediately once deployed.

## Known limitations

- **Not investment advice.** The Composite Signal is a simple, transparent weighted rule built for illustration — not a substitute for real research.
- **Swap and forward valuations are modeled**, not sourced from live market quotes (see Methodology above).
- **Corporate bond credit risk is not modeled** — only interest rate risk is captured in the duration/convexity and Monte Carlo calculations.
