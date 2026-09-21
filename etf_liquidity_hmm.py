"""
ETF Liquidity Regime Detection via Hidden Markov Model
=======================================================
Group assignment - theme: "Market microstructure insights"

Idea
----
Bond ETF liquidity is not constant: market makers arbitrage the ETF price
against the underlying bond basket in "normal" conditions, but during stress
episodes the underlying bonds stop trading, market makers step back, and the
ETF starts trading at a discount with wide intraday swings. This script uses
a 2-state Gaussian Hidden Markov Model (HMM) to classify each trading day,
objectively, into a "Normal" or "Shock" liquidity regime, using only
observable price/volume data (no subjective threshold rules).

Pipeline
--------
1. Data ingestion      -> daily OHLCV for a bond ETF (yfinance, or a local CSV)
2. Feature engineering -> liquidity/volatility proxies built from OHLCV alone
3. HMM fit              -> Gaussian HMM, 2 states, Baum-Welch (EM) via hmmlearn
4. State relabeling     -> order states by stress level so state 1 = Shock always
5. Decoding             -> full-sample Viterbi (retrospective, NOT a live signal)
6. Visualization        -> ETF price with shock-regime periods shaded

Data note
---------
Yahoo Finance gives daily OHLCV only - no real intraday bid-ask spread or
order-book depth. The features below are *proxies* built to approximate
liquidity stress from OHLCV:
  - Parkinson intraday volatility (from the High/Low range)
  - Amihud illiquidity ratio (price impact per unit of dollar volume)
  - Volume z-score (abnormal trading activity vs. its own recent history)
If you have Bloomberg/Refinitiv access, swap in real bid-ask spread and
market-depth data in `build_features()` - the rest of the pipeline is
unchanged.

Usage
-----
    python etf_liquidity_hmm.py --ticker LQD --start 2015-01-01
    python etf_liquidity_hmm.py --csv my_lqd_data.csv
    python etf_liquidity_hmm.py --demo          # synthetic data, no network needed

The --demo mode generates synthetic OHLCV data from a *known* underlying
2-state Markov chain, so you can sanity-check that the pipeline recovers
regimes that look like the ground truth before trusting it on real data.
"""

from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------------
# Palette (validated categorical pair - see dataviz skill / palette.md)
# ----------------------------------------------------------------------------
COLOR_PRICE = "#2a78d6"      # categorical slot 1 (blue)  - the price line
COLOR_SHOCK = "#eb6834"      # categorical slot 2 (orange) - shock-regime shading
COLOR_GRID = "#e1e0d9"       # hairline gridline
COLOR_AXIS = "#898781"       # muted axis/labels
COLOR_INK = "#0b0b0b"        # primary ink (title, price line already uses slot 1)
SURFACE = "#fcfcfb"          # chart surface

# Sequential blue ramp (steps 250->650 from the dataviz palette), used for the
# state-colored scatter + probability chart: regime severity is ORDINAL
# (Normal < Elevated < ... < Shock), so one hue light->dark is the correct
# encoding - never separate categorical hues for an ordered scale.
SEQUENTIAL_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
                   "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]


def ramp_colors(n_states: int) -> list[str]:
    """Evenly-spaced colors from the sequential ramp, one per state, ordered
    light (calm) -> dark (severe)."""
    idx = np.linspace(0, len(SEQUENTIAL_RAMP) - 1, n_states).round().astype(int)
    return [SEQUENTIAL_RAMP[i] for i in idx]


# ----------------------------------------------------------------------------
# 1. Data ingestion
# ----------------------------------------------------------------------------
def load_from_yfinance(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """Daily OHLCV via yfinance. Requires outbound internet access to Yahoo
    Finance - this will fail in network-restricted sandboxes; run it on your
    own machine or in Colab if that happens."""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(
            f"No data returned for {ticker}. Check the ticker, date range, "
            "and that this environment has outbound network access to Yahoo Finance."
        )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index.name = "Date"
    return df


def load_bidask_from_wrds(ticker: str, start: str, end: str | None, wrds_username: str) -> pd.Series:
    """Daily bid-ask spread (or another liquidity measure) from WRDS -
    SKELETON, not runnable as-is. Fill in the exact library/table/column
    names from the WRDS Query Tool (wrds-www.wharton.upenn.edu) for TAQ /
    "WRDS Intraday Indicators" (pre-aggregated daily liquidity measures,
    much simpler than raw NBBO tick data, which is one table per day) -
    the Query Tool's web form generates the correct current Python for
    your subscription, since WRDS reorganizes table names periodically.

    Returns a daily Series of the liquidity measure, indexed by date, to
    be merged into build_features() as a real replacement for the Amihud
    proxy (see the merge point marked below).

    Needs: pip install wrds   (then wrds.Connection() will prompt for your
    WRDS credentials, or pass wrds_username explicitly).
    """
    import wrds

    db = wrds.Connection(wrds_username=wrds_username)

    # --- EXAMPLE SHAPE ONLY - replace with the Query Tool's generated SQL ---
    # query = f"""
    #     SELECT date, <bid_ask_spread_column>
    #     FROM <library>.<table>
    #     WHERE ticker = '{ticker}'
    #       AND date BETWEEN '{start}' AND '{end or pd.Timestamp.today().date()}'
    #     ORDER BY date
    # """
    # df = db.raw_sql(query, date_cols=["date"])
    # db.close()
    # return df.set_index("date")["<bid_ask_spread_column>"].rename("bidask_spread")

    raise NotImplementedError(
        "Fill in the query above using the WRDS Query Tool's generated code "
        "for TAQ / WRDS Intraday Indicators, then remove this line."
    )


def load_nav_csv(path: str) -> pd.Series:
    """Loads a cleaned (Date, NAV) CSV - e.g. extracted from iShares' fund
    page "Historical" export (Date, NAV per Share, ...). Returns a daily
    NAV Series indexed by date, used by premium_discount_feature() below.
    This is the REAL liquidity-stress signal, not a proxy: when the ETF's
    creation/redemption arbitrage breaks down (dealers step back, the
    underlying bonds stop trading), the market price decouples from NAV -
    that decoupling IS the phenomenon this project is about."""
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.set_index("Date")["NAV"].sort_index()


def premium_discount_feature(close: pd.Series, nav: pd.Series) -> pd.Series:
    """|Premium/Discount to NAV| = |Close - NAV| / NAV, in percent. Absolute
    value because for liquidity-stress detection what matters is the SIZE
    of the dislocation, not its sign (a fund can trade above or below NAV
    depending on the direction of stress/flows) - same convention as the
    other non-negative stress features (park_vol, amihud)."""
    nav_aligned = nav.reindex(close.index).ffill()
    return ((close - nav_aligned).abs() / nav_aligned) * 100


def load_from_csv(path: str) -> pd.DataFrame:
    """Local CSV fallback - same schema as a Yahoo Finance / stooq export:
    Date, Open, High, Low, Close, Volume (Date as the first column or index)."""
    df = pd.read_csv(path)
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index.name = "Date"
    return df


def generate_demo_data(n_days: int = 2500, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic OHLCV generated from a KNOWN 2-state Markov chain, purely to
    verify the pipeline end-to-end without needing network access. Returns
    (ohlcv_df, true_state_series) - the true states are for validation only
    and are never shown to the HMM.

    Regime 0 (Normal): low return volatility, narrow intraday range, stable volume.
    Regime 1 (Shock):  high return volatility, wide intraday range, volume spikes.
    """
    rng = np.random.default_rng(seed)

    # True transition matrix: both regimes are persistent (stress is a "phase",
    # not day-to-day noise), shock regime is rarer and shorter-lived.
    P = np.array([[0.985, 0.015],
                  [0.070, 0.930]])

    true_state = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        true_state[t] = rng.choice([0, 1], p=P[true_state[t - 1]])

    # Regime-dependent daily return distribution
    mu = np.where(true_state == 0, 0.0001, -0.0008)
    sigma = np.where(true_state == 0, 0.0030, 0.0150)
    daily_ret = rng.normal(mu, sigma)

    close = 100 * np.cumprod(1 + daily_ret)
    open_ = np.empty(n_days)
    open_[0] = 100.0
    open_[1:] = close[:-1]

    # Intraday range wider in the shock regime (liquidity stress -> wide H-L)
    range_pct = np.where(true_state == 0,
                          rng.uniform(0.001, 0.004, n_days),
                          rng.uniform(0.010, 0.035, n_days))
    high = np.maximum(open_, close) * (1 + range_pct / 2)
    low = np.minimum(open_, close) * (1 - range_pct / 2)

    # Volume: log-normal, with spikes (market makers pulling back -> erratic
    # volume) and a lower base level in stress
    base_vol = np.where(true_state == 0, 6_000_000, 9_000_000)
    volume = rng.lognormal(mean=np.log(base_vol), sigma=np.where(true_state == 0, 0.15, 0.55))

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    df.index.name = "Date"
    return df, pd.Series(true_state, index=dates, name="true_state")


# ----------------------------------------------------------------------------
# 2. Feature engineering (liquidity/volatility proxies from OHLCV only)
# ----------------------------------------------------------------------------
def parkinson_volatility(df: pd.DataFrame) -> pd.Series:
    """Parkinson (1980) intraday volatility estimator from the High/Low range.
    Uses more information than close-to-close volatility because it exploits
    the whole day's trading range, which is exactly what widens when a
    market becomes hard to trade."""
    hl_ratio = np.log(df["High"] / df["Low"])
    return np.sqrt((hl_ratio ** 2) / (4 * np.log(2)))


def amihud_illiquidity(df: pd.DataFrame) -> pd.Series:
    """Amihud (2002) illiquidity ratio: |return| per unit of dollar volume
    traded. High value = a given amount of trading moves the price a lot,
    i.e. the market has little depth - the classic price-impact proxy."""
    ret = df["Close"].pct_change()
    dollar_volume = df["Close"] * df["Volume"]
    illiq = ret.abs() / dollar_volume
    return illiq


def volume_zscore(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """Rolling z-score of daily volume: how abnormal is today's trading
    activity relative to its own recent history."""
    vol = df["Volume"]
    return (vol - vol.rolling(window).mean()) / vol.rolling(window).std()


def build_features(df: pd.DataFrame, smooth_window: int = 5, z_window: int = 252,
                    extra_features: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Builds the observation features and standardizes them (rolling
    z-score) so the Gaussian HMM sees comparable scales. A short rolling
    mean is applied first to Parkinson vol and Amihud illiquidity (and any
    extra_features passed in) because they're notoriously noisy day-to-day
    (typical practice in the liquidity literature).

    extra_features: optional {name: Series} of additional real (non-proxy)
    liquidity measures, aligned to df's index by the caller - e.g. the
    |premium/discount to NAV| from premium_discount_feature(). Smoothed and
    standardized the same way as the built-in features, then dropped via
    --drop-features if it doesn't end up pulling its weight (use --n-states
    with BIC to check, same workflow as we used for vol_z)."""
    feat = pd.DataFrame(index=df.index)
    feat["park_vol"] = parkinson_volatility(df).rolling(smooth_window).mean()
    feat["amihud"] = amihud_illiquidity(df).rolling(smooth_window).mean()
    feat["vol_z"] = volume_zscore(df)
    for name, series in (extra_features or {}).items():
        feat[name] = series.reindex(df.index).rolling(smooth_window).mean()

    feat = feat.dropna()

    # Rolling standardization (expanding after the first z_window days) so
    # the feature scale is comparable across the whole sample without
    # leaking full-sample statistics into early observations.
    z = pd.DataFrame(index=feat.index, columns=feat.columns, dtype=float)
    for col in feat.columns:
        roll_mean = feat[col].rolling(z_window, min_periods=60).mean()
        roll_std = feat[col].rolling(z_window, min_periods=60).std()
        z[col] = (feat[col] - roll_mean) / roll_std
    z = z.dropna()
    return z


# ----------------------------------------------------------------------------
# 3-4-5. HMM fit, state relabeling, Viterbi decoding
# ----------------------------------------------------------------------------
def fit_hmm(X: np.ndarray, n_states: int = 2, n_init: int = 10, seed: int = 0):
    """Fits a Gaussian HMM with full covariance via Baum-Welch (EM). Because
    EM only finds a local optimum, we try several random initializations and
    keep the one with the highest log-likelihood."""
    from hmmlearn.hmm import GaussianHMM

    best_model, best_ll = None, -np.inf
    for i in range(n_init):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=1000,
            tol=1e-4,
            random_state=seed + i,
        )
        model.fit(X)
        ll = model.score(X)
        if ll > best_ll:
            best_ll, best_model = ll, model
    return best_model, best_ll


def bic_score(model, X: np.ndarray, n_states: int) -> float:
    """Bayesian Information Criterion for a fitted GaussianHMM (full
    covariance). Used to choose the number of states empirically instead of
    guessing: lower BIC is better. Free parameters counted: (k-1) initial
    probs + k(k-1) transition probs + k*d means + k*d(d+1)/2 covariance
    entries (full, symmetric) per state."""
    n_obs, d = X.shape
    n_params = (n_states - 1) + n_states * (n_states - 1) + n_states * d + n_states * d * (d + 1) // 2
    ll = model.score(X)
    return -2 * ll + n_params * np.log(n_obs)


def relabel_states_by_stress(model, X: np.ndarray) -> np.ndarray:
    """hmmlearn assigns state indices arbitrarily (state 0 is not
    necessarily "calm"). We relabel so that state indices are ordered by a
    composite stress score (mean of the standardized features) ascending:
    state 0 = lowest stress = Normal, last state = highest stress = Shock.
    This keeps the labeling consistent across ticker/reruns."""
    stress_score = model.means_.mean(axis=1)  # mean across the 3 features, per state
    order = np.argsort(stress_score)  # ascending: calmest state first
    remap = {old: new for new, old in enumerate(order)}
    return remap


def decode_regimes(model, X: np.ndarray, remap: dict) -> np.ndarray:
    """Full-sample Viterbi decoding: the single most likely state sequence
    given ALL the data. This is the right tool for a retrospective,
    descriptive classification ("when did the market experience liquidity
    stress, historically") - it is NOT appropriate for a live trading
    signal, which would need forward-only filtering instead (see the
    duration-hedging pipeline for that distinction)."""
    raw_states = model.predict(X)  # Viterbi under the hood for GaussianHMM
    return np.array([remap[s] for s in raw_states])


def apply_min_duration_filter(states: pd.Series, min_duration: int = 5) -> pd.Series:
    """Post-processing hysteresis filter: merges any contiguous regime run
    shorter than `min_duration` days into whichever neighboring run is
    longer. This does NOT refit the HMM - Baum-Welch/Viterbi already give
    the best day-by-day state path; this is a smoothing step on top of it,
    standard practice when the decoded regime is meant to be read as a
    "phase" (weeks of liquidity stress) rather than a tick-by-tick label
    that can flicker for a single unusually volatile day. Runs touching
    the start/end of the sample (no neighbor on one side) merge into
    whichever neighbor exists; if the whole array is one run, or has no
    runs under the threshold, it is returned unchanged.

    Use this if you see the HMM decoding many isolated 1-3 day "shock"
    spikes scattered through otherwise calm periods, and you want the
    regime series to reflect sustained episodes instead."""
    s = states.copy()
    while True:
        groups = (s != s.shift()).cumsum()
        run_lengths = s.groupby(groups).transform("size")
        if not (run_lengths < min_duration).any():
            break
        # merge the FIRST short run found, then recompute (indices/groups
        # shift after every merge, so we restart the scan each time)
        short_group_id = groups[run_lengths < min_duration].iloc[0]
        seg_index = groups[groups == short_group_id].index
        start_pos = s.index.get_loc(seg_index[0])
        end_pos = s.index.get_loc(seg_index[-1])

        prev_state = s.iloc[start_pos - 1] if start_pos > 0 else None
        next_state = s.iloc[end_pos + 1] if end_pos < len(s) - 1 else None

        if prev_state is None and next_state is None:
            break  # single run covering the whole series - nothing to merge into
        elif prev_state is None:
            new_state = next_state
        elif next_state is None:
            new_state = prev_state
        else:
            prev_run_len = (groups == groups.iloc[start_pos - 1]).sum()
            next_run_len = (groups == groups.iloc[end_pos + 1]).sum()
            new_state = prev_state if prev_run_len >= next_run_len else next_state

        s.iloc[start_pos:end_pos + 1] = new_state
    return s


def compute_state_probabilities(model, X: np.ndarray, remap: dict) -> np.ndarray:
    """Smoothed posterior state probabilities P(state=k | ALL the data),
    via the forward-backward algorithm (hmmlearn's predict_proba) - the
    same full-sample logic as the Viterbi decode used elsewhere here, so
    the two stay consistent (this is retrospective, not a live filtering
    signal - see decode_regimes docstring). Columns are reordered through
    `remap` to match the stress-ordered state labels used everywhere else
    in this script."""
    raw_probs = model.predict_proba(X)  # (n_obs, n_states)
    probs = np.zeros_like(raw_probs)
    for old, new in remap.items():
        probs[:, new] = raw_probs[:, old]
    return probs


# ----------------------------------------------------------------------------
# 6. Visualization
# ----------------------------------------------------------------------------
def plot_regimes(price: pd.Series, states: pd.Series, ticker: str, out_path: str, n_states: int = 2):
    """ETF price line with elevated/shock regime periods shaded in the
    background. One axis, one series (price) in a fixed categorical color;
    the regime is encoded by background shading (severity as a single-hue
    ramp, since states are ordered by stress) + legend, never by
    recoloring the price line itself."""
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # States are already ordered by stress (0 = calmest). Shade every
    # non-zero state with increasing alpha so severity reads as intensity,
    # not as separate hues - this generalizes cleanly beyond 2 states.
    max_state = n_states - 1
    alpha_max = 0.35
    groups = (states != states.shift()).cumsum()
    for _, seg in states.groupby(groups):
        s = seg.iloc[0]
        if s > 0:
            alpha = alpha_max * (s / max_state)
            ax.axvspan(seg.index[0], seg.index[-1], color=COLOR_SHOCK, alpha=alpha, lw=0)

    ax.plot(price.index, price.values, color=COLOR_PRICE, linewidth=1.4, zorder=3)

    ax.set_title(f"{ticker} - liquidity regimes decoded by a {n_states}-state HMM",
                 color=COLOR_INK, fontsize=13, fontweight="bold", pad=12, loc="left")
    ax.set_ylabel("Price ($)", color=COLOR_AXIS, fontsize=10)
    ax.tick_params(colors=COLOR_AXIS, labelsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(COLOR_GRID)
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8, zorder=0)

    legend_handles = [plt.Line2D([0], [0], color=COLOR_PRICE, lw=1.6, label=f"{ticker} close price")]
    for s in range(1, n_states):
        alpha = alpha_max * (s / max_state)
        label = f"Shock regime (state {s})" if s == max_state else f"Elevated regime (state {s})"
        legend_handles.append(Patch(facecolor=COLOR_SHOCK, alpha=alpha, label=label))
    ax.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def plot_regimes_detailed(price: pd.Series, states: pd.Series, probs: np.ndarray,
                           ticker: str, out_path: str, n_states: int = 2):
    """Two-panel diagnostic chart, x-axis shared (small multiples, not a
    dual-axis chart):
      top    - price as a scatter, each point colored by its decoded
               (Viterbi) regime, on a single-hue sequential ramp ordered by
               severity, with a thin muted line underneath for path
               readability;
      bottom - the HMM's smoothed state probabilities as a 100% stacked
               area, same ramp/order, same color meaning - shows how
               CONFIDENT the model is day by day, not just the hard label.
    """
    colors = ramp_colors(n_states)
    labels = []
    for k in range(n_states):
        if k == 0:
            labels.append("Normal (state 0)")
        elif k == n_states - 1:
            labels.append(f"Shock (state {k})")
        else:
            labels.append(f"Elevated (state {k})")

    fig, (ax_price, ax_prob) = plt.subplots(
        2, 1, figsize=(11, 7.5), dpi=150, sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1]},
    )
    fig.patch.set_facecolor(SURFACE)

    # --- top panel: colored scatter over a thin path line ---
    ax_price.set_facecolor(SURFACE)
    ax_price.plot(price.index, price.values, color=COLOR_GRID, linewidth=0.8, zorder=1)
    for k in range(n_states):
        mask = states.values == k
        ax_price.scatter(price.index[mask], price.values[mask],
                          color=colors[k], s=7, linewidths=0, zorder=2, label=labels[k])
    ax_price.set_title(f"{ticker} - regime-colored price and HMM state probabilities ({n_states} states)",
                        color=COLOR_INK, fontsize=13, fontweight="bold", pad=12, loc="left")
    ax_price.set_ylabel("Price ($)", color=COLOR_AXIS, fontsize=10)
    ax_price.tick_params(colors=COLOR_AXIS, labelsize=9)
    for spine in ["top", "right"]:
        ax_price.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax_price.spines[spine].set_color(COLOR_GRID)
    ax_price.grid(axis="y", color=COLOR_GRID, linewidth=0.8, zorder=0)
    ax_price.legend(loc="upper left", frameon=False, fontsize=9, markerscale=2)

    # --- bottom panel: 100% stacked posterior probabilities ---
    ax_prob.set_facecolor(SURFACE)
    ax_prob.stackplot(price.index, probs.T, colors=colors, linewidth=0, labels=labels)
    ax_prob.set_ylabel("P(state)", color=COLOR_AXIS, fontsize=10)
    ax_prob.set_ylim(0, 1)
    ax_prob.tick_params(colors=COLOR_AXIS, labelsize=9)
    for spine in ["top", "right"]:
        ax_prob.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax_prob.spines[spine].set_color(COLOR_GRID)
    ax_prob.grid(axis="y", color=COLOR_GRID, linewidth=0.6, zorder=0)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Summary stats (useful directly for the report / slides)
# ----------------------------------------------------------------------------
def summarize_regimes(states: pd.Series, feat: pd.DataFrame, model) -> str:
    lines = []
    n = len(states)
    for k in sorted(states.unique()):
        mask = states == k
        pct = mask.mean() * 100
        label = "Normal" if k == 0 else "Shock" if k == states.max() else f"State {k}"
        lines.append(f"  Regime {k} ({label}): {pct:5.1f}% of days ({mask.sum()} / {n})")

    # expected regime duration from the (relabeled) transition matrix diagonal:
    # E[duration in state i] = 1 / (1 - P_ii)
    lines.append("\nExpected regime persistence (avg. consecutive days):")
    groups = (states != states.shift()).cumsum()
    durations = states.groupby(groups).agg(["first", "size"])
    for k in sorted(states.unique()):
        avg_dur = durations.loc[durations["first"] == k, "size"].mean()
        lines.append(f"  Regime {k}: {avg_dur:.1f} days on average")

    lines.append("\nMean feature values by regime (standardized, z-scores):")
    feat_aligned = feat.loc[states.index]
    for k in sorted(states.unique()):
        means = feat_aligned[states == k].mean()
        lines.append(f"  Regime {k}: " + ", ".join(f"{c}={v:+.2f}" for c, v in means.items()))

    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="ETF liquidity regime detection via HMM")
    p.add_argument("--ticker", default="LQD", help="ETF ticker (default: LQD)")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--csv", default=None, help="Local OHLCV CSV instead of yfinance")
    p.add_argument("--demo", action="store_true", help="Synthetic data, no network needed")
    p.add_argument("--out", default=".", help="Output directory")
    p.add_argument("--n-states", default="2", help="Comma-separated candidates to compare via BIC, "
                                                     "e.g. '2,3'. With one value, that value is used directly.")
    p.add_argument("--drop-features", default=None,
                    help="Comma-separated feature names to exclude, e.g. 'vol_z' if it isn't "
                         "discriminating between regimes for your ticker.")
    p.add_argument("--min-duration", type=int, default=1,
                    help="Merge any decoded regime run shorter than this many days into its "
                         "longer neighbor (hysteresis filter). Default 1 = no filtering. "
                         "Use e.g. 5 to remove single-day 'shock' flickers and keep only "
                         "sustained episodes.")
    p.add_argument("--nav-csv", default=None,
                    help="Path to a (Date, NAV) CSV - e.g. extracted from the iShares fund page's "
                         "'Historical' export - to add |premium/discount to NAV| as a REAL "
                         "(non-proxy) liquidity feature, computed against the downloaded Close price.")
    # parse_known_args (not parse_args) so this also runs unmodified inside
    # Jupyter/Colab, which injects its own kernel launcher arguments
    # (e.g. "-f /root/.../kernel-....json") into sys.argv.
    args, _unknown = p.parse_known_args()

    true_state = None
    if args.demo:
        print("Running in --demo mode: synthetic OHLCV from a known 2-state Markov chain.")
        df, true_state = generate_demo_data()
        ticker_label = "DEMO"
    elif args.csv:
        print(f"Loading local CSV: {args.csv}")
        df = load_from_csv(args.csv)
        ticker_label = args.ticker
    else:
        print(f"Downloading {args.ticker} from Yahoo Finance ({args.start} to {args.end or 'today'})...")
        df = load_from_yfinance(args.ticker, args.start, args.end)
        ticker_label = args.ticker

    print(f"Loaded {len(df)} trading days: {df.index[0].date()} to {df.index[-1].date()}")

    extra_features = {}
    if args.nav_csv:
        nav = load_nav_csv(args.nav_csv)
        prem_disc = premium_discount_feature(df["Close"], nav)
        coverage = prem_disc.notna().mean() * 100
        print(f"Loaded NAV from {args.nav_csv}: {coverage:.1f}% date coverage against price history")
        extra_features["prem_disc"] = prem_disc

    feat = build_features(df, extra_features=extra_features)
    if args.drop_features:
        drop_cols = [c.strip() for c in args.drop_features.split(",")]
        feat = feat.drop(columns=[c for c in drop_cols if c in feat.columns])
        print(f"Dropped feature(s): {drop_cols}")
    print(f"Feature matrix: {feat.shape[0]} days x {feat.shape[1]} features "
          f"({', '.join(feat.columns)})")

    X = feat.values
    candidates = [int(k.strip()) for k in args.n_states.split(",")]

    fitted = {}
    print("\n--- Model comparison ---")
    for k in candidates:
        model_k, ll_k = fit_hmm(X, n_states=k, n_init=10)
        bic_k = bic_score(model_k, X, k)
        fitted[k] = (model_k, ll_k, bic_k)
        print(f"  k={k}: log-likelihood={ll_k:.1f}, BIC={bic_k:.1f}")

    best_k = min(fitted, key=lambda k: fitted[k][2])
    chosen_k = best_k if len(candidates) > 1 else candidates[0]
    if len(candidates) > 1:
        print(f"Lowest BIC: k={best_k} (using this one below; override by passing a single --n-states value)")
    model, ll, _ = fitted[chosen_k]
    print(f"\nUsing k={chosen_k} states. Best HMM log-likelihood: {ll:.1f}")

    remap = relabel_states_by_stress(model, X)
    states = pd.Series(decode_regimes(model, X, remap), index=feat.index, name="regime")

    if args.min_duration > 1:
        raw_states = states
        states = apply_min_duration_filter(states, args.min_duration)
        n_changed = (raw_states != states).sum()
        print(f"\nMin-duration filter (>= {args.min_duration} days): reassigned {n_changed} "
              f"days ({n_changed / len(states) * 100:.1f}% of sample) out of short flickers.")

    print("\n--- Regime summary ---")
    print(summarize_regimes(states, feat, model))

    if true_state is not None:
        # Only meaningful in --demo mode, where we know the ground truth.
        aligned_truth = true_state.loc[states.index]
        agreement = (aligned_truth.values == states.values).mean()
        # states could be inverted relative to the ground-truth labeling convention
        agreement = max(agreement, 1 - agreement)
        print(f"\n[demo check] Agreement with the TRUE simulated regime: {agreement * 100:.1f}%")

    out_path = f"{args.out}/{ticker_label.lower()}_liquidity_regimes.png"
    plot_regimes(df["Close"].loc[states.index], states, ticker_label, out_path, n_states=chosen_k)
    print(f"\nChart saved to: {out_path}")

    probs = compute_state_probabilities(model, X, remap)
    detail_path = f"{args.out}/{ticker_label.lower()}_liquidity_regimes_detailed.png"
    plot_regimes_detailed(df["Close"].loc[states.index], states, probs,
                           ticker_label, detail_path, n_states=chosen_k)
    print(f"Detailed chart (colored datapoints + HMM probabilities) saved to: {detail_path}")


if __name__ == "__main__":
    # main() has no exit code to return, and sys.exit() raises SystemExit,
    # which Jupyter/Colab surfaces as a spurious "An exception has occurred"
    # message even on a clean run - so just call it directly.
    main()
