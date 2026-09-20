# Fixed-Income-ETFs-Liquidity-Analysis
Fixed Income ETFs trade liquidly despite illiquid holdings because NAV and exchange price arise from separate processes — evidenced statistically on LQD and HYG, though the effect weakens under stress.

# ETF Liquidity: Regime Detection & Liquidity Transformation

Group assignment for **Advanced Fixed Income and Credit** (MSc Quantitative Finance, University of Bologna). Theme: *Market microstructure insights*.

Two complementary analyses on bond ETF liquidity, built around LQD (investment-grade credit) and HYG (high-yield credit):

1. **Main analysis — Regime Detection.** A Gaussian Hidden Markov Model classifies each trading day into a "Normal" or "Shock" liquidity regime from OHLCV-derived features, and relates regime shocks to the ETF premium/discount over NAV. Research question: *when does the arbitrage link between the ETF price and its underlying basket break down?*
2. **Satellite analysis — Liquidity Transformation.** An econometric test (NAV-smoothing model + Ljung–Box significance testing) of a logically prior question: *if an ETF holds an illiquid underlying, how can the ETF itself be liquid?* The two analyses are designed to be presented together — the main analysis explains the *mechanism* (primary/secondary market, market-maker continuous hedging, AP creation/redemption), the satellite analysis supplies the *statistical evidence* that the mechanism is at work.

Both analyses come in Italian and English (notebooks, LaTeX reports, and PDFs).

## Repository layout

```
.
├── README.md
│
├── main-analysis/                          # Regime detection via HMM
│   ├── etf_liquidity_hmm.py                # CLI script: data -> features -> HMM -> plots
│   ├── etf_liquidity_hmm.ipynb             # Same pipeline as a notebook (Italian)
│   ├── etf-liquidity-hmm-report.tex/.pdf   # Written report (English)
│   └── etf-liquidity-hmm-report-it.tex/.pdf# Written report (Italian)
│
├── satellite-analysis/                     # Liquidity transformation / NAV smoothing
│   ├── build_notebook_smoothing.py         # Notebook builder (Italian)
│   ├── build_notebook_smoothing_en.py      # Notebook builder (English)
│   ├── etf_liquidity_transformation.ipynb      # Built notebook (Italian)
│   ├── etf_liquidity_transformation_en.ipynb   # Built notebook (English)
│   ├── liquidity_transformation_report.tex/.pdf     # Detailed report (Italian)
│   ├── liquidity_transformation_report_en.tex/.pdf  # Detailed report (English)
│   └── liquidity_transformation_summary.tex/.pdf    # Short illustrated summary (Italian)
│
├── data/
│   ├── lqd_nav_history.csv                 # LQD historical NAV (iShares export)
│   └── hyg_nav_history.csv                 # HYG historical NAV (iShares export)
│
└── report_figs/                            # Figures embedded in the LaTeX reports
```

The file paths above are the intended layout; as delivered, all files sit flat in one folder — reorganize into these subfolders (and update the `\includegraphics{report_figs/...}` paths in the `.tex` files and the CSV paths in the notebook builders accordingly) if you split the repo this way.

## Research questions

- **Main analysis:** the ETF premium/discount, $\pi_t = P_t - NAV_t$, widens when Authorized Participants stop arbitraging the ETF against its basket. The HMM classifies days into liquidity regimes from price/volume alone, without hand-set thresholds, and the regimes line up with known stress episodes (2008, 2020) and with premium/discount behavior.
- **Satellite analysis:** if the underlying basket is illiquid, the NAV is computed from stale ("matrix-priced") bonds and should show positive serial autocorrelation (Getmansky–Lo–Makarov smoothing, $\rho_1 = \phi$). If the ETF's own exchange price is genuinely liquid — formed continuously on the order-driven secondary market, with market makers hedging inventory risk on liquid proxies rather than the underlying bonds — it should not inherit that autocorrelation. The gap $\hat\rho_1(NAV) - \hat\rho_1(P)$, tested for significance with Ljung–Box, is the empirical answer.

## Data

- **OHLCV** for LQD / HYG: fetched via `yfinance` (main analysis; also used in the satellite analysis's optional network extension). Not bundled — downloaded at run time.
- **Historical NAV** for LQD / HYG: bundled as CSV under `data/` (exported from iShares fund data), used by the satellite analysis so the core results reproduce with no network access.

## Requirements

```
python >= 3.10
numpy
pandas
matplotlib
scipy
hmmlearn        # main analysis only (Gaussian HMM / Baum-Welch)
yfinance        # optional: real-price extensions in both analyses
nbformat
nbclient        # or: jupyter nbconvert --execute
```

```bash
pip install numpy pandas matplotlib scipy hmmlearn yfinance nbformat nbclient
```

## Reproducing the results

**Main analysis** (regime detection):
```bash
python main-analysis/etf_liquidity_hmm.py --demo              # synthetic data, sanity check
python main-analysis/etf_liquidity_hmm.py --ticker LQD --start 2015-01-01
```
Key CLI flags: `--ticker`, `--start`/`--end`, `--csv` (local OHLCV instead of `yfinance`), `--nav-csv`, `--n-states` (BIC model selection over candidate state counts), `--out`.

**Satellite analysis** (liquidity transformation notebook):
```bash
# Build and execute the Italian notebook
python satellite-analysis/build_notebook_smoothing.py etf_liquidity_transformation.ipynb
jupyter nbconvert --to notebook --execute --inplace etf_liquidity_transformation.ipynb

# English version
python satellite-analysis/build_notebook_smoothing_en.py etf_liquidity_transformation_en.ipynb
jupyter nbconvert --to notebook --execute --inplace etf_liquidity_transformation_en.ipynb
```
By default the notebook runs with `DEMO = True` (synthetic validation + real NAV-only comparison, no network required). Set `DEMO = False` in the configuration cell and re-run to also download OHLCV via `yfinance` and compute the full real NAV-vs-price comparison — this requires network access and `yfinance` installed locally.

**LaTeX reports** (either language):
```bash
pdflatex liquidity_transformation_report.tex
pdflatex liquidity_transformation_report.tex   # second pass, resolves cross-references
```
The report's figures are produced by running the corresponding notebook first (`report_figs/*.png`).

## Key results (real data, LQD vs HYG)

| Fund | $\hat\rho_1$(NAV) | Ljung–Box $p$ | $\hat\rho_1$(ETF price) | Gap $\widehat\Delta$ |
|---|---|---|---|---|
| LQD (investment-grade) | 0.0999 | < 0.0001 | −0.0454 | 0.0687 |
| HYG (high-yield) | 0.3409 | < 0.0001 | −0.0051 | 0.2175 |

Both funds show highly significant NAV autocorrelation (evidence of underlying illiquidity), essentially no autocorrelation in the ETF's own price (evidence the wrapper is trading like a liquid asset), and a gap between the two roughly 3x larger for the less liquid high-yield fund — consistent with the research hypothesis. The gap also narrows during systemic stress (2008, 2020), consistent with the transformation mechanism weakening precisely when market makers find continuous hedging harder to sustain.

## Notes

- Both notebook builders (`build_notebook_smoothing*.py`) generate the `.ipynb` programmatically via `nbformat` rather than being hand-edited notebooks — edit the builder script and rebuild, rather than editing the `.ipynb` JSON directly.
- The Ljung–Box test is implemented directly (`scipy.stats.chi2`) with no `statsmodels` dependency.
- `report_figs/` filenames ending in `_en` are the English-labeled duplicates of the Italian figures, kept separate so neither language's report clobbers the other's figures when both notebooks are re-run.
