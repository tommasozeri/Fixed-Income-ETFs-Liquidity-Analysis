# Liquidity Transformation: NAV Smoothing Notebook

Satellite/complementary analysis for the **Advanced Fixed Income and Credit** group assignment (MSc Quantitative Finance, University of Bologna). Theme: *Market microstructure insights*.

**Research question:** if an ETF holds an illiquid underlying, how can the ETF itself be liquid?

This notebook does not describe the institutional mechanism (that is the main HMM-based analysis, covering the primary market, the secondary market, market-maker continuous hedging, and AP creation/redemption) — it tests, statistically, whether that mechanism leaves a measurable fingerprint in price data.

## Repository layout

```
satellite-analysis/
├── README.md
├── requirements.txt
│
├── data/                                   # Bundled NAV history (iShares exports)
│   ├── lqd_nav_history.csv
│   └── hyg_nav_history.csv
│
├── notebooks/                              # Builder scripts + built/executed notebooks
│   ├── build_notebook_smoothing.py             # Italian builder
│   ├── build_notebook_smoothing_en.py           # English builder
│   ├── etf_liquidity_transformation.ipynb       # Built notebook (Italian)
│   └── etf_liquidity_transformation_en.ipynb    # Built notebook (English)
│
├── reports/                                # Written reports (LaTeX source + compiled PDF)
│   ├── liquidity_transformation_report.tex/.pdf       # Detailed report (Italian)
│   ├── liquidity_transformation_report_en.tex/.pdf    # Detailed report (English)
│   └── liquidity_transformation_summary.tex/.pdf      # Short illustrated summary (Italian)
│
└── figures/                                # Figures embedded in the notebooks and the reports
    ├── smoothing_hmm_synthetic.png
    ├── smoothing_rolling_diagnostics(_en).png
    └── smoothing_real_nav_rho1(_en).png
```

`notebooks/`, `reports/`, and `data/`/`figures/` are siblings on purpose: every relative path inside the builder scripts and the `.tex` files (`../data/...`, `../figures/...`) assumes this exact layout, and both `jupyter execute`/`nbconvert` and `pdflatex` run with the working directory set to the file being processed — so as long as you invoke them from inside `notebooks/` or `reports/` respectively (as shown below), the relative paths resolve correctly.

The notebook is **generated, not hand-edited**: to change its content, edit the corresponding `build_notebook_smoothing*.py` script and rebuild — editing the `.ipynb` JSON directly will be overwritten on the next build.

## What's in the notebook

1. **The two starting models.** Before any estimation, two statistical models are fixed explicitly:
   - *NAV model* (Getmansky–Lo–Makarov, 2004): the NAV is a smoothed/stale average of true returns, $r^{NAV}_t = \sum_j \theta_j r^\ast_{t-j}$ with geometric weights $\theta_j=(1-\phi)\phi^j$. Proven result: $\rho_1(NAV) = \phi$ exactly.
   - *ETF price model*: $r^P_t = r^\ast_t + \eta_t$, i.i.d. idiosyncratic noise representing continuous secondary-market trading and market-maker hedging. Implies $\rho_1(P) \approx 0$.
   Each derivation is followed by an "In plain language" recap, written to be reused as-is in an oral presentation.
2. **Estimators.** Rolling lag-1 autocorrelation, the Lo–MacKinlay (1988) variance ratio $VR(q)$, and the **Ljung–Box test** (implemented directly with `scipy.stats.chi2`, no `statsmodels` dependency) for formal significance.
3. **Synthetic validation.** Long simulated samples with known $\phi$ confirm the estimators recover the true parameter and that Ljung–Box rejects/fails to reject $H_0$ as expected.
4. **Rolling diagnostics.** A synthetic path with a time-varying $\phi$ regime, to check the rolling estimates track regime shifts.
5. **Real data (NAV only, no network needed).** Applies the estimators to real LQD/HYG NAV history; includes a subsection ("why these numbers answer the research question") connecting the results directly back to the research question.
6. **Real data (full NAV-vs-price comparison).** Network-gated (`DEMO = False`, requires `yfinance`): downloads ETF OHLCV and computes the direct gap $\hat\rho_1(NAV) - \hat\rho_1(P)$ for the same fund/period.
7. **Conclusions.** Summary, link back to the main analysis, limitations, possible extensions.

## Requirements

```bash
pip install -r requirements.txt
# yfinance is only needed for Section 6 with DEMO = False
```

## How to run

```bash
cd notebooks

# Build
python build_notebook_smoothing.py etf_liquidity_transformation.ipynb

# Execute (either use Jupyter directly, or:)
jupyter execute --inplace etf_liquidity_transformation.ipynb
# equivalently: jupyter nbconvert --to notebook --execute --inplace etf_liquidity_transformation.ipynb

# English version
python build_notebook_smoothing_en.py etf_liquidity_transformation_en.ipynb
jupyter execute --inplace etf_liquidity_transformation_en.ipynb
```

By default `DEMO = True` in the configuration cell (Section 1): everything runs offline, using only the bundled NAV CSVs in `../data/`. Set `DEMO = False` and re-run to also execute Section 6, which needs network access and `yfinance` to download LQD/HYG OHLCV — `yf.download(ticker, period="max", ...)` is used deliberately (not the default 1-month window), since the 60-day rolling window needs the full price history to produce non-NaN estimates.

Figures are saved to `../figures/` (i.e. the `figures/` folder at the repository root, one level above `notebooks/`): `smoothing_rolling_diagnostics.png`, `smoothing_real_nav_rho1.png`, and (with `DEMO = False`) `smoothing_real_gap_<ticker>.png`. The English builder saves the same figures with an `_en` suffix, so running both language versions doesn't overwrite either set.

## Reports

```bash
cd reports
pdflatex liquidity_transformation_report.tex
pdflatex liquidity_transformation_report.tex   # second pass, resolves cross-references
```

Each report's `\includegraphics` calls point to `../figures/...`, so run `pdflatex` from inside `reports/` (as above) and make sure the notebook has been built/executed at least once first, so the figures actually exist.

## Key results (real data)

| Fund | $\hat\rho_1$(NAV), full-sample | $VR(5)$ | Ljung–Box $p$ |
|---|---|---|---|
| LQD (investment-grade) | 0.0999 | 1.2201 | < 0.0001 |
| HYG (high-yield) | 0.3409 | 1.9647 | < 0.0001 |

With `DEMO = False` (full NAV-vs-price comparison, rolling means): LQD gap $\widehat\Delta = 0.0687$, HYG gap $\widehat\Delta = 0.2175$ — roughly 3x larger for the less liquid fund, consistent with the research hypothesis. The gap narrows during 2008 and 2020, consistent with the liquidity-transformation mechanism weakening under systemic stress.

## Related files

A more detailed write-up of this same analysis, with full derivations and results tables, is available as `reports/liquidity_transformation_report.tex`/`.pdf` (Italian) and `reports/liquidity_transformation_report_en.tex`/`.pdf` (English); a shorter illustrated version is `reports/liquidity_transformation_summary.tex`/`.pdf`.
