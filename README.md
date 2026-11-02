# BazaarTalks — Global Market Scanners & Multi-Market Quant Research

A research platform for multi-market equity screening, point-in-time backtesting and factor research.
It scans the US, India, Japan, Korea and Europe for Darvas-box breakouts confirmed by Piotroski F-Score and
Coffee-Can quality tests, scores 19 markets with Trendlyne-style DVM (Durability / Valuation / Momentum),
and serves the results through a DuckDB warehouse, a FastAPI service and an SVG dashboard.
Outputs are styled Excel workbooks, CSV/parquet tables, markdown research notes and an HTML dashboard.

> Status: active research code. Educational use only; not investment advice. Backtests are pre-slippage
> unless a document says otherwise; fundamentals are point-in-time only for the US (SEC EDGAR) and
> current-snapshot elsewhere (yfinance). Retail-outlet / highway data lives in the separate
> [`retail-outlet-monitoring`](https://github.com/herrrickshaw/retail-outlet-monitoring) and
> [`fuel-retail-outlets`](https://github.com/herrrickshaw/fuel-retail-outlets) repos.

## What it does
- Full-universe scans per market (`full_{us,indian,japan,korea,european}_market_scan.py`): Darvas box +
  Piotroski + Coffee-Can, exported to styled Excel. **Triple Hit** = Darvas breakout and F-Score >= 7/9 and
  Coffee-Can pass (revenue CAGR > 10%, average ROCE > 15%, D/E < 1, positive earnings and FCF every year).
- Lookahead-free US backtest from SEC EDGAR filings (`pit_fundamentals.py`, `pit_backtest.py`) and a
  price-based point-in-time backtest for 18 other markets (`pit_global.py`).
- Factor research: QMJ quality, PEAD, Amihud liquidity, seasonality, crowding, option-implied, ESG
  (`quality_factor.py`, `pead_factor.py`, `liquidity_factor.py`, `factor_research.py`, ...).
- Trendlyne-style DVM scoring and GGG/GGB/BBG classification across 19 markets
  (`dvm_engine.py`, `dvm_global.py`, `fundamentals_global.py`, `dvm_composite.py`).
- Decision layer: constrained portfolios, VaR, meta-screen conviction, sector rotation, alerts
  (`portfolio.py`, `risk.py`, `meta_screen.py`, `sector_rotation.py`, `alerts.py`).
- Serving: a DuckDB warehouse (`warehouse.py`), FastAPI REST + versioned watchlist CRUD (`serve.py`,
  `watchlist_store.py`) and a dependency-free SVG dashboard (`dashboard.py`, `charts.py`).

## Quick start
```bash
git clone https://github.com/herrrickshaw/BazaarTalks.git
cd BazaarTalks
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python full_indian_market_scan.py --top 200      # scan one market -> styled Excel workbook
```
Other entry points (all local-data, no network unless noted):
```bash
python dvm_global.py --screen trendlyne_technical   # technical screen over 19 markets
python fundamentals_global.py --top 40              # source fundamentals once (network, cached)
python dvm_composite.py                             # GGG/GGB/BBG composite ranking
python pit_backtest.py --universe scan --years 5    # point-in-time US backtest
python pipeline.py --all US                         # warehouse -> validate -> analyze -> dashboard
python pipeline.py --validate                       # data_quality.py alone; non-zero exit on FAIL
```
Python 3.9+. The full [User Guide](USER_GUIDE.md) covers every module.

## How it works
Every scanner runs the same five stages:

| Stage | Module | Output |
|---|---|---|
| 1. Universe fetch | per-market `fetch_*` in `full_*_market_scan.py` | symbol list |
| 2. Bulk OHLC | `stock_utils.bulk_download`, `market_data_cache.py`, `apiclient.py` | price frames |
| 3. Darvas classification | `darvas_core.py` (shared kernel) via each scanner's `compute_darvas_box` | box top/bottom + signal |
| 4. Fundamentals on candidates | `fundamental_scan()` per scanner, helpers in `stock_utils.py` | F-Score, Coffee-Can |
| 5. Export | `save_excel` / `style_sheet` | styled `.xlsx` |

`pipeline.py` wraps these stages plus the warehouse build, data-quality gate, DVM + accumulation analysis and
dashboard; each stage skips gracefully when a view or cache is not built locally.

Key design decisions:
- **Current bar is excluded from Darvas box formation**, so a breakout is never part of its own box.
- **Each market keeps its own output shape.** The box-finding kernel is shared (`darvas_core.py`) but wrappers
  preserve per-market column lookup, rounding (whole units for JPY/KRW) and result keys.
- **Cache policy is separate from storage.** `ohlc_cache.py` holds the fresh / stale / miss rules behind an
  `OhlcStore` protocol; `market_store.py` supplies the Cassandra store and publishes a Kafka CDC event on every
  write. Cassandra and Kafka are mandatory there: failures raise `MarketStoreUnavailable` / `CDCPublishError`
  instead of silently reverting to uncached yfinance calls.
- **Charts are plain SVG** (`charts.py`), with no matplotlib/plotly dependency.
- **Live India fundamentals** come from your own Trendlyne / Screener.in accounts via cookie sessions
  (`trendlyne_session.py`, `screener_session.py`); no password is ever submitted by this code. See
  [TRENDLYNE_SCREENER_ACCESS.md](TRENDLYNE_SCREENER_ACCESS.md); `python pipeline.py --live IN --limit 50`
  backfills `fundamentals_cache.db`.

## Repository layout
| Path | Purpose |
|---|---|
| `full_*_market_scan.py` | Five market scanners (US, India, Japan, Korea, Europe) |
| `darvas_core.py`, `ohlc_cache.py`, `stock_utils.py`, `marketdata.py` | Shared pure building blocks |
| `market_store.py`, `stream_pipeline.py`, `flink_screens.py`, `kafka_*_consumer.py`, `cdc/` | Cassandra / Kafka / Flink data backbone |
| `dvm_*.py`, `fundamentals_global.py`, `pit_*.py`, `*_factor.py`, `factor_research.py` | Scoring, point-in-time and factor research |
| `portfolio.py`, `risk.py`, `meta_screen.py`, `alerts.py`, `sector_rotation.py` | Decision layer |
| `warehouse.py`, `serve.py`, `dashboard.py`, `charts.py`, `data_quality.py` | Serving and observability |
| `tests/` | pytest suite (`test_core.py`, `test_darvas_core.py`, `test_ohlc_cache.py`, golden data) |
| `architecture/`, `safe/` | TOGAF governance check and SAFe backlog (run in CI) |
| `*.md` (root) | Per-module write-ups; index below |
| `*.parquet`, `*.db`, `*.csv`, `cache_seed_local/` | Tracked datasets and small result stores |

## Data sources
| Source | What is taken | Access route | Licence / caveat |
|---|---|---|---|
| yfinance | OHLC and fundamentals, all markets | Python library | Unofficial Yahoo API; personal/research use |
| SEC EDGAR | US point-in-time fundamentals (filed-date filtered) | public API | Public domain |
| NSE / BSE | India universe and quotes | `nsepython`, `bseindia` | Exchange terms apply |
| JPX / KRX | Japan and Korea universes | `kabupy`, `pykrx` | Third-party libraries, unpinned |
| Trendlyne, Screener.in | India live fundamentals | your own paid account, cookie session | Account terms apply |
| OpenAlex, Crossref, arXiv | Literature scout | public APIs | Open metadata |
| Ken French / AQR | Factor benchmarks | public downloads | See [DATA_SOURCES.md](DATA_SOURCES.md) |

## Configuration
No secrets are committed. Credentials for Trendlyne/Screener.in are browser cookies you supply locally;
Cassandra is expected at `127.0.0.1:9042` and Kafka at `$KAFKA_BOOTSTRAP` (default `localhost:9092`).
`marketdata.SEED` points at a local parquet cache under `~/Downloads/...`; India's seed is in `cache_seed_local/`.

## Testing
```bash
pip install pytest pandas numpy scipy scikit-learn requests   # plus yfinance/openpyxl for scanner tests
pytest -q            # 139 tests; no network, Cassandra or Kafka required
```
`tests/golden/darvas_golden.json` pins each scanner's `compute_darvas_box` output (captured from the
pre-refactor code) on deterministic synthetic panels. Scanner tests stub `bseindia` / `pykrx` when absent.

## Limitations and known issues
- Outside the US fundamentals are a snapshot of today, so non-US factor sizes are directional only.
- All backtest returns are before trading costs unless a document says otherwise ([NET_OF_COST.md](NET_OF_COST.md)).
- Universes are current constituents (survivorship bias); see [REMEDIATION.md](REMEDIATION.md), `survivorship.py`.
- `CHECKSUMS.sha256` covers only part of the tracked tree, so the CI integrity step is not a reliable gate.
- Repo-wide `ruff check` reports about 100 pre-existing findings in legacy modules; the new modules are clean.
- Some modules read absolute `~/Downloads/...` paths on the author's machine and will need a local data tree.
- Stray QA render images (`cv-4.jpg`, `rm-15.jpg`) are tracked in the root; they are manuscript page renders.

## Documentation index
- Guides: [User Guide](USER_GUIDE.md), [Glossary](GLOSSARY.md), [Architecture](ARCHITECTURE.md),
  [Architecture map](ARCHITECTURE_MAP.md), [Schema](SCHEMA.md), [Warehouse](WAREHOUSE.md), [Infra](INFRA.md),
  [SDLC](SDLC.md), [Security](SECURITY.md), [Performance](PERFORMANCE.md), [TOGAF](architecture/TOGAF.md),
  [SAFe](safe/SAFE.md), [Watchlist store](WATCHLIST_STORE.md), [CDC deployment](cdc/CDC_DEPLOYMENT.md)
- Screening: [Screens](SCREENS.md), [Global DVM](GLOBAL_DVM.md), [Fundamental screen](FUNDAMENTAL_SCREEN.md),
  [DVM composite](DVM_COMPOSITE.md), [ML screen discovery](ML_SCREEN_DISCOVERY.md)
- Research: [Paper](RESEARCH_PAPER.md), [detailed](RESEARCH_PAPER_DETAILED.md), [simple](RESEARCH_PAPER_SIMPLE.md),
  [Screen viability](SCREEN_VIABILITY_RESULTS.md), [PIT scope](SCOPE_PIT_FUNDAMENTALS.md),
  [PIT results](PIT_BACKTEST_RESULTS.md), [ML viability](ML_VIABILITY.md), [Factor research](FACTOR_RESEARCH.md),
  [Quality (QMJ)](QUALITY_FACTOR.md), [Scout](SCOUT.md), [PEAD](PEAD.md), [Liquidity](LIQUIDITY.md),
  [Benchmark](BENCHMARK.md), [HFT selection](HFT_SELECTION.md), [Darvas x volume](DARVAS_VOLUME.md),
  [Watchlists](WATCHLISTS.md), [Accumulation](ACCUMULATION_SCREENER.md), [Corporate actions](CORPORATE_ACTIONS.md),
  [Frontier gaps](FRONTIER_GAPS.md), [Earnings x liquidity](EARNINGS_LIQUIDITY.md),
  [Decision layer](DECISION_LAYER.md), [Net of cost](NET_OF_COST.md)
- Using this repo with an AI assistant: [HOW_TO_USE_WITH_AI.md](HOW_TO_USE_WITH_AI.md)
- Zero-install tour: [Colab notebook](https://colab.research.google.com/github/herrrickshaw/global-market-scanners/blob/main/notebooks/DVM_Global_Colab.ipynb)

## Related repositories
[`retail-outlet-monitoring`](https://github.com/herrrickshaw/retail-outlet-monitoring) and
[`fuel-retail-outlets`](https://github.com/herrrickshaw/fuel-retail-outlets) hold the non-equity data that
was split out of this project.

## Licence
No licence file yet — all rights reserved.
