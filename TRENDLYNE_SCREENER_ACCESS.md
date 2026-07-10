# Providing Trendlyne / Screener.in access (for live India fundamentals)

`trendlyne_session.py` and `screener_session.py` pull real fundamentals
(PE, PB, market cap, ROE, ROCE, D/E, growth, dividend yield) from your own
paid Trendlyne and Screener.in accounts. **Neither module ever submits a
password anywhere** — both work off a session cookie you obtain by logging in
yourself in a normal browser, the same way you'd use any API key.

Two reasons for this design, not just one:
1. Screener.in's login is a plain server-rendered form — automating it would
   be easy, but entering a real password into a website isn't something this
   tooling does, on principle, regardless of whose account it is.
2. Trendlyne's login is additionally behind Google reCAPTCHA v3, which is
   specifically built to detect and block headless browser automation. Trying
   to defeat that would mean actively circumventing the site's own anti-bot
   protection — a different thing entirely from politely using your own paid
   account for read-only lookups, and not something to attempt.

## 1. Get the cookies

Log into **screener.in** and **trendlyne.com** in your normal browser (already
logged in works too). Then, for each site:

1. Open DevTools → **Application** tab (Chrome) / **Storage** (Firefox)
2. Left sidebar → **Cookies** → select the site's origin
3. Copy the values of:
   - **screener.in**: `sessionid`, `csrftoken`
   - **trendlyne.com**: the cookie named `.trendlyne` (HttpOnly, Secure — this
     is the actual session token; ignore `_ga*`, `g_state`, `TLCDayNudgeLogin`,
     which are just analytics/UI state), and `csrftoken`

## 2. Set the credentials

```bash
cp .env.example .env      # then paste your real cookie values into .env
python screener_session.py RELIANCE    # expect: logged in ✓
python trendlyne_session.py RELIANCE   # expect: logged in ✓
```

`_load_dotenv()` in both modules auto-loads `.env` (checking both this repo
and the sibling `working-files-repo/python_files/.env`, so one file can serve
both projects).

**Google Colab** — use the 🔑 Secrets panel, then:
```python
import os
from google.colab import userdata
for k in ("SCREENER_SESSIONID", "SCREENER_CSRFTOKEN",
          "TRENDLYNE_SESSIONID", "TRENDLYNE_CSRFTOKEN"):
    os.environ[k] = userdata.get(k)
```

## 3. Sessions expire

When a fetch starts failing with `AuthError`, the cookie has expired — just
log in again in your browser and refresh the value in `.env`. There's no
"refresh token" flow here since nothing automates the login itself.

## 4. Run a batch backfill

```bash
python live_fundamentals.py --batch RELIANCE,TCS,INFY
python pipeline.py --live IN --limit 50 --pause 1.5   # first N India tickers from the companies view
python pipeline.py --live IN --pause 1.5               # full backfill (~5,300 tickers — slow, be deliberate)
```

Results land in `fundamentals_cache.db` (gitignored, rebuilt on demand) with
a `source` column (`trendlyne`, `screener.in`, or `trendlyne+screener.in`),
immediately visible through `warehouse.py`'s `fundamentals` view and
`ticker_view.py`'s scorecard — no other wiring needed.

## Security & etiquette

- Cookie values only — a password never touches this code, ever.
- Use a `--pause` between requests (default 1.5s in `live_fundamentals.py`,
  matching the platform's existing rate-limit convention) — this is your own
  account's data, fetched politely, not a scrape of someone else's.
- Respect both sites' Terms of Service. A full India-universe backfill
  (~5,300 tickers) is a real, deliberate action — start with `--limit` for a
  test run before committing to the full set.
- Session cookies are credentials — treat them exactly like a password:
  `.env` only, gitignored, never pasted into chat/logs/commits.
