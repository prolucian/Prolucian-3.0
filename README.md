# Prolucian 3.0 – Signals + Bias + Telegram

This bot posts **hourly signals**, **TP ladders (TP1..TP3(+TP4/5 optional))**, **daily bias**, and **session-rotated charts** to your Telegram **Main** and **VIP** groups.

## 1) Required Config Vars (Heroku → Settings → Reveal Config Vars)
- `TELEGRAM_BOT_TOKEN` – your bot token
- `MAIN_CHAT_ID` – e.g. `-1001234567890`
- `VIP_CHAT_ID` – e.g. `-1009876543210` (optional)
- `ACCOUNT_BALANCE` – starting equity (used for sizing text)
- `RISK_PCT` – % risk per trade (default 5, clamps to 0.05..15)
- `STOP_PCT` – 0.0005 = 0.05% of price
- `TIMEZONE` – `Africa/Johannesburg`
- `SYMBOLS_MAIN` – comma-separated tickers
- `SYMBOLS_VIP` – comma-separated tickers
- `MAIN_INVITE_LINK` – optional, shared on `/subscribe`
- `VIP_INVITE_LINK` – optional, shared on `/subscribe`

(Optional)
- `TE_API_KEY` – TradingEconomics API (news/bias add-on)
- `STRIPE_SECRET` – billing stub
- `COPY_TRADE_WEBHOOK` – copy-trade destination

## 2) Deploy (Heroku)
- Create app → Deploy tab → **GitHub** → connect this repo.
- **Settings → Reveal Config Vars** → add the vars above.
- **Manual Deploy** → Deploy branch.
- **Resources** → turn ON `worker: python bot.py`.

## 3) Commands
- `/start` – hello + quick checks
- `/status` – shows balances, risk, sessions, symbols
- `/subscribe` – replies with invite links (if set)
- `/signal EURUSD=X` – on-demand analysis for a symbol
- `/setbalance 15000` – update in-memory balance
- `/risk 7` – set risk % (clamped 0.05..15)
- `/dailybias` – post bias now

## Notes
- TP3 is **always** the 0.786 Fibonacci level (from most recent swing).
- Optional TP4/TP5 appear only if **momentum ≥ 0.80** heuristic and 4h window probability check passes.
- SL distance uses `STOP_PCT` (default 0.05% of price). Position sizing text shows how many trades remain before balance exhaustion at the chosen risk.
