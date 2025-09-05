import os, io, math, asyncio, datetime as dt, zoneinfo, json, textwrap, random
import numpy as np
import pandas as pd
import yfinance as yf
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, ContextTypes
import plotly.graph_objects as go
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volume import OnBalanceVolumeIndicator

# =========================
# Config from environment
# =========================
TZ = zoneinfo.ZoneInfo(os.getenv("TIMEZONE", "Africa/Johannesburg"))
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MAIN_CHAT = os.getenv("MAIN_CHAT_ID")
VIP_CHAT  = os.getenv("VIP_CHAT_ID")
MAIN_INVITE = os.getenv("MAIN_INVITE_LINK", "")
VIP_INVITE  = os.getenv("VIP_INVITE_LINK", "")
ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "10000"))
RISK_PCT = float(os.getenv("RISK_PCT", "5"))
STOP_PCT = float(os.getenv("STOP_PCT", "0.0005"))  # 0.05% of price by default
ENABLE_DAILY_BIAS = os.getenv("ENABLE_DAILY_BIAS", "1") == "1"
ENABLE_HOURLY_SCAN = os.getenv("ENABLE_HOURLY_SCAN", "1") == "1"
SESSION_ROTATION = os.getenv("SESSION_ROTATION", "1") == "1"

SYMBOLS_MAIN = [s.strip() for s in os.getenv("SYMBOLS_MAIN", "EURUSD=X,XAUUSD=X,BTC-USD,USO,^NDX").split(",") if s.strip()]
SYMBOLS_VIP  = [s.strip() for s in os.getenv("SYMBOLS_VIP",  "ETH-USD,GBPUSD=X,USDJPY=X,CL=F,GC=F").split(",") if s.strip()]

RISK_PCT = max(0.05, min(15.0, RISK_PCT))  # clamp 0.05..15

# =========================
# Helpers
# =========================
def now_sast():
    return dt.datetime.now(TZ)

def session_name(t: dt.datetime) -> str:
    # Rough session windows in SAST
    h = t.hour
    if 1 <= h < 9: return "Asian"
    if 9 <= h < 17: return "European"
    return "American"

def fetch(symbol: str, days=30, interval="1h") -> pd.DataFrame:
    df = yf.download(symbol, period=f"{days}d", interval=interval, auto_adjust=True, progress=False)
    if df.empty: return df
    df = df.dropna().copy()
    df.index = df.index.tz_convert(TZ) if df.index.tzinfo else df.index.tz_localize(TZ)
    return df

def ema(series, length):
    return EMAIndicator(series, window=length).ema_indicator()

def rsi(series, length=14):
    return RSIIndicator(series, window=length).rsi()

def obv(close, volume):
    return OnBalanceVolumeIndicator(close, volume).on_balance_volume()

def slope(series, lookback=20):
    if len(series) < lookback: return 0.0
    y = series[-lookback:]
    x = np.arange(len(y))
    m, b = np.polyfit(x, y, 1)
    return float(m)

def recent_swing(df: pd.DataFrame, lookback=100):
    win = df.tail(lookback)
    hi = win['High'].max()
    lo = win['Low'].min()
    hi_t = win['High'].idxmax()
    lo_t = win['Low'].idxmin()
    # define swing direction by which came last
    direction = "up" if lo_t < hi_t else "down"
    return dict(hi=hi, lo=lo, hi_t=hi_t, lo_t=lo_t, direction=direction)

def fib_levels(hi, lo):
    diff = hi - lo
    levels = {
        "0.236": hi - 0.236*diff,
        "0.382": hi - 0.382*diff,
        "0.500": hi - 0.500*diff,
        "0.618": hi - 0.618*diff,
        "0.786": hi - 0.786*diff,  # TP3 target
    }
    return levels

def volume_profile(df: pd.DataFrame, bins=24):
    # Simple price-volume histogram as order-flow proxy
    prices = (df['High']+df['Low']+df['Close'])/3
    vols = df['Volume'].fillna(0)
    hist, edges = np.histogram(prices, bins=bins, weights=vols)
    centers = (edges[:-1] + edges[1:]) / 2
    # Return main node for confluence checks
    if hist.sum() == 0:
        hvn = float(prices.iloc[-1])
    else:
        hvn = float(centers[np.argmax(hist)])
    return hvn, centers, hist

def momentum_probability(df: pd.DataFrame) -> float:
    # Heuristic ">= 0.80" momentum within next 4h:
    # combine RSI(14), 200-EMA slope flatness, OBV trend
    r = rsi(df['Close'])
    rs = float(r.iloc[-1]) if not r.empty else 50.0

    ema200 = ema(df['Close'], 200)
    s200 = abs(slope(ema200, 24))
    flat_bonus = 1.0 if s200 < (df['Close'].iloc[-1]*1e-5) else 0.8  # flatter is better

    obv_series = obv(df['Close'], df['Volume'].fillna(0))
    obv_s = slope(obv_series, 24)
    obv_score = 1.0 if obv_s > 0 else 0.7

    rsi_score = (rs-30)/40  # 30..70 -> 0..1
    rsi_score = max(0, min(1, rsi_score))

    p = 0.6*rsi_score + 0.2*flat_bonus + 0.2*obv_score
    return float(max(0.0, min(1.0, p)))

def position_sizing_text(price: float, stop_pct: float, risk_pct: float, balance: float):
    # loss per unit = price * stop_pct
    loss_per_unit = price * stop_pct
    risk_amount = balance * (risk_pct/100.0)
    size = risk_amount / max(1e-12, loss_per_unit)
    # how many losing trades until balance 0 (approx)
    trades_left = balance / max(1e-12, risk_amount)
    return size, risk_amount, trades_left

def confluence_check(df: pd.DataFrame):
    # EMA stack + 20 near 200 cross, 200 relatively flat + HVN near 20/200
    c = df.copy()
    c['ema2'] = ema(c['Close'], 2)
    c['ema5'] = ema(c['Close'], 5)
    c['ema20'] = ema(c['Close'], 20)
    c['ema75'] = ema(c['Close'], 75)
    c['ema200'] = ema(c['Close'], 200)

    last = c.iloc[-1]
    s200 = slope(c['ema200'], 48)
    flat = abs(s200) < (last.Close*1e-5)

    hvn, _, _ = volume_profile(c.tail(240), bins=24)
    near_cross = abs(last.ema20 - last.ema200) / last.Close < 0.0025  # ~0.25%
    hvn_near = abs(hvn - last.ema20)/last.Close < 0.005 or abs(hvn - last.ema200)/last.Close < 0.005

    buy_stack = last.ema2 > last.ema5 > last.ema20 > last.ema200
    sell_stack = last.ema2 < last.ema5 < last.ema20 < last.ema200

    if flat and near_cross and hvn_near and buy_stack:
        return "BUY", c
    if flat and near_cross and hvn_near and sell_stack:
        return "SELL", c
    return None, c

def build_targets(df: pd.DataFrame, side: str):
    sw = recent_swing(df, lookback=200)
    fibs = fib_levels(sw["hi"], sw["lo"])
    close = float(df['Close'].iloc[-1])

    # TP1 = EMA75 pullback; TP2 = nearest key fib (0.382/0.5/0.618 by swing dir); TP3 = 0.786
    e75 = float(ema(df['Close'], 75).iloc[-1])

    if side == "BUY":
        # choose next fib above price as TP2 (if swing up), else nearest
        candidates = [("TP1", e75)]
        # sorted ascending for buys
        for k in ["0.382","0.500","0.618","0.786"]:
            candidates.append((k, fibs[k]))
        # ensure > price
        tps = [(n,v) for (n,v) in candidates if v > close]
        if not tps:  # fallback
            tps = [("TP1", e75), ("0.786", fibs["0.786"])]
    else:
        candidates = [("TP1", e75)]
        for k in ["0.618","0.500","0.382","0.786"]:
            candidates.append((k, fibs[k]))
        # ensure < price for sells
        tps = [(n,v) for (n,v) in candidates if v < close]
        if not tps:
            tps = [("TP1", e75), ("0.786", fibs["0.786"])]

    # Guarantee TP3 = 0.786 in the ladder if feasible
    has_786 = any(n=="0.786" for n,_ in tps)
    if not has_786:
        tps.append(("0.786", fibs["0.786"]))
    # Keep unique and ordered by distance from price in correct direction
    if side == "BUY":
        tps = sorted(list({n:v for n,v in tps}.items()), key=lambda x: x[1])
    else:
        tps = sorted(list({n:v for n,v in tps}.items()), key=lambda x: x[1], reverse=True)
    return tps[:5]  # cap TP list

def momentum_extension(df: pd.DataFrame, side: str):
    # If probability >=0.80, extend with TP4 / TP5 in the 1:4.5..1:7 zone as extra
    p = momentum_probability(df)
    return p >= 0.80, p

def build_chart(df_cooked: pd.DataFrame, side: str, entry: float, sl: float, tps: list, title: str):
    # Plotly candle + EMAs + shaded zones
    df = df_cooked.copy().tail(200)
    fig = go.Figure()
    fig.add_candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price")

    for length, color, name in [(2,"#AAAAAA","EMA2"),(5,"#888888","EMA5"),(20,"#0066FF","EMA20"),
                                (50,"#9999FF","EMA50"),(75,"#00BBBB","EMA75"),(200,"#000000","EMA200")]:
        if f"ema{length}" not in df.columns:
            df[f"ema{length}"] = ema(df['Close'], length)
        fig.add_trace(go.Scatter(x=df.index, y=df[f"ema{length}"], mode="lines", name=name, line=dict(width=1.4, color=color)))

    # SL zone (purple)
    y0, y1 = (sl, entry) if side=="BUY" else (entry, sl)
    fig.add_shape(type="rect", x0=df.index[-180], x1=df.index[-1], y0=min(y0,y1), y1=max(y0,y1),
                  line=dict(width=0), fillcolor="rgba(128,0,128,0.18)", layer="below")

    # TP layered zones (black, red, gold, green)
    colors = ["#00000055","#FF000055","#C0A00055","#00AA0055"]
    for i,(name,price) in enumerate(tps[:4]):
        low, high = (entry, price) if (side=="BUY") else (price, entry)
        fig.add_shape(type="rect", x0=df.index[-180], x1=df.index[-1], y0=min(low,high), y1=max(low,high),
                      line=dict(width=0), fillcolor=colors[i], layer="below")

    # Entry line + labels
    fig.add_hline(y=entry, line=dict(color="#555", width=1, dash="dot"), annotation_text=f"Entry {entry:.4f}")

    # TP markers
    for idx,(name,price) in enumerate(tps, start=1):
        fig.add_hline(y=price, line=dict(color="#222", width=1),
                      annotation_text=f"TP{idx} {name}: {price:.4f}")

    # SL
    fig.add_hline(y=sl, line=dict(color="#800080", width=2),
                  annotation_text=f"SL {sl:.4f}")

    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=720, width=1100,
                      template="plotly_white")
    # Export to PNG via kaleido
    buf = io.BytesIO()
    fig.write_image(buf, format="png", scale=2)
    buf.seek(0)
    return buf

def analyze_symbol(symbol: str, balance: float, risk_pct: float, stop_pct: float):
    df = fetch(symbol, days=60, interval="1h")
    if df.empty: return None

    side, cooked = confluence_check(df)
    if not side: return None

    last = cooked.iloc[-1]
    entry = float(last['Close'])
    # SL from STOP_PCT of price
    sl = entry * (1 - stop_pct) if side=="BUY" else entry * (1 + stop_pct)

    # Targets
    tps = build_targets(cooked, side)
    extend, p = momentum_extension(cooked, side)
    # If extension allowed, add TP4/TP5 in 1:4.5..1:7 R:R band relative to SL distance
    rr_min, rr_max = 4.5, 7.0
    risk_distance = abs(entry - sl)
    if extend and len(tps) < 5:
        # compute target by R:R
        for rr in [rr_min, (rr_min+rr_max)/2.0, rr_max]:
            if side=="BUY":
                t = entry + rr * risk_distance
                tps.append((f"RR{rr:.1f}", t))
            else:
                t = entry - rr * risk_distance
                tps.append((f"RR{rr:.1f}", t))
            if len(tps) >= 5:
                break

    # Size + risk text
    size, risk_amt, trades_left = position_sizing_text(entry, stop_pct, risk_pct, balance)

    # Chart
    title = f"{symbol} · {side} · {session_name(now_sast())}"
    png = build_chart(cooked, side, entry, sl, tps, title)

    # Round display
    def r4(x): return float(f"{x:.4f}")

    return {
        "symbol": symbol,
        "side": side,
        "entry": r4(entry),
        "sl": r4(sl),
        "tps": [(n, r4(v)) for (n,v) in tps],
        "risk_pct": risk_pct,
        "stop_pct": stop_pct,
        "size_hint": round(size, 4),
        "risk_amt": round(risk_amt, 2),
        "trades_left": round(trades_left, 1),
        "momentum_p": round(p, 2),
        "png": png
    }

async def send_signal(context: ContextTypes.DEFAULT_TYPE, chat_id: str, sig: dict, vip=False):
    # Compose message with emoji policy
    sess = session_name(now_sast())
    mood_line = "React: 👍💡 | 😐🍋 | 👎💧"
    style = "Scalp/Intraday"  # placeholder; could infer by ATR later

    tp_lines = []
    for i,(name,price) in enumerate(sig["tps"], start=1):
        tag = " (TP3 core)" if name in ("0.786","TP3") or i==3 else ""
        tp_lines.append(f"TP{i} {name}: {price:.4f}{tag}")

    text = textwrap.dedent(f"""
    {'💎 VIP' if vip else '📣 MAIN'} · {sess}
    {sig['symbol']} · {sig['side']} · {style}

    Entry: {sig['entry']:.4f}
    SL: {sig['sl']:.4f}  (0.05% price)  | Risk: {sig['risk_pct']}% of equity = ${sig['risk_amt']:.2f}
    Size hint (not advice): ~{sig['size_hint']} units  | Trades till $0 at this risk: ~{sig['trades_left']}

    Momentum follow-through 4h: {int(sig['momentum_p']*100)}%
    """ + "\n".join(tp_lines) + f"""

    Notes:
    • Confluence = 20EMA~200EMA near-cross + flat 200 + HVN proximity + EMA stack (2>5>20>200 for BUY, inverse for SELL).
    • TP3 fixed at 0.786 Fib. TP4/5 appear only if momentum ≥ 80%.
    • Chart zones: Purple=SL, Black/Red/Gold/Green=TP areas.

    {mood_line}
    """).strip()

    await context.bot.send_photo(chat_id=chat_id, photo=InputFile(sig["png"], filename=f"{sig['symbol']}.png"), caption=text)

# =========================
# Telegram handlers
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 Prolucian 3.0 bot online.\nUse /status, /signal SYMBOL, /subscribe, /risk N, /setbalance N, /dailybias")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = textwrap.dedent(f"""
    ⏱ {now_sast():%Y-%m-%d %H:%M} {session_name(now_sast())}
    Balance: ${ACCOUNT_STATE['balance']:.2f}
    Risk: {ACCOUNT_STATE['risk_pct']}% (min 0.05, max 15)
    StopPct: {ACCOUNT_STATE['stop_pct']*100:.3f}%
    Main: {MAIN_CHAT or '—'} | VIP: {VIP_CHAT or '—'}
    Symbols MAIN: {', '.join(SYMBOLS_MAIN)}
    Symbols VIP: {', '.join(SYMBOLS_VIP)}
    """).strip()
    await update.message.reply_text(msg)

async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = []
    if MAIN_INVITE:
        lines.append(f"Main invite: {MAIN_INVITE}")
    if VIP_INVITE:
        lines.append(f"VIP invite: {VIP_INVITE}")
    if not lines:
        lines.append("No invite links configured yet.")
    await update.message.reply_text("\n".join(lines))

async def setbalance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ACCOUNT_STATE
    try:
        v = float(context.args[0])
        ACCOUNT_STATE['balance'] = max(10.0, v)
        await update.message.reply_text(f"✅ Balance set to ${ACCOUNT_STATE['balance']:.2f}")
    except Exception:
        await update.message.reply_text("Usage: /setbalance 15000")

async def setrisk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ACCOUNT_STATE
    try:
        v = float(context.args[0])
        v = max(0.05, min(15.0, v))
        ACCOUNT_STATE['risk_pct'] = v
        await update.message.reply_text(f"✅ Risk set to {v}%")
    except Exception:
        await update.message.reply_text("Usage: /risk 5  (0.05..15)")

async def signal_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sym = context.args[0] if context.args else None
    if not sym:
        await update.message.reply_text("Usage: /signal EURUSD=X")
        return
    await update.message.reply_text(f"Analyzing {sym}…")
    sig = analyze_symbol(sym, ACCOUNT_STATE['balance'], ACCOUNT_STATE['risk_pct'], ACCOUNT_STATE['stop_pct'])
    if not sig:
        await update.message.reply_text("No valid setup right now.")
        return
    await send_signal(context, update.effective_chat.id, sig, vip=False)

async def dailybias(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await post_daily_bias(context, MAIN_CHAT or update.effective_chat.id, title="Daily Bias (Main)")
    if VIP_CHAT:
        await post_daily_bias(context, VIP_CHAT, title="Daily Bias (VIP)")

# =========================
# Schedulers / Jobs
# =========================
ACCOUNT_STATE = {
    "balance": ACCOUNT_BALANCE,
    "risk_pct": RISK_PCT,
    "stop_pct": STOP_PCT,
}

async def scan_and_post(context: ContextTypes.DEFAULT_TYPE):
    # Main
    for sym in SYMBOLS_MAIN:
        sig = analyze_symbol(sym, ACCOUNT_STATE['balance'], ACCOUNT_STATE['risk_pct'], ACCOUNT_STATE['stop_pct'])
        if sig and MAIN_CHAT:
            await send_signal(context, MAIN_CHAT, sig, vip=False)
        await asyncio.sleep(1.5)
    # VIP
    for sym in SYMBOLS_VIP:
        sig = analyze_symbol(sym, ACCOUNT_STATE['balance'], ACCOUNT_STATE['risk_pct'], ACCOUNT_STATE['stop_pct'])
        if sig and VIP_CHAT:
            await send_signal(context, VIP_CHAT, sig, vip=True)
        await asyncio.sleep(1.5)

async def post_daily_bias(context: ContextTypes.DEFAULT_TYPE, chat_id: str, title="Daily Bias"):
    # Lightweight bias summary per symbol (trend vs EMA200, RSI, momentum p)
    lines = [f"🗞 {title} · {now_sast():%Y-%m-%d}"]
    for sym in SYMBOLS_MAIN[:3] + SYMBOLS_VIP[:3]:
        df = fetch(sym, days=60, interval="1h")
        if df.empty: 
            lines.append(f"{sym}: no data")
            continue
        e200 = ema(df['Close'], 200).iloc[-1]
        close = df['Close'].iloc[-1]
        rs = rsi(df['Close']).iloc[-1]
        p = int(momentum_probability(df)*100)
        bias = "Bull" if close > e200 else "Bear"
        lines.append(f"{sym}: {bias} | RSI {rs:.0f} | 4h follow-through {p}%")
    msg = "\n".join(lines)
    await context.bot.send_message(chat_id=chat_id, text=msg)

def schedule_jobs(app: Application):
    sched = AsyncIOScheduler(timezone=str(TZ))
    if ENABLE_HOURLY_SCAN:
        sched.add_job(lambda: app.create_task(scan_and_post(app.bot)), "cron", minute=0)
    if ENABLE_DAILY_BIAS:
        # Post around 07:30 SAST pre-Europe
        sched.add_job(lambda: app.create_task(post_daily_bias(app, MAIN_CHAT, "Daily Bias (Main)")), "cron", hour=7, minute=30)
        if VIP_CHAT:
            sched.add_job(lambda: app.create_task(post_daily_bias(app, VIP_CHAT, "Daily Bias (VIP)")), "cron", hour=7, minute=32)
    sched.start()

# =========================
# App bootstrap
# =========================
def main():
    if not BOT_TOKEN:
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN")

    app = Application.builder().token(BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(CommandHandler("setbalance", setbalance))
    app.add_handler(CommandHandler("risk", setrisk))
    app.add_handler(CommandHandler("signal", signal_cmd))
    app.add_handler(CommandHandler("dailybias", dailybias))

    # Startup notifier
    async def _on_start(_):
        try:
            if MAIN_CHAT:
                await app.bot.send_message(chat_id=MAIN_CHAT, text="✅ Prolucian 3.0 online (MAIN).")
            if VIP_CHAT:
                await app.bot.send_message(chat_id=VIP_CHAT, text="💎 Prolucian 3.0 online (VIP).")
        except Exception as e:
            print("Startup notify error:", e)

    app.post_init = _on_start

    # Schedule
    schedule_jobs(app)

    # Run
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
