# =========================  BULLSEYE (All-in-one)  =========================
# Telegram signal engine + web (3D charts) + scheduler in a single file.
# Paste into bot.py. Run on Heroku with a "web" dyno (Procfile: web: python bot.py)
#
# REQUIRED ENV (Config Vars):
#   TELEGRAM_BOT_TOKEN : str  | BotFather token
#   MAIN_CHAT_ID       : str  | e.g. -1001234567890
#   VIP_CHAT_ID        : str  | e.g. -1009876543210 (optional but recommended)
#   ACCOUNT_BALANCE    : float| starting balance (e.g. 10000)
#   RISK_PCT           : float| 0.05 .. 15 (default 5)
#   STOP_PCT           : float| default 0.0005 (0.05% of price)
#   TIMEZONE           : str  | Africa/Johannesburg
# OPTIONAL:
#   SYMBOLS_MAIN, SYMBOLS_VIP (CSV lists). TE_API_KEY, STRIPE_SECRET, STRIPE_WEBHOOK_SECRET.
# ==========================================================================

import os, io, math, json, threading, asyncio, textwrap, random
import datetime as dt, zoneinfo
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

import yfinance as yf
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volume import OnBalanceVolumeIndicator

import plotly.graph_objects as go
from flask import Flask, request, render_template_string
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, ContextTypes

# ------------------------- CONFIG -------------------------
TZ = zoneinfo.ZoneInfo(os.getenv("TIMEZONE", "Africa/Johannesburg"))
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
MAIN_CHAT = os.getenv("MAIN_CHAT_ID", "")
VIP_CHAT  = os.getenv("VIP_CHAT_ID", "")
MAIN_INVITE = os.getenv("MAIN_INVITE_LINK", "")
VIP_INVITE  = os.getenv("VIP_INVITE_LINK", "")

ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "10000"))
RISK_PCT = float(os.getenv("RISK_PCT", "5"))
STOP_PCT = float(os.getenv("STOP_PCT", "0.0005"))  # 0.05% of price (not balance)
RISK_PCT = max(0.05, min(15.0, RISK_PCT))

TE_API_KEY = os.getenv("TE_API_KEY", "")  # optional
STRIPE_SECRET = os.getenv("STRIPE_SECRET", "")  # optional
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")  # optional

# Default 30 "volatile" symbols (majors/minors/metals/oil/indices/crypto)
DEFAULT_SYMBOLS = [
    "EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","NZDUSD=X","USDCHF=X","USDCAD=X",
    "XAUUSD=X","XAGUSD=X","XPTUSD=X","XPDUSD=X","CL=F","NG=F","GC=F","SI=F",
    "^NDX","^SPX","^DJI","^VIX","^RUT","^FTSE","^N225","^HSI",
    "BTC-USD","ETH-USD","SOL-USD","BNB-USD","DOGE-USD",
    "USO","GLD","HYG"
][:30]

# MAIN cadence: interval -> max hold (minutes)
MAIN_TIERS = {
    "2m": 15,
    "5m": 30,
    "15m": 45,
    "30m": 120,
}
# VIP cadence
VIP_TIERS = {
    "45m": 360,     # 6h
    "60m": 10080,   # 1 week (7*24*60)
    "240m": 43200,  # 1 month (~30*24*60)
}

# ------------------------- STATE --------------------------
STATE = {
    "balance": ACCOUNT_BALANCE,
    "risk_pct": RISK_PCT,
    "stop_pct": STOP_PCT,
    "main_symbols": [],
    "vip_symbols": [],
    "active_signals": {},  # key: signal_id -> dict(...)
    "analytics": {
        "Asian":   {"signals":0, "wins":0, "points":0.0},
        "European":{"signals":0, "wins":0, "points":0.0},
        "American":{"signals":0, "wins":0, "points":0.0},
    }
}

def now_sast() -> dt.datetime: return dt.datetime.now(TZ)

def session_name(t: Optional[dt.datetime]=None) -> str:
    t = t or now_sast()
    h = t.hour
    if 1 <= h < 9: return "Asian"
    if 9 <= h < 17: return "European"
    return "American"

# ---------------------- DATA HELPERS ----------------------
def yf_interval(interval: str) -> Tuple[str, int]:
    """Map our intervals to yfinance-friendly and lookback days."""
    if interval in ("1m","2m","5m","15m","30m","60m","90m","1h","1d","5d"): return interval, 7
    # synthesize:
    if interval == "45m": return "15m", 20
    if interval == "240m": return "60m", 180  # 4h synthesized from 1h
    return "60m", 60

def fetch(symbol: str, interval: str, bars: int=400) -> pd.DataFrame:
    yf_int, days = yf_interval(interval)
    df = yf.download(symbol, period=f"{days}d", interval=yf_int, auto_adjust=True, progress=False)
    if df.empty: return df
    df = df.dropna().copy()
    # synthesize 45m from 15m, 240m from 60m
    if interval == "45m":
        df = df.resample("45T").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    elif interval == "240m":
        df = df.resample("240T").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    df.index = df.index.tz_convert(TZ) if df.index.tzinfo else df.index.tz_localize(TZ)
    if len(df) > bars: df = df.tail(bars)
    return df

def ema(s: pd.Series, n: int) -> pd.Series: return EMAIndicator(s, window=n).ema_indicator()
def rsi(s: pd.Series, n: int=14) -> pd.Series: return RSIIndicator(s, window=n).rsi()
def obv(c: pd.Series, v: pd.Series) -> pd.Series: return OnBalanceVolumeIndicator(c, v.fillna(0)).on_balance_volume()

def slope(series: pd.Series, lookback=24) -> float:
    if len(series) < lookback: return 0.0
    y = series.iloc[-lookback:]
    x = np.arange(len(y))
    m, _ = np.polyfit(x, y, 1)
    return float(m)

def volume_profile(df: pd.DataFrame, bins=30) -> Tuple[float, np.ndarray, np.ndarray]:
    prices = (df['High']+df['Low']+df['Close'])/3
    vols = df['Volume'].fillna(0)
    hist, edges = np.histogram(prices, bins=bins, weights=vols)
    centers = (edges[:-1] + edges[1:]) / 2
    hvn = float(centers[np.argmax(hist)]) if hist.sum() > 0 else float(prices.iloc[-1])
    return hvn, centers, hist

def swing(df: pd.DataFrame, lookback=200) -> Dict[str, Any]:
    w = df.tail(lookback)
    hi = w['High'].max(); lo = w['Low'].min()
    return {"hi": float(hi), "lo": float(lo)}

def fib_levels(hi: float, lo: float) -> Dict[str,float]:
    d = hi - lo
    return {
        "0.236": hi - 0.236*d,
        "0.382": hi - 0.382*d,
        "0.500": hi - 0.500*d,
        "0.618": hi - 0.618*d,
        "0.786": hi - 0.786*d, # TP3
    }

def momentum_prob_4h(df_1h: pd.DataFrame) -> float:
    """Approximate 4h momentum using 1h data (RSI + OBV trend + flat 200 bonus)."""
    if df_1h.empty: return 0.5
    close = df_1h['Close']
    r = rsi(close, 14); rs = float(r.iloc[-1]) if not r.empty else 50.0
    e200 = ema(close, 200); s200 = abs(slope(e200, 24))
    flat_bonus = 1.0 if s200 < (close.iloc[-1]*1e-5) else 0.8
    obv_s = slope(obv(close, df_1h['Volume']), 24)
    obv_score = 1.0 if obv_s>0 else 0.7
    rsi_score = (rs-30)/40; rsi_score = max(0, min(1, rsi_score))
    p = 0.6*rsi_score + 0.2*flat_bonus + 0.2*obv_score
    return float(max(0.0, min(1.0, p)))

# ---------------------- SIGNAL ENGINE ---------------------
def confluence_check(df: pd.DataFrame) -> Tuple[Optional[str], pd.DataFrame]:
    c = df.copy()
    for L in (2,5,20,50,75,200): c[f"ema{L}"] = ema(c['Close'], L)
    last = c.iloc[-1]
    s200 = slope(c['ema200'], 48)
    flat = abs(s200) < (last.Close*1e-5)
    hvn, _, _ = volume_profile(c.tail(240), 30)
    near_cross = abs(last.ema20 - last.ema200) / max(1e-9,last.Close) < 0.0025
    hvn_near = (abs(hvn-last.ema20)/last.Close < 0.005) or (abs(hvn-last.ema200)/last.Close < 0.005)
    buy_stack  = last.ema2 > last.ema5 > last.ema20 > last.ema200
    sell_stack = last.ema2 < last.ema5 < last.ema20 < last.ema200
    if flat and near_cross and hvn_near and buy_stack:  return "BUY", c
    if flat and near_cross and hvn_near and sell_stack: return "SELL", c
    return None, c

def build_targets(cooked: pd.DataFrame, side: str) -> List[Tuple[str,float]]:
    sw = swing(cooked, 200); fibs = fib_levels(sw["hi"], sw["lo"])
    close = float(cooked['Close'].iloc[-1])
    ema75 = float(EMAIndicator(cooked['Close'], 75).ema_indicator().iloc[-1])

    if side=="BUY":
        cands = [("TP1",ema75),("0.382",fibs["0.382"]),("0.500",fibs["0.500"]),("0.618",fibs["0.618"]),("0.786",fibs["0.786"])]
        tps = [(n,v) for n,v in cands if v>close]
        if not tps: tps=[("TP1",ema75),("0.786",fibs["0.786"])]
    else:
        cands = [("TP1",ema75),("0.618",fibs["0.618"]),("0.500",fibs["0.500"]),("0.382",fibs["0.382"]),("0.786",fibs["0.786"])]
        tps = [(n,v) for n,v in cands if v<close]
        if not tps: tps=[("TP1",ema75),("0.786",fibs["0.786"])]

    # ensure 0.786 (TP3 core) present
    have_786 = any(n=="0.786" for n,_ in tps)
    if not have_786: tps.append(("0.786",fibs["0.786"]))

    # sort
    if side=="BUY": tps = sorted(list({n:v for n,v in tps}.items()), key=lambda x:x[1])
    else:           tps = sorted(list({n:v for n,v in tps}.items()), key=lambda x:x[1], reverse=True)
    return tps[:5]

def rr_extension(entry: float, sl: float, side: str, rr: float) -> float:
    risk = abs(entry - sl)
    return (entry + rr*risk) if side=="BUY" else (entry - rr*risk)

def analyze(symbol: str, interval: str, balance: float, risk_pct: float, stop_pct: float) -> Optional[Dict[str,Any]]:
    df = fetch(symbol, interval, 500)
    if df.empty: return None
    side, cooked = confluence_check(df)
    if not side: return None

    entry = float(cooked['Close'].iloc[-1])
    sl = entry*(1 - stop_pct) if side=="BUY" else entry*(1 + stop_pct)
    tps = build_targets(cooked, side)

    # Momentum check (4h approx from 1h series)
    df_1h = fetch(symbol, "60m", 600)
    p4h = momentum_prob_4h(df_1h)
    if p4h >= 0.80:
        # Add TP4, TP5 at fixed RR steps if room remains
        for rr in (4.5, 7.0):
            tps.append((f"RR{rr}", rr_extension(entry, sl, side, rr)))
            if len(tps) >= 5: break

    # sizing
    loss_per_unit = entry*stop_pct
    risk_amt = balance*(risk_pct/100.0)
    size = risk_amt/max(1e-12,loss_per_unit)
    trades_left = balance/max(1e-12,risk_amt)

    r4 = lambda x: float(f"{x:.4f}")
    return {
        "symbol": symbol, "interval": interval, "side": side,
        "entry": r4(entry), "sl": r4(sl),
        "tps": [(n, r4(v)) for n,v in tps],
        "risk_pct": risk_pct, "stop_pct": stop_pct,
        "risk_amt": round(risk_amt,2), "size_hint": round(size,4),
        "trades_left": round(trades_left,1),
        "p4h": round(p4h,2),
        "df_cooked": cooked
    }

# ------------------------- CHARTS -------------------------
def chart_2d_png(sig: Dict[str,Any], title: str) -> io.BytesIO:
    df = sig["df_cooked"].copy().tail(240)
    entry, sl = sig["entry"], sig["sl"]
    side = sig["side"]

    fig = go.Figure()
    fig.add_candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price")
    for L, color, name in [(2,"#bbb","EMA2"),(5,"#999","EMA5"),(20,"#0066FF","EMA20"),
                           (50,"#9cf","EMA50"),(75,"#00bbbb","EMA75"),(200,"#000","EMA200")]:
        col=f"ema{L}"
        if col not in df.columns: df[col]=ema(df['Close'], L)
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=name, line=dict(width=1.3, color=color)))

    # SL purple zone
    y0,y1=(sl,entry) if side=="BUY" else (entry,sl)
    fig.add_shape(type="rect", x0=df.index[-180], x1=df.index[-1], y0=min(y0,y1), y1=max(y0,y1),
                  line=dict(width=0), fillcolor="rgba(128,0,128,0.18)", layer="below")
    # TP layers
    colors=["#00000055","#FF000055","#C0A00055","#00AA0055"]
    for i,(n,p) in enumerate(sig["tps"][:4]):
        low,high=(entry,p) if side=="BUY" else (p,entry)
        fig.add_shape(type="rect", x0=df.index[-180], x1=df.index[-1], y0=min(low,high), y1=max(low,high),
                      line=dict(width=0), fillcolor=colors[i], layer="below")

    # lines
    fig.add_hline(y=entry, line=dict(color="#555", width=1, dash="dot"), annotation_text=f"Entry {entry:.4f}")
    for i,(n,p) in enumerate(sig["tps"], start=1):
        fig.add_hline(y=p, line=dict(color="#222", width=1), annotation_text=f"TP{i} {n}: {p:.4f}")
    fig.add_hline(y=sl, line=dict(color="#800080", width=2), annotation_text=f"SL {sl:.4f}")

    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=720, width=1100, template="plotly_white")
    buf = io.BytesIO(); fig.write_image(buf, format="png", scale=2); buf.seek(0); return buf

def chart_3d_html(df: pd.DataFrame, symbol: str) -> str:
    d = df.tail(400).copy()
    t = np.arange(len(d))
    price = d['Close'].values
    vol = np.log1p(d['Volume'].fillna(0).values.astype(float))
    fig = go.Figure(data=[go.Scatter3d(x=t,y=price,z=vol,mode='lines',line=dict(width=6))])
    fig.update_layout(title=f"{symbol} · 3D Price/Volume Path",
                      scene=dict(xaxis_title="Bars",yaxis_title="Price",zaxis_title="log(Vol+1)"))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")

# -------------------- TELEGRAM POSTING -------------------
async def post_signal(context: ContextTypes.DEFAULT_TYPE, chat_id: str, sig: Dict[str,Any], vip: bool, hold_minutes: int):
    sess = session_name()
    title = f"{sig['symbol']} · {sig['side']} · {sig['interval']} · {sess}"
    png = chart_2d_png(sig, title)
    tps_txt=[]
    for i,(n,p) in enumerate(sig["tps"], start=1):
        tag = " (TP3 core)" if n in ("0.786","TP3") or i==3 else ""
        tps_txt.append(f"TP{i} {n}: {p:.4f}{tag}")
    cap = textwrap.dedent(f"""
    {'💎 VIP' if vip else '📣 MAIN'} · {sess}
    {sig['symbol']} · {sig['side']} · Scan: {sig['interval']} · Max Hold: {hold_minutes}m

    Entry: {sig['entry']:.4f}
    SL: {sig['sl']:.4f}  (stop {sig['stop_pct']*100:.3f}% of price)
    Risk: {sig['risk_pct']}% = ${sig['risk_amt']:.2f} · Size≈{sig['size_hint']} · Trades left≈{sig['trades_left']}
    4h momentum follow-through: {int(sig['p4h']*100)}%

    """ + "\n".join(tps_txt) + """

    Reactions: 👍🔥💡  |  😐🤔🍋  |  👎😡💧
    Rules: 20≈200 (flat) + HVN near + EMA(2/5/20/200) stack · TP3 fixed @ Fib 0.786.
    """).strip()

    msg = await context.bot.send_photo(chat_id=chat_id, photo=InputFile(png, filename=f"{sig['symbol']}.png"), caption=cap)

    # register expiry
    sig_id = f"{msg.chat_id}:{msg.message_id}"
    expire_at = now_sast() + dt.timedelta(minutes=hold_minutes)
    STATE["active_signals"][sig_id] = {"chat_id": msg.chat_id, "message_id": msg.message_id,
                                       "symbol": sig["symbol"], "side": sig["side"], "expire_at": expire_at}

# ------------------------- SCANNERS -----------------------
def symbols_list(which: str) -> List[str]:
    env_key = "SYMBOLS_MAIN" if which=="main" else "SYMBOLS_VIP"
    env_val = os.getenv(env_key, "").strip()
    if env_val:
        syms = [s.strip() for s in env_val.split(",") if s.strip()]
    else:
        syms = DEFAULT_SYMBOLS.copy()
    return syms

async def run_tier(context: ContextTypes.DEFAULT_TYPE, tier_map: Dict[str,int], chat_id: str, vip: bool):
    for interval, hold_m in tier_map.items():
        # keep watchlists in STATE so /addsymbol and /removesymbol are effective
        lst = STATE["vip_symbols"] if vip else STATE["main_symbols"]
        if not lst:
            lst = symbols_list("vip" if vip else "main")
            if vip: STATE["vip_symbols"]=lst
            else:   STATE["main_symbols"]=lst

        for sym in lst:
            sig = analyze(sym, interval, STATE["balance"], STATE["risk_pct"], STATE["stop_pct"])
            if sig:
                await post_signal(context, chat_id, sig, vip=vip, hold_minutes=hold_m)
            await asyncio.sleep(1.0)  # be gentle to API

async def sweep_expired(context: ContextTypes.DEFAULT_TYPE):
    now = now_sast()
    to_del = []
    for sid, info in STATE["active_signals"].items():
        if now >= info["expire_at"]:
            try:
                await context.bot.send_message(chat_id=info["chat_id"],
                    text=f"⏳ Signal expired: {info['symbol']} · {info['side']}")
            except: pass
            to_del.append(sid)
    for sid in to_del: STATE["active_signals"].pop(sid, None)

# --------------------- TELEGRAM COMMANDS -----------------
async def cmd_start(update: Update, _: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🎯 Bullseye online. /status /signal SYMBOL /addsymbol TCKR /removesymbol TCKR /listsymbols /risk N /setbalance N /dailybias")

async def cmd_status(update: Update, _: ContextTypes.DEFAULT_TYPE):
    msg = textwrap.dedent(f"""
    ⏱ {now_sast():%Y-%m-%d %H:%M} {session_name()}
    Balance: ${STATE['balance']:.2f} | Risk: {STATE['risk_pct']}% | StopPct: {STATE['stop_pct']*100:.3f}%
    Main chat: {MAIN_CHAT or '—'} | VIP: {VIP_CHAT or '—'}
    Main symbols: {', '.join(STATE['main_symbols'] or symbols_list('main'))}
    VIP symbols: {', '.join(STATE['vip_symbols'] or symbols_list('vip'))}
    """).strip()
    await update.message.reply_text(msg)

async def cmd_setbalance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        v = float(context.args[0]); v=max(10.0,v); STATE["balance"]=v
        await update.message.reply_text(f"✅ Balance set to ${v:.2f}")
    except: await update.message.reply_text("Usage: /setbalance 15000")

async def cmd_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        v = float(context.args[0]); v=max(0.05,min(15.0,v)); STATE["risk_pct"]=v
        await update.message.reply_text(f"✅ Risk set to {v}%")
    except: await update.message.reply_text("Usage: /risk 5")

async def cmd_addsymbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        sym = context.args[0]
    except:
        return await update.message.reply_text("Usage: /addsymbol EURUSD=X [main|vip]")
    tier = (context.args[1].lower() if len(context.args)>1 else "main")
    lst = STATE["vip_symbols"] if tier=="vip" else STATE["main_symbols"]
    if not lst: lst = symbols_list("vip" if tier=="vip" else "main")
    if sym not in lst:
        lst.append(sym)
        if tier=="vip": STATE["vip_symbols"]=lst
        else: STATE["main_symbols"]=lst
        await update.message.reply_text(f"✅ Added {sym} to {tier.upper()} list.")
    else:
        await update.message.reply_text(f"ℹ️ {sym} already in {tier.upper()} list.")

async def cmd_removesymbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        sym = context.args[0]
    except:
        return await update.message.reply_text("Usage: /removesymbol EURUSD=X [main|vip]")
    tier = (context.args[1].lower() if len(context.args)>1 else "main")
    lst = STATE["vip_symbols"] if tier=="vip" else STATE["main_symbols"]
    if not lst: lst = symbols_list("vip" if tier=="vip" else "main")
    if sym in lst:
        lst.remove(sym)
        if tier=="vip": STATE["vip_symbols"]=lst
        else: STATE["main_symbols"]=lst
        await update.message.reply_text(f"🗑 Removed {sym} from {tier.upper()} list.")
    else:
        await update.message.reply_text(f"❌ {sym} not found in {tier.upper()} list.")

async def cmd_listsymbols(update: Update, _: ContextTypes.DEFAULT_TYPE):
    mains = STATE["main_symbols"] or symbols_list("main")
    vips  = STATE["vip_symbols"]  or symbols_list("vip")
    await update.message.reply_text(f"Main ({len(mains)}): {', '.join(mains)}\nVIP ({len(vips)}): {', '.join(vips)}")

async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await update.message.reply_text("Usage: /signal EURUSD=X [2m|5m|15m|30m|45m|60m|240m]")
    sym = context.args[0]
    interval = (context.args[1] if len(context.args)>1 else "15m")
    await update.message.reply_text(f"Analyzing {sym} @ {interval}…")
    sig = analyze(sym, interval, STATE["balance"], STATE["risk_pct"], STATE["stop_pct"])
    if not sig: return await update.message.reply_text("No valid setup right now.")
    hold = MAIN_TIERS.get(interval) or VIP_TIERS.get(interval) or 60
    await post_signal(context, update.effective_chat.id, sig, vip=False, hold_minutes=hold)

async def cmd_dailybias(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await post_daily_bias(context, update.effective_chat.id, "Daily Bias")

# -------------------- DAILY BIAS (summary) ----------------
async def post_daily_bias(context: ContextTypes.DEFAULT_TYPE, chat_id: str, title="Daily Bias"):
    lines=[f"🗞 {title} · {now_sast():%Y-%m-%d}"]
    pool = (STATE["main_symbols"] or symbols_list("main"))[:3] + (STATE["vip_symbols"] or symbols_list("vip"))[:3]
    for sym in pool:
        df = fetch(sym, "60m", 240)
        if df.empty:
            lines.append(f"{sym}: no data"); continue
        e200 = ema(df['Close'],200).iloc[-1]
        close = df['Close'].iloc[-1]
        rs = rsi(df['Close'],14).iloc[-1]
        p = int(momentum_prob_4h(df)*100)
        bias = "Bull" if close>e200 else "Bear"
        lines.append(f"{sym}: {bias} | RSI {rs:.0f} | 4h follow-through {p}%")
    await context.bot.send_message(chat_id=chat_id, text="\n".join(lines))

# ------------------------- SCHEDULER ----------------------
def schedule_all(app: Application):
    sched = AsyncIOScheduler(timezone=str(TZ))

    # MAIN tiers (continuous)
    for minute in (0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58):
        sched.add_job(lambda: app.create_task(run_tier(app, {"2m": MAIN_TIERS["2m"]}, MAIN_CHAT, False)),
                      "cron", minute=minute)
    sched.add_job(lambda: app.create_task(run_tier(app, {"5m": MAIN_TIERS["5m"]}, MAIN_CHAT, False)),
                  "cron", minute="*/5")
    sched.add_job(lambda: app.create_task(run_tier(app, {"15m": MAIN_TIERS["15m"]}, MAIN_CHAT, False)),
                  "cron", minute="*/15")
    sched.add_job(lambda: app.create_task(run_tier(app, {"30m": MAIN_TIERS["30m"]}, MAIN_CHAT, False)),
                  "cron", minute="*/30")

    # VIP tiers
    sched.add_job(lambda: app.create_task(run_tier(app, {"45m": VIP_TIERS["45m"]}, VIP_CHAT, True)),
                  "cron", minute="0,45")
    sched.add_job(lambda: app.create_task(run_tier(app, {"60m": VIP_TIERS["60m"]}, VIP_CHAT, True)),
                  "cron", minute=0)
    # every 4 hours on the hour
    sched.add_job(lambda: app.create_task(run_tier(app, {"240m": VIP_TIERS["240m"]}, VIP_CHAT, True)),
                  "cron", minute=0, hour="0,4,8,12,16,20")

    # Expiry sweeper every 2 minutes
    sched.add_job(lambda: app.create_task(sweep_expired(app)), "cron", minute="*/2")

    # Daily bias 07:30 SAST (main) & 07:32 (vip)
    sched.add_job(lambda: app.create_task(post_daily_bias(app, MAIN_CHAT, "Daily Bias (Main)")), "cron", hour=7, minute=30)
    if VIP_CHAT:
        sched.add_job(lambda: app.create_task(post_daily_bias(app, VIP_CHAT, "Daily Bias (VIP)")), "cron", hour=7, minute=32)

    sched.start()

# --------------------------- WEB --------------------------
server = Flask(__name__)

HTML = """
<!doctype html><html><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{{title}}</title></head>
<body style="background:#101318;color:#e6e6e6;font-family:Inter,system-ui,Arial">
<div style="max-width:1100px;margin:24px auto">
<h2 style="font-weight:700">{{title}}</h2>
<div>{{desc}}</div>
<div id="plot">{{plot|safe}}</div>
<p style="opacity:.7;margin-top:16px">Rotate / pinch 3D. Session: {{session}} · SAST.</p>
</div></body></html>
"""

@server.get("/")
def root():
    return "Bullseye running."

@server.get("/chart3d")
def chart3d_endpoint():
    symbol = request.args.get("symbol","BTC-USD")
    interval = request.args.get("interval","60m")
    df = fetch(symbol, interval, 600)
    if df.empty: return "No data", 404
    return render_template_string(HTML, title=f"3D · {symbol}",
                                  desc=f"Price–Volume 3D path (interval {interval}).",
                                  plot=chart_3d_html(df, symbol),
                                  session=session_name())

# --------------------------- MAIN -------------------------
def run_web():
    from waitress import serve
    port = int(os.getenv("PORT","8080"))
    serve(server, host="0.0.0.0", port=port)

def main():
    if not BOT_TOKEN: raise SystemExit("Missing TELEGRAM_BOT_TOKEN")
    app = Application.builder().token(BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("setbalance", cmd_setbalance))
    app.add_handler(CommandHandler("risk", cmd_risk))
    app.add_handler(CommandHandler("addsymbol", cmd_addsymbol))
    app.add_handler(CommandHandler("removesymbol", cmd_removesymbol))
    app.add_handler(CommandHandler("listsymbols", cmd_listsymbols))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("dailybias", cmd_dailybias))

    async def on_start(_):
        try:
            if MAIN_CHAT: await app.bot.send_message(chat_id=MAIN_CHAT, text="✅ Bullseye online (MAIN).")
            if VIP_CHAT:  await app.bot.send_message(chat_id=VIP_CHAT,  text="💎 Bullseye online (VIP).")
        except Exception as e: print("Startup notify error:", e)

    app.post_init = on_start

    # schedule all jobs
    schedule_all(app)

    # start web (3D)
    threading.Thread(target=run_web, daemon=True).start()

    # polling
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
