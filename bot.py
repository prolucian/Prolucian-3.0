import os, asyncio, logging, math
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
import psycopg2
from psycopg2.extras import RealDictCursor
from telegram import Update, InputFile
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import plotly.graph_objects as go
from io import BytesIO

# 🔑 Load secrets from .env
from dotenv import load_dotenv
load_dotenv()

# =================== CONFIG ===================
TOKEN        = os.getenv("TELEGRAM_BOT_TOKEN")
MAIN_CHAT    = int(os.getenv("TELEGRAM_MAIN_CHAT","0"))
VIP_CHAT     = int(os.getenv("TELEGRAM_VIP_CHAT","0"))
DATABASE_URL = os.getenv("DATABASE_URL","")
TZ           = timezone(timedelta(hours=2)) # Johannesburg

DEFAULT_BALANCE  = float(os.getenv("DEFAULT_BALANCE","1000"))
DEFAULT_RISK_PCT = float(os.getenv("DEFAULT_RISK_PCT","5"))
STOP_PCT         = float(os.getenv("STOP_PCT","0.05"))
RR_FINAL         = float(os.getenv("RR_FINAL","9"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bullseye")

# =================== DB ===================
def db():
    return psycopg2.connect(DATABASE_URL, sslmode="require", cursor_factory=RealDictCursor)

def init_db():
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id SERIAL PRIMARY KEY,
            symbol TEXT, timeframe TEXT, audience TEXT,
            side TEXT, entry FLOAT, stop_loss FLOAT,
            tp1 FLOAT, tp2 FLOAT, tp3 FLOAT,
            lots FLOAT, risk_pct FLOAT, balance FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""")
        conn.commit()

def save_signal(rec: dict):
    with db() as conn, conn.cursor() as cur:
        fields = ",".join(rec.keys())
        ph     = ",".join(["%s"]*len(rec))
        cur.execute(f"INSERT INTO signals ({fields}) VALUES ({ph});", tuple(rec.values()))
        conn.commit()

# =================== Strategy ===================
def velez_signal(df: pd.DataFrame):
    close = df["Close"]
    sma2, sma5 = close.rolling(2).mean(), close.rolling(5).mean()
    sma20, sma75, sma200 = close.rolling(20).mean(), close.rolling(75).mean(), close.rolling(200).mean()
    if len(close) < 210: return None

    c = float(close.iloc[-1])
    m2,m5,m20,m75,m200 = map(float,[sma2.iloc[-1],sma5.iloc[-1],sma20.iloc[-1],sma75.iloc[-1],sma200.iloc[-1]])
    if any(pd.isna(x) for x in [m2,m5,m20,m75,m200]): return None

    m200_prev = float(sma200.iloc[-5]) if not pd.isna(sma200.iloc[-5]) else m200
    flat200 = abs((m200-m200_prev)/m200_prev) < 0.001
    near = abs(m20-m200) < 0.002*c

    if flat200 and near:
        if m2 > m5 > m20 > m200:
            return dict(side="BUY", entry=c,
                        sl=c*(1-STOP_PCT/100),
                        tp1=m75, tp2=c*1.01, tp3=c*(1+RR_FINAL*STOP_PCT/100))
        if m2 < m5 < m20 < m200:
            return dict(side="SELL", entry=c,
                        sl=c*(1+STOP_PCT/100),
                        tp1=m75, tp2=c*0.99, tp3=c*(1-RR_FINAL*STOP_PCT/100))
    return None

# =================== Risk ===================
def lot_size(balance, risk_pct, entry, sl, pip_value=1):
    risk_amount = balance*(risk_pct/100)
    stop_dist = abs(entry-sl)
    return round(risk_amount/max(stop_dist,1e-9),4)

def can_blow(balance, risk_pct):
    r, n, b = risk_pct/100,0,balance
    while b>1 and n<1000: b*=(1-r);n+=1
    return n

# =================== Message + 3D Chart ===================
def make_chart(df, sig, symbol):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"]
    )])
    fig.add_hline(y=sig["entry"], line_color="blue")
    fig.add_hline(y=sig["sl"], line_color="red")
    fig.add_hline(y=sig["tp1"], line_color="green")
    fig.add_hline(y=sig["tp2"], line_color="orange")
    fig.add_hline(y=sig["tp3"], line_color="purple")
    buf = BytesIO()
    fig.write_image(buf, format="png")
    buf.seek(0)
    return buf

def format_msg(symbol, tf, audience, sig, balance, risk_pct):
    lots = lot_size(balance, risk_pct, sig["entry"], sig["sl"])
    blows = can_blow(balance, risk_pct)
    return (
f"🎯 Bullseye Monster — {audience}\n"
f"⏱ TF {tf} | {symbol}\n"
f"{sig['side']} | Entry {sig['entry']:.4f} | SL {sig['sl']:.4f}\n"
f"TP1 {sig['tp1']:.4f} | TP2 {sig['tp2']:.4f} | TP3 {sig['tp3']:.4f}\n"
f"Risk {risk_pct}% → Lots {lots} | Blows @ {blows}\n"
)# =================== Universe & Volatility ===================
UNIVERSE = list(set([
    # FX majors & minors
    "EURUSD=X","GBPUSD=X","USDJPY=X","USDCHF=X","USDCAD=X","AUDUSD=X","NZDUSD=X",
    "EURGBP=X","EURJPY=X","GBPJPY=X","AUDJPY=X","EURAUD=X","CADJPY=X","CHFJPY=X",
    # Metals, energy, ags
    "GC=F","SI=F","HG=F","CL=F","NG=F","PL=F","PA=F","ZC=F","ZW=F","ZS=F","KC=F","SB=F","CC=F",
    # Indices
    "^GSPC","^NDX","^DJI","^RUT","^VIX","^N225","^FTSE","^GDAXI","^FCHI","^HSI",
    # Crypto
    "BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","ADA-USD","DOGE-USD",
]))

async def most_volatile_top30() -> list:
    ranked = []
    for s in UNIVERSE:
        try:
            d = yf.download(s, period="3mo", interval="1d", progress=False, auto_adjust=True)
            if len(d) < 20:
                continue
            atr = ta.atr(d["High"], d["Low"], d["Close"], length=14).iloc[-1]
            vol = float(atr / max(1e-9, d["Close"].iloc[-1]))
            ranked.append((s, vol))
        except Exception as e:
            log.warning(f"Vol rank fail {s}: {e}")
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [s for s,_ in ranked[:30]]

# =================== 3D Chart Sending ===================
async def send_signal_with_chart(app, chat_id, symbol, timeframe, audience, sig, balance, risk_pct):
    # pick an interval & period large enough for visuals
    interval = {"2m":"2m","5m":"5m","15m":"15m","30m":"30m","45m":"30m","1h":"60m","4h":"60m"}.get(timeframe,"15m")
    period   = "60d" if timeframe in ("1h","4h") else "10d"
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    if len(df) == 0:
        await app.bot.send_message(chat_id=chat_id, text="(chart unavailable right now)")
        return
    buf = make_chart(df.tail(300), sig, symbol)
    caption = format_msg(symbol, timeframe, audience, sig, DEFAULT_BALANCE, DEFAULT_RISK_PCT)
    await app.bot.send_photo(chat_id=chat_id, photo=InputFile(buf, filename=f"{symbol}_{timeframe}.png"),
                             caption=caption)

# =================== Scanner ===================
async def scan_once(app, symbols, timeframe, audience, chat_id, balance, risk_pct):
    for sym in symbols:
        try:
            interval = {"2m":"2m","5m":"5m","15m":"15m","30m":"30m","45m":"30m","1h":"60m","4h":"60m"}[timeframe]
            period   = "60d" if timeframe in ("1h","4h") else "10d"
            d = yf.download(sym, period=period, interval=interval, progress=False, auto_adjust=True)
            if len(d) < 210:
                continue
            sig = velez_signal(d)
            if sig:
                rec = dict(symbol=sym, timeframe=timeframe, audience=audience, side=sig["side"],
                           entry=sig["entry"], stop_loss=sig["sl"], tp1=sig["tp1"], tp2=sig["tp2"], tp3=sig["tp3"],
                           lots=lot_size(balance, risk_pct, sig["entry"], sig["sl"]),
                           risk_pct=risk_pct, balance=balance)
                save_signal(rec)
                await send_signal_with_chart(app, chat_id, sym, timeframe, audience, sig, balance, risk_pct)
                await asyncio.sleep(0.5)
        except Exception as e:
            log.error(f"scan_once error {sym} {timeframe}: {e}")

# =================== Reports ===================
def report_df(period: str) -> pd.DataFrame:
    with db() as conn, conn.cursor() as cur:
        if period == "daily":
            cur.execute("SELECT * FROM signals WHERE created_at >= NOW() - INTERVAL '1 day' ORDER BY created_at DESC;")
        elif period == "weekly":
            cur.execute("SELECT * FROM signals WHERE created_at >= NOW() - INTERVAL '7 days' ORDER BY created_at DESC;")
        elif period == "monthly":
            cur.execute("SELECT * FROM signals WHERE created_at >= NOW() - INTERVAL '30 days' ORDER BY created_at DESC;")
        else:
            cur.execute("SELECT * FROM signals ORDER BY created_at DESC;")
        rows = cur.fetchall()
    return pd.DataFrame(rows)

def summarize_report(df: pd.DataFrame, label: str) -> str:
    if df.empty:
        return f"📊 {label}: No signals recorded."
    by_tf = df.groupby("timeframe").size().sort_values(ascending=False)
    by_side = df.groupby("side").size()
    total = len(df)
    txt = [f"📊 <b>{label}</b>", f"Total signals: <b>{total}</b>"]
    if not by_tf.empty:
        txt.append("By TF: " + ", ".join([f"{i}:{int(v)}" for i,v in by_tf.items()]))
    if not by_side.empty:
        txt.append("By Side: " + ", ".join([f"{i}:{int(v)}" for i,v in by_side.items()]))
    # crude “strike rate” placeholder: count TP1 reachable if tp1 beyond sl distance
    # (Real PnL requires trade tracking; here we approximate potential)
    est_hits = 0
    for _,r in df.iterrows():
        if r["side"] == "BUY" and r["tp1"] > r["entry"] and r["stop_loss"] < r["entry"]:
            est_hits += 1
        if r["side"] == "SELL" and r["tp1"] < r["entry"] and r["stop_loss"] > r["entry"]:
            est_hits += 1
    txt.append(f"Est. TP1 reach probability: <b>{(est_hits/total*100):.1f}%</b> (approx)")
    return "\n".join(txt)

async def report_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    period = (ctx.args[0] if ctx.args else "daily").lower()
    if period not in ("daily","weekly","monthly","all"):
        period = "daily"
    df = report_df(period)
    text = summarize_report(df, period.capitalize())
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

# =================== Billing (Stripe optional) ===================
STRIPE_SECRET   = os.getenv("STRIPE_SECRET","")
PRICE_MAIN_ID   = os.getenv("PRICE_MAIN_ID","")   # e.g., price_xxx for $2.99/mo or $30/yr
PRICE_VIP_ID    = os.getenv("PRICE_VIP_ID","")    # e.g., price_yyy for $9.99/mo or $91/yr
SUCCESS_URL     = os.getenv("SUCCESS_URL","https://t.me/")  # fallback

def init_billing_tables():
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id BIGINT PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            last_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS subscriptions (
            user_id BIGINT REFERENCES users(user_id) ON DELETE CASCADE,
            tier TEXT, -- 'main' or 'vip'
            expires_at TIMESTAMP,
            PRIMARY KEY(user_id, tier)
        );""")
        conn.commit()

def upsert_user(u):
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
        INSERT INTO users(user_id,username,first_name,last_name)
        VALUES(%s,%s,%s,%s)
        ON CONFLICT (user_id) DO UPDATE SET username=EXCLUDED.username;""",
        (u.id, u.username, u.first_name, u.last_name))
        conn.commit()

def set_subscription(user_id: int, tier: str, days: int):
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
        INSERT INTO subscriptions(user_id,tier,expires_at)
        VALUES(%s,%s,NOW() + (%s || ' days')::interval)
        ON CONFLICT (user_id,tier) DO UPDATE
        SET expires_at = GREATEST(subscriptions.expires_at, NOW()) + (%s || ' days')::interval;
        """, (user_id, tier, days, days))
        conn.commit()

def is_active(user_id: int, tier: str) -> bool:
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM subscriptions WHERE user_id=%s AND tier=%s AND expires_at > NOW();", (user_id,tier))
        return cur.fetchone() is not None

async def subscribe_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Creates a Stripe Checkout link if STRIPE_SECRET present; else instructions."""
    upsert_user(update.effective_user)
    plan = (ctx.args[0].lower() if ctx.args else "main")  # 'main' or 'vip'
    period = (ctx.args[1].lower() if len(ctx.args)>1 else "monthly")  # 'monthly' or 'yearly'
    if plan not in ("main","vip"): plan="main"
    if period not in ("monthly","yearly"): period="monthly"

    if not STRIPE_SECRET:
        await update.message.reply_text(
            "Payments API not configured.\nAsk admin to add you manually:\n"
            "• /grantvip <user_id> <days>\n• /grantmain <user_id> <days>\n"
            "Or set STRIPE_SECRET, PRICE_MAIN_ID, PRICE_VIP_ID in .env.")
        return

    import stripe
    stripe.api_key = STRIPE_SECRET
    price = PRICE_MAIN_ID if plan=="main" else PRICE_VIP_ID
    if not price:
        await update.message.reply_text("Stripe price ID missing in .env.")
        return

    # For monthly vs yearly, you should create two separate price IDs in Stripe and switch here if needed.
    # For simplicity we reuse PRICE_*_ID regardless of period (configure in Stripe per usage).
    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": price, "quantity": 1}],
        success_url=SUCCESS_URL + "?session_id={CHECKOUT_SESSION_ID}",
        cancel_url=SUCCESS_URL,
        metadata={"tg_user_id": str(update.effective_user.id), "plan": plan, "period": period}
    )
    await update.message.reply_text(
        f"🔐 Pay for <b>{plan.upper()}</b> ({period})\n"
        f"Checkout: {session.url}\n\nAfter payment, send:\n/redeem {session.id}",
        parse_mode=ParseMode.HTML
    )

async def redeem_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """User pastes session_id; we verify payment and activate sub."""
    if not STRIPE_SECRET:
        await update.message.reply_text("Stripe not configured.")
        return
    if not ctx.args:
        await update.message.reply_text("Usage: /redeem <session_id>")
        return
    session_id = ctx.args[0]
    import stripe
    stripe.api_key = STRIPE_SECRET
    try:
        sess = stripe.checkout.Session.retrieve(session_id, expand=["subscription","customer"])
        if sess.payment_status != "paid":
            await update.message.reply_text("Payment not completed yet.")
            return
        plan = sess.metadata.get("plan","main")
        period = sess.metadata.get("period","monthly")
        days = 30 if period=="monthly" else 365
        user_id = int(sess.metadata.get("tg_user_id") or update.effective_user.id)
        upsert_user(update.effective_user)
        set_subscription(user_id, plan, days)
        await update.message.reply_text(f"✅ {plan.upper()} activated for {days} days.")
    except Exception as e:
        await update.message.reply_text(f"Redeem error: {e}")

async def status_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    upsert_user(u)
    active_main = is_active(u.id, "main")
    active_vip  = is_active(u.id, "vip")
    await update.message.reply_text(
        f"👤 @{u.username or u.id}\n"
        f"Main: {'ACTIVE' if active_main else '—'}\n"
        f"VIP:  {'ACTIVE' if active_vip else '—'}"
    )

# Admin helpers (manually grant)
ADMIN_IDS = set([ ])  # optionally add your Telegram user id here for admin commands

async def grantvip_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("Not authorized.")
        return
    if len(ctx.args) < 2:
        await update.message.reply_text("Usage: /grantvip <user_id> <days>")
        return
    uid = int(ctx.args[0]); days = int(ctx.args[1])
    set_subscription(uid, "vip", days)
    await update.message.reply_text(f"VIP granted to {uid} for {days} days.")

async def grantmain_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("Not authorized.")
        return
    if len(ctx.args) < 2:
        await update.message.reply_text("Usage: /grantmain <user_id> <days>")
        return
    uid = int(ctx.args[0]); days = int(ctx.args[1])
    set_subscription(uid, "main", days)
    await update.message.reply_text(f"MAIN granted to {uid} for {days} days.")

# =================== Loops ===================
async def loop_main(app):
    while True:
        try:
            top = await most_volatile_top30()
            await scan_once(app, top, "2m",  "MAIN", MAIN_CHAT, DEFAULT_BALANCE, DEFAULT_RISK_PCT)
            await scan_once(app, top, "5m",  "MAIN", MAIN_CHAT, DEFAULT_BALANCE, DEFAULT_RISK_PCT)
            await scan_once(app, top, "15m", "MAIN", MAIN_CHAT, DEFAULT_BALANCE, DEFAULT_RISK_PCT)
            await scan_once(app, top, "30m", "MAIN", MAIN_CHAT, DEFAULT_BALANCE, DEFAULT_RISK_PCT)
        except Exception as e:
            log.error(f"loop_main: {e}")
        await asyncio.sleep(60)

async def loop_vip(app):
    while True:
        try:
            top = await most_volatile_top30()
            await scan_once(app, top, "45m", "VIP", VIP_CHAT, DEFAULT_BALANCE, DEFAULT_RISK_PCT)
            await asyncio.sleep(60)
            await scan_once(app, top, "1h",  "VIP", VIP_CHAT, DEFAULT_BALANCE, DEFAULT_RISK_PCT)
            await asyncio.sleep(60)
            await scan_once(app, top, "4h",  "VIP", VIP_CHAT, DEFAULT_BALANCE, DEFAULT_RISK_PCT)
        except Exception as e:
            log.error(f"loop_vip: {e}")
        await asyncio.sleep(60*30)

# =================== Commands (Part 2) ===================
async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    upsert_user(update.effective_user)
    await update.message.reply_text(
        "🚀 Bullseye Monster online.\n"
        "/scan SYMBOL  – manual signal check\n"
        "/report [daily|weekly|monthly|all]\n"
        "/subscribe [main|vip] [monthly|yearly]\n"
        "/redeem <session_id>\n"
        "/status"
    )

async def scan_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    symbol = (ctx.args[0] if ctx.args else "EURUSD=X").upper()
    try:
        d = yf.download(symbol, period="10d", interval="15m", progress=False, auto_adjust=True)
        if len(d) < 210:
            await update.message.reply_text("Not enough data.")
            return
        sig = velez_signal(d)
        if not sig:
            await update.message.reply_text("No valid confluence right now.")
            return
        rec = dict(symbol=symbol, timeframe="15m", audience="MANUAL", side=sig["side"],
                   entry=sig["entry"], stop_loss=sig["sl"], tp1=sig["tp1"], tp2=sig["tp2"], tp3=sig["tp3"],
                   lots=lot_size(DEFAULT_BALANCE, DEFAULT_RISK_PCT, sig["entry"], sig["sl"]),
                   risk_pct=DEFAULT_RISK_PCT, balance=DEFAULT_BALANCE)
        save_signal(rec)
        await send_signal_with_chart(ctx.application, update.effective_chat.id, symbol, "15m", "MANUAL", sig, DEFAULT_BALANCE, DEFAULT_RISK_PCT)
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

# =================== MAIN ===================
async def main():
    if not (TOKEN and MAIN_CHAT and VIP_CHAT and DATABASE_URL):
        raise RuntimeError("Missing env vars (.env): TELEGRAM_BOT_TOKEN, TELEGRAM_MAIN_CHAT, TELEGRAM_VIP_CHAT, DATABASE_URL")
    init_db()
    init_billing_tables()

    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("scan", scan_cmd))
    app.add_handler(CommandHandler("report", report_cmd))
    app.add_handler(CommandHandler("subscribe", subscribe_cmd))
    app.add_handler(CommandHandler("redeem", redeem_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("grantvip", grantvip_cmd))
    app.add_handler(CommandHandler("grantmain", grantmain_cmd))

    asyncio.create_task(loop_main(app))
    asyncio.create_task(loop_vip(app))
    log.info("Bullseye Monster started.")
    await app.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
