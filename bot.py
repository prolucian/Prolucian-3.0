import os, asyncio, logging, time, math
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
import psycopg2
from psycopg2.extras import RealDictCursor
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ========= CONFIG via Railway Variables =========
TOKEN            = os.getenv("TELEGRAM_BOT_TOKEN")           # required
MAIN_CHAT        = int(os.getenv("TELEGRAM_MAIN_CHAT","0"))  # required
VIP_CHAT         = int(os.getenv("TELEGRAM_VIP_CHAT","0"))   # required
DATABASE_URL     = os.getenv("DATABASE_URL","")              # required (Railway Postgres)
TZ               = timezone(timedelta(hours=2))              # Africa/Johannesburg (UTC+2)

# Risk settings (can be overridden with secrets if you want)
DEFAULT_BALANCE  = float(os.getenv("DEFAULT_BALANCE","1000"))
DEFAULT_RISK_PCT = float(os.getenv("DEFAULT_RISK_PCT","5"))    # 0.05%–15% allowed; 5% default
STOP_PCT         = float(os.getenv("STOP_PCT","0.05"))         # 0.05% default stop distance
RR_FINAL         = float(os.getenv("RR_FINAL","9"))            # TP3 = 1:9

# ========= LOGGING =========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("bullseye")

# ========= DB =========
def db():
    return psycopg2.connect(DATABASE_URL, sslmode="require", cursor_factory=RealDictCursor)

def init_db():
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id SERIAL PRIMARY KEY,
            symbol TEXT,
            audience TEXT,
            timeframe TEXT,
            side TEXT,
            entry DOUBLE PRECISION,
            stop_loss DOUBLE PRECISION,
            tp1 DOUBLE PRECISION,
            tp2 DOUBLE PRECISION,
            tp3 DOUBLE PRECISION,
            rr_final DOUBLE PRECISION,
            lots DOUBLE PRECISION,
            risk_pct DOUBLE PRECISION,
            balance DOUBLE PRECISION,
            expires_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS throttles (
            symbol TEXT PRIMARY KEY,
            last_sent TIMESTAMP
        );
        """)
        conn.commit()

def throttle_ok(symbol: str, min_minutes: int) -> bool:
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT last_sent FROM throttles WHERE symbol=%s;", (symbol,))
        row = cur.fetchone()
        now = datetime.now(tz=TZ)
        if not row:
            cur.execute("INSERT INTO throttles(symbol,last_sent) VALUES(%s,%s);", (symbol, now))
            conn.commit()
            return True
        last = row["last_sent"].replace(tzinfo=TZ) if row["last_sent"].tzinfo is None else row["last_sent"]
        if now - last >= timedelta(minutes=min_minutes):
            cur.execute("UPDATE throttles SET last_sent=%s WHERE symbol=%s;", (now, symbol))
            conn.commit()
            return True
        return False

def save_signal(rec: dict):
    with db() as conn, conn.cursor() as cur:
        fields = ",".join(rec.keys())
        ph     = ",".join(["%s"]*len(rec))
        cur.execute(f"INSERT INTO signals ({fields}) VALUES ({ph});", tuple(rec.values()))
        conn.commit()

# ========= SYMBOL UNIVERSE =========
# We’ll dynamically pick the current 30 most volatile from this broad universe
UNIVERSE = list(set([
    # FX Majors & Minors (Yahoo format)
    "EURUSD=X","GBPUSD=X","USDJPY=X","USDCHF=X","USDCAD=X","AUDUSD=X","NZDUSD=X",
    "EURGBP=X","EURJPY=X","GBPJPY=X","AUDJPY=X","EURAUD=X","CADJPY=X","CHFJPY=X",
    # Metals & Energy & Softs
    "GC=F","SI=F","HG=F","CL=F","NG=F","PL=F","PA=F","ZC=F","ZW=F","ZS=F","KC=F","SB=F","CC=F",
    # Indices
    "^GSPC","^NDX","^DJI","^RUT","^VIX","^N225","^FTSE","^GDAXI","^FCHI","^HSI",
    # Crypto (Yahoo tickers)
    "BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","ADA-USD","DOGE-USD",
]))

async def most_volatile_top30() -> list:
    # daily data, last 60 days; rank by ATR% or stdev of returns
    df = []
    for s in UNIVERSE:
        try:
            d = yf.download(s, period="3mo", interval="1d", progress=False, auto_adjust=True)
            if len(d) < 20: 
                continue
            atr = ta.atr(d["High"], d["Low"], d["Close"], length=14).iloc[-1]
            vol = float(atr / d["Close"].iloc[-1])
            df.append((s, vol))
        except Exception as e:
            log.warning(f"Vol rank fail {s}: {e}")
    df = sorted(df, key=lambda x: x[1], reverse=True)[:30]
    return [s for s,_ in df]

# ========= STRATEGY (Oliver Velez flavored moving-average confluence) =========
def velez_signal(df: pd.DataFrame):
    # SMA 2/5/20/75/200 + flat 200MA approx + 20 near 200
    close = df["Close"]
    sma2  = close.rolling(2).mean()
    sma5  = close.rolling(5).mean()
    sma20 = close.rolling(20).mean()
    sma75 = close.rolling(75).mean()
    sma200= close.rolling(200).mean()

    c = close.iloc[-1]
    m2,m5,m20,m75,m200 = sma2.iloc[-1],sma5.iloc[-1],sma20.iloc[-1],sma75.iloc[-1],sma200.iloc[-1]
    if any(map(lambda x: np.isnan(x), [m2,m5,m20,m75,m200])): 
        return None

    # "Relatively flat" 200 MA => slope small
    m200_prev = sma200.iloc[-5]
    flat200 = abs((m200 - m200_prev)/m200_prev) < 0.001  # <0.1% over 5 bars
    near_20_200 = abs(m20 - m200) < 0.002*c             # within ~0.2% of price

    if flat200 and near_20_200:
        # BUY stack
        if m2 > m5 > m20 > m200:
            side = "BUY"
            entry = float(c)
            sl    = entry * (1 - STOP_PCT/100)         # 0.05% default
            tp1   = float(m75)                          # pullback to 75MA
            tp2   = entry * (1 + 0.786/100 * 100/9) if tp1 <= entry else entry * 1.01
            tp3   = entry * (1 + (RR_FINAL * (STOP_PCT/100)))  # ~1:9
            return dict(side=side, entry=entry, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3)
        # SELL stack
        if m2 < m5 < m20 < m200:
            side = "SELL"
            entry = float(c)
            sl    = entry * (1 + STOP_PCT/100)
            tp1   = float(m75)
            tp2   = entry * (1 - 0.786/100 * 100/9) if tp1 >= entry else entry * 0.99
            tp3   = entry * (1 - (RR_FINAL * (STOP_PCT/100)))
            return dict(side=side, entry=entry, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3)
    return None

def lot_size(balance: float, risk_pct: float, entry: float, stop: float, pip_value=1.0):
    risk_amount = balance * (risk_pct/100.0)
    stop_dist   = abs(entry - stop)
    if stop_dist == 0:
        return 0.0
    lots = risk_amount / (stop_dist * pip_value)
    return max(0.0, round(lots, 4))

def can_open_until_blown(balance: float, risk_pct: float):
    # consecutive full-stop losses until zero (geometric decay)
    r = risk_pct/100.0
    if r<=0 or r>=1: return 0
    n=0
    b=balance
    while b>1:
        b *= (1-r)
        n+=1
        if n>1000: break
    return n

# ========= MESSAGE BUILDER =========
def format_signal(symbol, timeframe, audience, sig, balance, risk_pct):
    lots = lot_size(balance, risk_pct, sig["entry"], sig["sl"])
    blows = can_open_until_blown(balance, risk_pct)
    exp = datetime.now(tz=TZ) + {
        "2m": timedelta(minutes=15),
        "5m": timedelta(minutes=30),
        "15m": timedelta(minutes=45),
        "30m": timedelta(hours=2),
        "45m": timedelta(hours=6),
        "1h": timedelta(days=7),
        "4h": timedelta(days=30),
    }[timeframe]

    text = (
f"🎯 <b>Bullseye Monster</b> — <i>{audience}</i>\n"
f"🕒 TF: <b>{timeframe}</b> | ⏳ Expires: <code>{exp.strftime('%Y-%m-%d %H:%M')}</code> JHB\n"
f"📈 {symbol} | <b>{sig['side']}</b>\n"
f"Entry: <code>{sig['entry']:.4f}</code>\n"
f"SL:    <code>{sig['sl']:.4f}</code>  (Stop {STOP_PCT:.2f}%)\n"
f"TP1:   <code>{sig['tp1']:.4f}</code>\n"
f"TP2:   <code>{sig['tp2']:.4f}</code>\n"
f"TP3:   <code>{sig['tp3']:.4f}</code>  (≈ R:{RR_FINAL:.1f})\n"
f"💰 Bal: ${balance:,.2f} | Risk: {risk_pct:.2f}% → Lots: <b>{lots}</b>\n"
f"🧮 Worst-case consecutive losses until $0: <b>{blows}</b>\n"
    )
    # Save to DB
    save_signal(dict(
        symbol=symbol, audience=audience, timeframe=timeframe, side=sig["side"],
        entry=sig["entry"], stop_loss=sig["sl"], tp1=sig["tp1"], tp2=sig["tp2"], tp3=sig["tp3"],
        rr_final=RR_FINAL, lots=lots, risk_pct=risk_pct, balance=balance, expires_at=exp
    ))
    return text

# ========= SCANNER =========
async def scan_once(app, symbols, timeframe, audience, chat_id, balance, risk_pct, throttle_min):
    for sym in symbols:
        try:
            if not throttle_ok(sym, throttle_min):  # avoid spamming the same ticker too often
                continue
            # Pick interval mapping for Yahoo
            interval = {
                "2m":"2m","5m":"5m","15m":"15m","30m":"30m","45m":"30m","1h":"60m","4h":"60m"
            }[timeframe]
            period = "60d" if timeframe in ("1h","4h") else "10d"
            d = yf.download(sym, period=period, interval=interval, progress=False, auto_adjust=True)
            if len(d)<210: 
                continue
            sig = velez_signal(d)
            if sig:
                msg = format_signal(sym, timeframe, audience, sig, balance, risk_pct)
                await app.bot.send_message(chat_id=chat_id, text=msg, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
                await asyncio.sleep(0.4)
        except Exception as e:
            log.error(f"scan_once error {sym} {timeframe}: {e}")

# ========= SCHEDULER LOOPS =========
async def loop_main(app):
    # MAIN schedules:
    # 2m scan every 2m, 5m every 5m, 15m every 15m, 30m every 30m
    while True:
        top = await most_volatile_top30()
        await scan_once(app, top, "2m",  "MAIN", MAIN_CHAT, DEFAULT_BALANCE, DEFAULT_RISK_PCT, throttle_min=2)
        await scan_once(app, top, "5m",  "MAIN", MAIN_CHAT, DEFAULT_BALANCE, DEFAULT_RISK_PCT, throttle_min=5)
        await scan_once(app, top, "15m", "MAIN", MAIN_CHAT, DEFAULT_BALANCE, DEFAULT_RISK_PCT, throttle_min=15)
        await scan_once(app, top, "30m", "MAIN", MAIN_CHAT, DEFAULT_BALANCE, DEFAULT_RISK_PCT, throttle_min=30)
        await asyncio.sleep(60)  # small spacer; each scan uses its own throttle

async def loop_vip(app):
    # VIP schedules:
    # 45m (scan every 45m), 1h (hourly), 4h (every 4h)
    while True:
        top = await most_volatile_top30()
        await scan_once(app, top, "45m", "VIP", VIP_CHAT, DEFAULT_BALANCE, DEFAULT_RISK_PCT, throttle_min=45)
        await asyncio.sleep(60)  # spacer
        await scan_once(app, top, "1h",  "VIP", VIP_CHAT, DEFAULT_BALANCE, DEFAULT_RISK_PCT, throttle_min=60)
        await asyncio.sleep(60)
        await scan_once(app, top, "4h",  "VIP", VIP_CHAT, DEFAULT_BALANCE, DEFAULT_RISK_PCT, throttle_min=240)
        # Sleep until next 45-minute boundary
        await asyncio.sleep(60*30)

# ========= COMMANDS =========
async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 Bullseye Monster is online and fully automated.\n/use /scan SYMBOL to force a scan")

async def scan_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    symbol = (ctx.args[0] if ctx.args else "EURUSD=X").upper()
    try:
        d = yf.download(symbol, period="10d", interval="15m", progress=False, auto_adjust=True)
        if len(d)<210:
            await update.message.reply_text("Not enough data for signal.")
            return
        sig = velez_signal(d)
        if not sig:
            await update.message.reply_text("No valid confluence right now.")
            return
        msg = format_signal(symbol, "15m", "MANUAL", sig, DEFAULT_BALANCE, DEFAULT_RISK_PCT)
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

# ========= MAIN =========
async def main():
    init_db()
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("scan", scan_cmd))
    # background loops
    asyncio.create_task(loop_main(app))
    asyncio.create_task(loop_vip(app))
    log.info("Bullseye Monster started.")
    await app.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
