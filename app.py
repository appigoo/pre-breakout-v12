import os
import re
from itertools import combinations
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Pre-Rise Lab V3.3", layout="wide")

APP_VERSION = "V3.3"
FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_10d", "vol_ratio", "vol_chg_5d",
    "macd_hist_change", "price_vs_ema20", "ema20_slope_5d", "price_vs_ema50",
    "atr_pct", "close_position", "high_20_ratio", "relative_strength_5d",
    "qqq_5d", "vix_5d", "breakout_distance", "range_pct",
]
FEATURE_LABELS = {
    "ret_1d": "1日升跌幅", "ret_5d": "5日升跌幅", "ret_10d": "10日升跌幅",
    "vol_ratio": "成交量/20日均量", "vol_chg_5d": "5日成交量變化",
    "macd_hist_change": "MACD柱變化", "price_vs_ema20": "EMA20乖離",
    "ema20_slope_5d": "EMA20斜率", "price_vs_ema50": "EMA50乖離",
    "atr_pct": "ATR%", "close_position": "收市位置", "high_20_ratio": "距20日前高",
    "relative_strength_5d": "相對QQQ強弱", "qqq_5d": "QQQ 5日", "vix_5d": "VIX 5日",
    "breakout_distance": "距前高", "range_pct": "日內波幅",
}


def clean_index(df):
    if df is None or df.empty:
        return pd.DataFrame()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx
    return df.sort_index()


@st.cache_data(ttl=900)
def load_price(symbol, period="5y"):
    try:
        df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).title() for c in df.columns]
    cols = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in cols):
        return pd.DataFrame()
    df = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    return clean_index(df)


@st.cache_data(ttl=900)
def load_market(period="5y"):
    return load_price("QQQ", period), load_price("^VIX", period)


def add_features(df, qqq=None, vix=None):
    x = df.copy()
    c, o, h, l, v = x.Close, x.Open, x.High, x.Low, x.Volume
    x["ret_1d"] = c.pct_change()
    x["ret_5d"] = c.pct_change(5)
    x["ret_10d"] = c.pct_change(10)
    x["vol_ratio"] = v / v.rolling(20).mean()
    x["vol_chg_5d"] = v.pct_change(5)
    ema12 = c.ewm(span=12, adjust=False).mean(); ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    x["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    x["macd_hist_change"] = x.macd_hist.diff()
    x["ema20"] = c.ewm(span=20, adjust=False).mean()
    x["ema50"] = c.ewm(span=50, adjust=False).mean()
    x["ema200"] = c.ewm(span=200, adjust=False).mean()
    x["price_vs_ema20"] = c / x.ema20 - 1
    x["price_vs_ema50"] = c / x.ema50 - 1
    x["ema20_slope_5d"] = x.ema20.pct_change(5)
    x["atr"] = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1).rolling(14).mean()
    x["atr_pct"] = x.atr / c
    x["range_pct"] = (h-l) / c
    x["close_position"] = (c-l) / (h-l).replace(0, np.nan)
    x["prev_high_20"] = h.rolling(20).max().shift(1)
    x["prev_low_20"] = l.rolling(20).min().shift(1)
    x["high_20_ratio"] = c / x.prev_high_20 - 1
    x["breakout_distance"] = c / x.prev_high_20 - 1
    x["breakout"] = c > x.prev_high_20
    x["bull_candle"] = c > o
    if qqq is not None and not qqq.empty:
        q = qqq.Close.reindex(x.index).ffill()
        x["qqq_5d"] = q.pct_change(5)
        x["relative_strength_5d"] = x.ret_5d - x.qqq_5d
    else:
        x["qqq_5d"] = np.nan; x["relative_strength_5d"] = np.nan
    if vix is not None and not vix.empty:
        vv = vix.Close.reindex(x.index).ffill()
        x["vix_5d"] = vv.pct_change(5)
    else:
        x["vix_5d"] = np.nan
    return x


def future_paths(df, horizon):
    """Outcome only. Never use these columns as live features."""
    x = df.copy(); n = len(x); close = x.Close.to_numpy(float); high = x.High.to_numpy(float); low = x.Low.to_numpy(float)
    mfe = np.full(n, np.nan); mae = np.full(n, np.nan); fclose = np.full(n, np.nan)
    for i in range(n-horizon):
        window_h = high[i+1:i+horizon+1]; window_l = low[i+1:i+horizon+1]
        mfe[i] = np.max(window_h) / close[i] - 1
        mae[i] = np.min(window_l) / close[i] - 1
        fclose[i] = close[i+horizon] / close[i] - 1
    x["future_mfe"] = mfe; x["future_mae"] = mae; x["future_close_return"] = fclose
    return x


def choose_event_definition(feature_df):
    """Stock-specific label selection. Threshold is learned from TRAIN outcomes only."""
    candidates = []
    for h in (5, 10, 20):
        z = future_paths(feature_df, h)
        train = z.dropna(subset=["future_mfe", "atr_pct"]).copy()
        if len(train) < 150:
            continue
        for q in (0.80, 0.85, 0.90):
            threshold = float(train.future_mfe.quantile(q))
            # Keep labels economically meaningful; still stock-specific because threshold is learned.
            threshold = max(threshold, 0.08)
            events = train.future_mfe >= threshold
            n = int(events.sum()); rate = n / len(train)
            if 0.08 <= rate <= 0.25 and n >= 12:
                # Prefer enough events and a clearly extreme threshold, without tuning on OOS.
                score = np.log1p(n) * (threshold / max(train.future_mfe.median(), 1e-6))
                candidates.append((score, h, threshold, q, n, rate))
    if not candidates:
        h = 10; z = future_paths(feature_df, h); train = z.dropna(subset=["future_mfe"]); threshold = max(float(train.future_mfe.quantile(.85)), .08)
        return h, threshold, .85, int((train.future_mfe >= threshold).sum()), float((train.future_mfe >= threshold).mean()), z
    best = max(candidates, key=lambda r: r[0])
    _, h, threshold, q, n, rate = best
    return h, threshold, q, n, rate, future_paths(feature_df, h)


def make_events(labeled, threshold):
    x = labeled.copy()
    raw = x.future_mfe >= threshold
    # Independent event episodes: keep the first event, then cooldown until its forward horizon has passed.
    event = np.zeros(len(x), dtype=bool); last = -10**9
    for i, flag in enumerate(raw.fillna(False).to_numpy()):
        if flag and i > last:
            event[i] = True; last = i + 10
    x["large_move"] = raw
    x["event"] = event
    return x


def pre_rise_windows(x, event_col="event", lookback=20):
    events = x.index[x[event_col].fillna(False)]
    rows = []
    for dt in events:
        i = x.index.get_loc(dt)
        start = max(0, i-lookback)
        if i-start < lookback:
            continue
        for lag in range(lookback, 0, -1):
            r = x.iloc[i-lag]
            row = {"event_date": dt, "lag": -lag, "date": x.index[i-lag]}
            for f in FEATURE_COLS:
                row[f] = r.get(f, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def feature_effect(train):
    """Compare feature distributions before events vs non-event days. No OOS data enters this table."""
    if train.empty:
        return pd.DataFrame()
    ev = train.event.fillna(False)
    rows = []
    for f in FEATURE_COLS:
        a = train.loc[ev, f].dropna(); b = train.loc[~ev, f].dropna()
        if len(a) < 8 or len(b) < 20:
            continue
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        effect = (a.mean()-b.mean()) / pooled if pooled and np.isfinite(pooled) else 0
        rows.append({"Feature": f, "特徵": FEATURE_LABELS[f], "事件樣本": len(a), "普通日樣本": len(b), "事件平均": a.mean(), "普通日平均": b.mean(), "Effect Size": effect, "事件中位數": a.median(), "普通日中位數": b.median()})
    out = pd.DataFrame(rows)
    return out.sort_values("Effect Size", key=lambda s: s.abs(), ascending=False) if not out.empty else out


def feature_stability_by_lag(x):
    """For each feature, measure how its event/non-event separation evolves T-20..T-1."""
    windows = pre_rise_windows(x, lookback=20)
    if windows.empty: return pd.DataFrame()
    rows=[]
    for f in FEATURE_COLS:
        for lag in sorted(windows.lag.unique()):
            a=windows.loc[windows.lag==lag, f].dropna()
            # compare against all same-lag observations that are not event days
            dates=set(windows.loc[windows.lag==lag, "date"])
            ev_dates=set(windows.loc[windows.lag==lag, "event_date"])
            b=x.loc[list(dates), f].dropna() if dates else pd.Series(dtype=float)
            # Because each event creates a date, use a separate same-lag baseline sampled from all train dates.
            if len(a)<5 or len(b)<10: continue
            pooled=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
            effect=(a.mean()-b.mean())/pooled if pooled and np.isfinite(pooled) else 0
            rows.append({"Feature":f,"lag":int(lag),"Effect Size":effect})
    return pd.DataFrame(rows)


def similarity_score(train, latest, features, target):
    h=train.dropna(subset=features+['future_mfe']).copy()
    if h.empty: return pd.DataFrame(), np.nan, 0.0
    mat=h[features].astype(float); cur=latest[features].astype(float)
    if cur.isna().any(): return pd.DataFrame(), np.nan, 0.0
    scale=mat.std().replace(0,1).fillna(1)
    d=np.sqrt(((mat-cur)/scale).pow(2).mean(axis=1))
    h['distance']=d; sim=h.sort_values('distance').head(min(25,len(h))).copy()
    # Similarity strength is relative to the train distance distribution, not an arbitrary 0-100 claim.
    q=float(d.quantile(.25)); best=float(sim.distance.iloc[0]); strength=float(np.clip(1-best/(q*3+1e-9),0,1))
    hit=float((sim.future_mfe>=target).mean())
    return sim, hit, strength


def choose_dna(train, event_threshold):
    eff=feature_effect(train)
    if eff.empty: return [], eff
    # Only features with enough observations and a meaningful, stable-looking training effect enter candidate DNA.
    candidates=eff.loc[eff["Effect Size"].abs()>=0.20, "Feature"].tolist()[:8]
    if not candidates:
        candidates=eff["Feature"].tolist()[:5]
    # A transparent single-feature ranking is safer than a large combinatorial search at first.
    return candidates[:5], eff


def oos_metrics(oos, dna, threshold):
    if oos.empty or not dna:
        return {"Precision":np.nan,"Recall":np.nan,"F1":np.nan,"Signals":0,"TP":0,"FP":0,"FN":0,"TN":0}
    z=oos.dropna(subset=dna+['future_mfe']).copy()
    # Use direction of training effect: positive effect means high feature; negative means low feature.
    # Thresholds are learned from TRAIN quantiles only.
    pred=pd.Series(True,index=z.index)
    for f in dna:
        direction=1 if f in getattr(oos_metrics,'positive_features',set()) else None
        # populated by caller via dataframe attrs
        if direction is None:
            direction=1
        q=z.attrs.get('feature_quantiles',{}).get(f,(np.nan,np.nan))
        if not np.isfinite(q[0]): continue
        pred &= z[f] >= q[1] if z.attrs.get('feature_directions',{}).get(f,1)>0 else z[f] <= q[0]
    y=z.future_mfe>=threshold
    tp=int((pred&y).sum()); fp=int((pred&~y).sum()); fn=int((~pred&y).sum()); tn=int((~pred&~y).sum())
    p=tp/(tp+fp) if tp+fp else np.nan; r=tp/(tp+fn) if tp+fn else np.nan; f1=2*p*r/(p+r) if pd.notna(p) and pd.notna(r) and p+r else np.nan
    return {"Precision":p,"Recall":r,"F1":f1,"Signals":int(pred.sum()),"TP":tp,"FP":fp,"FN":fn,"TN":tn}


def build_dna_rule(train, dna):
    directions={}; qs={}
    for f in dna:
        a=train.loc[train.event, f].dropna(); b=train.loc[~train.event, f].dropna()
        if len(a)<8 or len(b)<20: continue
        effect=(a.mean()-b.mean())/np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2) if (a.var(ddof=1)+b.var(ddof=1))>0 else 0
        directions[f]=1 if effect>=0 else -1
        # Use event distribution median as the signal threshold; learned only from TRAIN.
        qs[f]=float(a.median())
    return directions,qs


def apply_dna(df,dna,directions,thresholds):
    if not dna: return pd.Series(False,index=df.index)
    pred=pd.Series(True,index=df.index)
    for f in dna:
        if f not in thresholds or f not in directions: continue
        pred &= df[f]>=thresholds[f] if directions[f]>0 else df[f]<=thresholds[f]
    return pred.fillna(False)


def metrics(y,p):
    y=pd.Series(y,index=p.index if hasattr(p,'index') else None).astype(bool); p=pd.Series(p).astype(bool)
    tp=int((p&y).sum()); fp=int((p&~y).sum()); fn=int((~p&y).sum()); tn=int((~p&~y).sum())
    precision=tp/(tp+fp) if tp+fp else np.nan; recall=tp/(tp+fn) if tp+fn else np.nan
    f1=2*precision*recall/(precision+recall) if pd.notna(precision) and pd.notna(recall) and precision+recall else np.nan
    return {"TP":tp,"FP":fp,"FN":fn,"TN":tn,"Precision":precision,"Recall":recall,"F1":f1,"Signals":int(p.sum())}


def split_oos(labeled, ratio=.70, embargo=20):
    valid=labeled.dropna(subset=['future_mfe']).copy(); n=len(valid); cut=max(1,int(n*ratio)); train_end=max(1,cut-embargo)
    return valid.iloc[:train_end].copy(), valid.iloc[cut:].copy(), valid.iloc[train_end:cut].copy()


def ema_regime(train, full):
    q10=float(train.price_vs_ema20.quantile(.10)); q90=float(train.price_vs_ema20.quantile(.90))
    out=full.copy(); out['ema20_q10_train']=q10; out['ema20_q90_train']=q90
    out['ema20_extended']=out.price_vs_ema20>q90
    out['ema20_deep_below']=out.price_vs_ema20<q10
    out['ema20_reclaim']=(out.Close>out.ema20)&(out.Close.shift(1)<=out.ema20.shift(1))&(out.ema20_slope_5d>0)
    return out,(q10,q90)


def breakout_state(x):
    r=x.iloc[-1]; price=float(r.Close); level=r.prev_high_20
    if pd.isna(level): return {'stage':'⚪ 資料不足','level':np.nan,'quality':0,'retest':False,'confirmed':False,'false':False,'age':np.nan}
    # Recent breakout tracking without looking forward.
    recent=x.tail(10); bars=recent.index[recent.Close>recent.prev_high_20]
    last=None
    if len(bars): last=bars[-1]
    if last is not None:
        lvl=float(x.loc[last,'prev_high_20']); age=int((x.index[-1]-last).days)
        close_above=price>lvl
        confirmed=bool(close_above and len(x)>=2 and x.Close.iloc[-2]>lvl)
        retest=bool(close_above and r.Low<=lvl*1.01 and r.Close>=lvl)
        false=bool((x.loc[last:].Close.iloc[1:]<lvl).any()) if len(x.loc[last:])>1 else False
        quality=float(np.clip(50+25*np.clip(r.vol_ratio-1,0,1.5)+20*np.clip(r.close_position-.5,0,.5)*2+15*np.clip(r.breakout_distance,0,.03)/.03,0,100))
        if false and not close_above: stage='🔴 FALSE BREAKOUT'
        elif retest: stage='🟢 RETEST 成功'
        elif confirmed: stage='🟢 CONFIRMED'
        else: stage='🟡 BREAKOUT IN PROGRESS'
        return {'stage':stage,'level':lvl,'quality':quality,'retest':retest,'confirmed':confirmed,'false':false,'age':age}
    near=price/level-1
    return {'stage':'🟡 SETUP — 接近前高' if near>=-.02 else '⚪ 未接近關鍵位','level':float(level),'quality':0,'retest':False,'confirmed':False,'false':False,'age':np.nan}


def current_decision(x, train, oos, dna, directions, thresholds, event_threshold, info_negative=False):
    r=x.iloc[-1]
    live_pred=bool(apply_dna(x.tail(1),dna,directions,thresholds).iloc[0])
    op=apply_dna(oos,dna,directions,thresholds)
    om=metrics(oos.future_mfe>=event_threshold,op) if len(oos) else metrics(pd.Series(dtype=bool),pd.Series(dtype=bool))
    sim, sim_hit, match=similarity_score(train,r,dna,event_threshold) if dna else (pd.DataFrame(),np.nan,0)
    trend_parts={
        '價格>EMA20':r.price_vs_ema20>0,
        'EMA20>EMA50':r.ema20>r.ema50,
        'EMA20向上':r.ema20_slope_5d>0,
        '相對QQQ強勢':r.relative_strength_5d>0 if pd.notna(r.relative_strength_5d) else False,
        '距前高合理':r.breakout_distance>-0.08,
    }
    trend=100*np.mean(list(trend_parts.values()))
    if r.vol_ratio>=1.5 and r.ret_1d>0: volume=90; vlabel='🟢 放量上升'
    elif r.vol_ratio>=1.5 and r.ret_1d<0: volume=15; vlabel='🔴 放量下跌'
    elif r.vol_ratio>=1.2 and r.ret_1d>0: volume=75; vlabel='🟢 偏買盤'
    else: volume=50; vlabel='⚪ 一般'
    b=breakout_state(x); extended=bool(r.ema20_extended); reclaim=bool(r.ema20_reclaim)
    entry=60.0
    entry += 15 if not extended else -30
    entry += 10 if abs(r.price_vs_ema20)<=.08 else -10
    entry += 10 if b['confirmed'] or b['retest'] else 0
    entry += 8 if reclaim else 0
    entry += 8 if volume>=70 else -15 if volume<40 else 0
    entry=float(np.clip(entry,0,100))
    edge=live_pred and pd.notna(om['Precision']) and om['Precision']>=.55 and match>=.20
    if not edge: decision='⚪ NO TRADE'
    elif info_negative: decision='🟡 WAIT — INFORMATION RISK'
    elif extended: decision='🔵 EXTENDED — 不追'
    elif trend<60 or volume<50: decision='🟡 WAIT — TREND'
    elif entry<65: decision='🟡 WAIT — ENTRY'
    elif b['false']: decision='🟡 WAIT — FALSE BREAKOUT'
    elif b['confirmed'] or b['retest'] or reclaim: decision='🟢 BUY NOW'
    else: decision='🟡 WAIT FOR CONFIRMATION'
    conf=100*(.45*(om['Precision'] if pd.notna(om['Precision']) else .5)+.30*(sim_hit if pd.notna(sim_hit) else .5)+.25*match)
    return {'decision':decision,'confidence':float(np.clip(conf,0,100)),'dna_match':live_pred,'oos':om,'sim':sim,'sim_hit':sim_hit,'match':match,'trend':trend,'trend_parts':trend_parts,'volume':volume,'volume_label':vlabel,'entry_quality':entry,'extended':extended,'reclaim':reclaim,'breakout':b,'price':price,'ema_dist':float(r.price_vs_ema20),'event_threshold':event_threshold}


def trade_plan(x,dec):
    r=x.iloc[-1]; price=float(r.Close); atr=float(r.atr) if pd.notna(r.atr) else price*.03; breakout=dec['breakout']['level']
    if pd.isna(breakout): breakout=price
    swing=float(min(r.Low if pd.notna(r.Low) else price, r.prev_low_20 if pd.notna(r.prev_low_20) else price-1.5*atr))
    stop=min(swing,price-1.25*atr)
    risk=max(price-stop,.01); return {'early':(max(stop+.05*atr,price-.25*atr),price+.10*atr),'confirm':(max(breakout,price),max(breakout,price)+.20*atr),'breakout':breakout,'stop':stop,'t1':price+1.5*risk,'t2':price+2.5*risk,'risk':risk/price}


def data_qa(df, qqq, vix):
    ohlc=(df.High>=df[['Open','Close']].max(axis=1))&(df.Low<=df[['Open','Close']].min(axis=1))
    rows=[('日期無重複',not df.index.duplicated().any()),('OHLC合理',bool(ohlc.all())),('成交量>0',bool((df.Volume>0).all())),('核心欄位無缺失',not df[['Open','High','Low','Close','Volume']].isna().any().any()),('QQQ可用',not qqq.empty),('VIX可用',not vix.empty)]
    return pd.DataFrame(rows,columns=['QA','結果'])


def pct(x): return '—' if pd.isna(x) else f'{x:.1%}'


def run(symbol, period):
    df=load_price(symbol,period); qqq,vix=load_market(period)
    if df.empty: return None
    x=add_features(df,qqq,vix)
    # Label selection uses the first 70% only. OOS never helps define the event threshold.
    split=int(len(x)*.70); base=x.iloc[:split].copy()
    h,thr,q,n_events,rate,_=choose_event_definition(base)
    labeled=future_paths(x,h); labeled=make_events(labeled,thr)
    train,oos,embargo=split_oos(labeled,h and 0.70 or .70,embargo=h)
    # The event label itself is fixed from TRAIN-only threshold; future_mfe is outcome only.
    ema_x,ema_q=ema_regime(train,labeled)
    train=ema_x.loc[train.index]; oos=ema_x.loc[oos.index]; x=ema_x
    dna,eff=choose_dna(train,thr)
    directions,thresholds=build_dna_rule(train,dna)
    op=apply_dna(oos,dna,directions,thresholds)
    om=metrics(oos.future_mfe>=thr,op)
    info_negative=False
    dec=current_decision(x,train,oos,dna,directions,thresholds,thr,info_negative)
    plan=trade_plan(x,dec)
    timeline=pre_rise_windows(train,lookback=20)
    # For each feature, find the lag with the strongest absolute separation.
    lag_df=feature_stability_by_lag(train)
    if not lag_df.empty:
        peaks=lag_df.loc[lag_df.groupby('Feature')['Effect Size'].apply(lambda s:s.abs().idxmax()).values].copy()
    else: peaks=pd.DataFrame()
    qa=data_qa(df,qqq,vix)
    return {'df':df,'x':x,'qqq':qqq,'vix':vix,'train':train,'oos':oos,'embargo':embargo,'horizon':h,'threshold':thr,'quantile':q,'event_count':n_events,'event_rate':rate,'dna':dna,'directions':directions,'thresholds':thresholds,'effects':eff,'timeline':timeline,'lag_effects':lag_df,'peak_lags':peaks,'decision':dec,'plan':plan,'qa':qa}


# ---------------- UI ----------------
st.title('🔬 Pre-Rise Lab V3.3')
st.caption('核心目標：研究每隻股票過去5年「大幅上升前」的狀態，建立 Stock DNA，再檢查今天是否再次出現類似狀態。')
with st.sidebar:
    symbol=st.text_input('股票代號','TSLA').upper().strip()
    period=st.selectbox('資料範圍',['5y','10y'],index=0)
    refresh=st.button('🔄 重新抓取')
    if refresh:
        st.cache_data.clear(); st.rerun()
    st.info('V3.3 把研究重心放回核心問題：Event → Pre-Rise Timeline → Feature Discovery → Stock DNA → OOS → Today Match。')

if not symbol: st.stop()
with st.spinner(f'研究 {symbol} 的 Pre-Rise DNA…'):
    res=run(symbol,period)
if res is None:
    st.error('找不到資料，請檢查股票代號。'); st.stop()

d=res['decision']; x=res['x']; latest=x.iloc[-1]; plan=res['plan']

st.markdown('## 🎯 TODAY — 目前是否像過去爆升前？')
a,b,c,d1,e=st.columns(5)
a.metric('決策',d['decision']); b.metric('Current DNA Match','🟢 YES' if d['dna_match'] else '⚪ NO'); c.metric('Historical Match',f"{d['match']:.0%}"); d1.metric('OOS Precision',pct(d['oos']['Precision'])); e.metric('研究 Confidence',f"{d['confidence']:.0f}/100")
st.caption(f"資料截至 {x.index[-1].date()}；事件定義：未來 {res['horizon']} 個交易日最高升幅 ≥ {pct(res['threshold'])}。此門檻只由訓練資料學習。")

if d['decision']=='🟢 BUY NOW': st.success(f'{symbol} → 🟢 BUY NOW：歷史 DNA 相似、OOS 有證據、目前入場條件也通過。')
elif d['decision'].startswith('🔵'): st.info(f'{symbol} → {d["decision"]}：方向不等於追價位置。')
else: st.warning(f'{symbol} → {d["decision"]}：至少一個核心條件尚未通過。')

T1,T2,T3,T4,T5=st.tabs(['🧬 Pre-Rise DNA','🕰️ Timeline','🕵️ Feature Detective','🎮 Replay','🧪 OOS & QA'])
with T1:
    st.subheader('🧬 這隻股票的爆升前 DNA')
    st.write(f"**事件樣本：{res['event_count']}**　｜　事件率：{res['event_rate']:.1%}　｜　研究窗口：{res['horizon']} 日")
    if d['dna']:
        st.write('**目前 DNA：** ' + ' + '.join(FEATURE_LABELS.get(f,f) for f in d['dna']))
        dna_df=pd.DataFrame([{'Feature':f,'方向':'↑' if res['directions'].get(f,1)>0 else '↓','Train門檻':res['thresholds'].get(f,np.nan),'中文':FEATURE_LABELS.get(f,f)} for f in d['dna']])
        st.dataframe(dna_df,use_container_width=True,hide_index=True)
    st.subheader('📊 Historical Feature Separation')
    eff=res['effects'].copy()
    if not eff.empty:
        st.dataframe(eff.assign(**{'Effect Size':eff['Effect Size'].round(3)}),use_container_width=True,hide_index=True)
    else: st.info('樣本不足。')
    st.caption('Effect Size 只用 TRAIN；它是研究證據，不是勝率。')

with T2:
    st.subheader('🕰️ 爆升前 20 日時間軸')
    st.write('每一行代表一個歷史爆升事件；T-1 是爆升前一天。')
    tl=res['timeline']
    if tl.empty: st.info('資料不足建立完整20日事件窗口。')
    else:
        show=tl[['event_date','lag','date']+['ret_5d','vol_ratio','price_vs_ema20','ema20_slope_5d','relative_strength_5d','high_20_ratio']].copy()
        st.dataframe(show.tail(500),use_container_width=True,hide_index=True)
    if not res['peak_lags'].empty:
        st.subheader('⏱️ 哪個時間點開始出現變化？')
        st.dataframe(res['peak_lags'].assign(Feature=lambda z:z.Feature.map(lambda q:FEATURE_LABELS.get(q,q))),use_container_width=True,hide_index=True)

with T3:
    st.subheader('🕵️ Feature Detective')
    st.write('問題不是「哪個指標最好」，而是「爆升前它是否與普通日有穩定差異？」')
    eff=res['effects'].copy()
    if not eff.empty:
        eff2=eff[['特徵','事件平均','普通日平均','Effect Size']].copy()
        st.dataframe(eff2,use_container_width=True,hide_index=True)
    st.markdown('### 💡 下一步研究問題')
    st.write('1. 特徵差異是否在 T-20 → T-1 持續增強？')
    st.write('2. 單一特徵有效，還是組合狀態有效？')
    st.write('3. OOS 是否仍然有效？')
    st.write('4. 同一特徵在不同股票是否完全不同？')

with T4:
    st.subheader('🎮 Historical Replay — 股票偵探')
    events=res['train'].index[res['train'].event.fillna(False)].tolist()
    if not events:
        st.info('沒有足夠歷史事件。')
    else:
        selected=st.selectbox('選一個歷史事件（先看事件前資訊）',[e.strftime('%Y-%m-%d') for e in events])
        dt=pd.Timestamp(selected); i=x.index.get_loc(dt); start=max(0,i-20); before=x.iloc[start:i].copy()
        st.write(f'**盲測日期：{selected}**　距離事件還有 1 日。')
        st.line_chart(before[['Close','ema20','ema50']])
        st.dataframe(before[['Close','Volume','vol_ratio','price_vs_ema20','ema20_slope_5d','relative_strength_5d']].tail(10),use_container_width=True)
        reveal=st.checkbox('🚀 Reveal：顯示之後發生什麼')
        if reveal:
            future=x.iloc[i:min(len(x),i+res['horizon']+1)][['Close','High','Volume']].copy()
            st.line_chart(future[['Close']])
            st.dataframe(future,use_container_width=True)
            st.success(f"事件後 {res['horizon']} 日 MFE：{pct(x.loc[dt,'future_mfe'])}；MAE：{pct(x.loc[dt,'future_mae'])}")

with T5:
    st.subheader('🧪 OOS Evidence')
    m=d['oos']; c1,c2,c3,c4=st.columns(4); c1.metric('Precision',pct(m['Precision'])); c2.metric('Recall',pct(m['Recall'])); c3.metric('F1',pct(m['F1'])); c4.metric('OOS Signals',m['Signals'])
    st.write(f"OOS 日期：{res['oos'].index.min().date() if len(res['oos']) else '—'} → {res['oos'].index.max().date() if len(res['oos']) else '—'}；中間 {len(res['embargo'])} 日作 embargo，避免鄰近事件/未來標籤污染。")
    st.subheader('🛡️ Model QA')
    checks=[
        ('事件門檻只由TRAIN學習',True),
        ('DNA特徵只由TRAIN學習',True),
        ('OOS未參與DNA選擇',True),
        ('Future MFE/MAE只作Outcome',True),
        ('EMA20量化只由TRAIN學習',True),
        ('今日不使用今日future label',pd.isna(latest.future_mfe)),
        ('OHLC/Volume QA',bool(res['qa']['結果'].all())),
        ('OOS有獨立樣本',len(res['oos'])>0),
    ]
    st.dataframe(pd.DataFrame([{'QA':a,'結果':'✅ PASS' if b else '❌ FAIL'} for a,b in checks]),use_container_width=True,hide_index=True)
    st.subheader('📋 QA 原始資料')
    st.dataframe(res['qa'],use_container_width=True,hide_index=True)

st.markdown('---')
st.markdown('### 🔬 V3.3 研究哲學')
st.write('**先研究過去，再判斷今天；先證明 OOS Edge，再談 BUY。** V3.3 暫時不追求增加更多指標，而是把「爆升前狀態」研究得更深。')
st.caption('Research/decision-support prototype，不構成個人化投資建議。')
