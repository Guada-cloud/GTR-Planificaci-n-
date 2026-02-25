# utils_ops.py — núcleo Plan vs Real con mapeo flexible y comparación robusta
from __future__ import annotations
from typing import Optional, Tuple, List, Dict
import pandas as pd
import numpy as np
from io import BytesIO
from pathlib import Path
import re
from pandas.api.types import is_numeric_dtype

# ==========================
# Sinónimos/abreviaturas (español)
# ==========================
ALIAS_PLAN = {
    "Fecha": ["fecha","date","dia","día"],
    "Hora":  ["hora","hr","time","tiempo","h"],
    "Base":  ["base","sede","plataforma","plataf.","pto","site"],
    "CAT":   ["cat","centro","categoria","categoría","centro atencion","centro de atencion"],
    "Servicios_Planificados": [
        "svc plan","serv plan","servicios plan","servicios proyectados","svc proy",
        "proyectado","planificado","llamadas planificadas","llam plan","calls plan"
    ],
    "Moviles_Planificados": [
        "mov plan","moviles plan","mov req","mov requeridos","dotacion plan","dotación plan",
        "agentes plan","staff plan","operadores plan","plantilla plan","dot plan"
    ],
    "Llamadas_Planificadas": ["llamadas planificadas","llam plan","calls plan"]
}
ALIAS_REAL = {
    "Fecha": ["fecha","date","dia","día"],
    "Hora":  ["hora","hr","time","tiempo","h"],
    "Base":  ["base","sede","plataforma","plataf.","pto","site"],
    "CAT":   ["cat","centro","categoria","categoría","centro atencion","centro de atencion"],
    "Servicios_Reales": [
        "svc real","serv real","servicios reales","observado","reales","llamadas reales","calls"
    ],
    "Moviles_Reales": [
        "mov reales","mov real","mov x nomina","moviles x nomina","dotacion","dotación",
        "agentes","operadores","staff","plantilla","dot eff","dot efectiva"
    ],
    "Servicios_Derivados": ["derivaciones","derivados","transbordos","svc derivados","servicios derivados"],
    "Llamadas_Reales":     ["llamadas reales","llam real","calls reales"]
}

# ==========================
# Normalización y mapeo
# ==========================
def _norm_key(s: str) -> str:
    s = str(s).strip().lower()
    s = (s.replace("á","a").replace("é","e").replace("í","i")
           .replace("ó","o").replace("ú","u").replace("ñ","n"))
    s = re.sub(r"[^a-z0-9 ]+"," ", s)
    s = re.sub(r"\s+"," ", s)
    return s

def _token_set(s: str) -> set:
    return set(_norm_key(s).split())

def _score_alias(col_name: str, aliases: List[str]) -> float:
    c = _norm_key(col_name)
    if c in {_norm_key(a) for a in aliases}:  # exacto
        return 1.0
    for a in aliases:                          # substring
        if _norm_key(a) in c or c in _norm_key(a):
            return 0.8
    ts = _token_set(c)
    best = 0.0
    for a in aliases:
        ta = _token_set(a)
        if not ts or not ta:
            continue
        j = len(ts & ta)/len(ts | ta)         # Jaccard
        best = max(best, j)
    return best

def guess_mapping(df: pd.DataFrame, spec: Dict[str,List[str]]) -> Dict[str, Optional[str]]:
    mapping: Dict[str, Optional[str]] = {k:"" for k in spec.keys()}
    cols = list(df.columns)

    # Si primera fila parece encabezado “colado”
    if df.shape[0] and df.iloc[0].astype(str).str.len().mean() > 3 and df.iloc[0].nunique() == df.shape[1]:
        df.columns = df.iloc[0].astype(str).values
        df.drop(df.index[0], inplace=True)
        cols = list(df.columns)

    for canon, aliases in spec.items():
        best_col, best_score = None, 0.0
        for c in cols:
            sc = _score_alias(str(c), aliases+[canon])
            if sc > best_score:
                best_col, best_score = c, sc
        mapping[canon] = best_col if best_score >= 0.55 else ""

    # Detección de Hora si no apareció
    if not mapping.get("Hora"):
        for c in cols:
            series = df[c].astype(str).str.strip()
            hhmm = series.str.match(r"^\d{1,2}:\d{2}$", na=False).mean()
            if hhmm > 0.4:
                mapping["Hora"] = c; break
    return mapping

def apply_mapping(df: pd.DataFrame, mapping: Dict[str,str], kind: str) -> pd.DataFrame:
    if kind == "plan":
        want = ["Fecha","Hora","Base","CAT","Moviles_Planificados","Servicios_Planificados","Llamadas_Planificadas"]
    else:
        want = ["Fecha","Hora","Base","CAT","Moviles_Reales","Servicios_Reales","Servicios_Derivados","Llamadas_Reales"]
    out = df.rename(columns={v:k for k,v in mapping.items() if v}).copy()
    for k in want:
        if k not in out.columns: out[k] = np.nan
    return out[want].copy()

# ==========================
# Limpieza numérica y tiempo
# ==========================
def coerce_number(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan,
                   "#¿NOMBRE?": np.nan, "#¡NOMBRE?": np.nan, "#VALUE!": np.nan}, regex=False)
    def _fix_one(x: str):
        if x is np.nan or x is None: return np.nan
        txt = str(x)
        if "," in txt and "." in txt:
            if txt.rfind(",") > txt.rfind("."): txt = txt.replace(".","").replace(",",".")
            else: txt = txt.replace(",","")
        elif "," in txt: txt = txt.replace(",",".")
        try: return float(txt)
        except: return np.nan
    return s.map(_fix_one)

def enrich_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Fecha"] = pd.to_datetime(out["Fecha"], errors="coerce").dt.date
    if "Hora" in out:
        if is_numeric_dtype(out["Hora"]):
            frac = (pd.to_numeric(out["Hora"], errors="coerce") % 1)
            td = pd.to_timedelta(frac, unit="D")
            out["Hora"] = (pd.Timestamp("1900-01-01") + td).dt.time
        else:
            out["Hora"] = pd.to_datetime(out["Hora"].astype(str), errors="coerce").dt.time
    out["HoraStr"] = pd.to_datetime(out["Hora"].astype(str), errors="coerce").dt.strftime("%H:%M")
    out["Fecha_dt"] = pd.to_datetime(out["Fecha"])
    iso = out["Fecha_dt"].dt.isocalendar()
    out["Año"] = out["Fecha_dt"].dt.year
    out["Mes"] = out["Fecha_dt"].dt.month
    out["Semana"] = iso.week
    out["Dia"] = out["Fecha_dt"].dt.day

    def _band(hhmm: str)->str:
        try: h = int(hhmm[:2])
        except: return "NA"
        if   0<=h<6:  return "MADRUGADA"
        if   6<=h<13: return "MAÑANA"
        if  13<=h<20: return "TARDE"
        return "NOCHE"
    out["Franja"] = out["HoraStr"].map(_band)
    return out

# ==========================
# Merge y KPIs
# ==========================
def merge_plan_real(plan: pd.DataFrame, real: pd.DataFrame) -> pd.DataFrame:
    keys = ["Fecha","Hora","Base"]
    if "CAT" in plan.columns or "CAT" in real.columns:
        for df in (plan, real):
            if "CAT" not in df: df["CAT"] = np.nan
        keys.append("CAT")
    merged = pd.merge(plan, real, on=keys, how="outer", indicator=True, suffixes=("_Plan","_Real"))
    merged["Status"] = np.select(
        [merged["_merge"].eq("left_only"), merged["_merge"].eq("right_only"), merged["_merge"].eq("both")],
        ["No ejecutado","No planificado","OK"], default="Desconocido"
    )
    merged.drop(columns=["_merge"], inplace=True)
    return merged

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["Servicios_Planificados","Servicios_Reales","Moviles_Planificados","Moviles_Reales",
              "Llamadas_Planificadas","Llamadas_Reales","Servicios_Derivados"]:
        if c not in out: out[c] = np.nan
        out[c] = coerce_number(out[c])

    out["Dif_Servicios"] = (out["Servicios_Reales"] - out["Servicios_Planificados"]).astype(float)
    out["Dif_Moviles"]   = (out["Moviles_Reales"]    - out["Moviles_Planificados"]).astype(float)

    out["Desvio_Servicios_%"] = np.where(out["Servicios_Planificados"]>0, out["Dif_Servicios"]/out["Servicios_Planificados"]*100, np.nan)
    out["Desvio_Moviles_%"]   = np.where(out["Moviles_Planificados"]>0,   out["Dif_Moviles"]/out["Moviles_Planificados"]*100,   np.nan)

    out["Efectividad"] = np.where(out["Servicios_Planificados"]>0, 1 - (out["Dif_Servicios"].abs()/out["Servicios_Planificados"]), np.nan)
    out["AE"]  = (out["Servicios_Reales"] - out["Servicios_Planificados"]).abs()
    out["APE"] = np.where(out["Servicios_Planificados"]>0, out["AE"]/out["Servicios_Planificados"], np.nan)
    out["Bias"]= (out["Servicios_Planificados"] - out["Servicios_Reales"])

    out["Clasificacion"] = np.select(
        [
            out.get("Status","").astype(str).eq("No ejecutado"),
            out.get("Status","").astype(str).eq("No planificado"),
            out["Dif_Servicios"].fillna(0).eq(0),
            out["Dif_Servicios"].fillna(0)>0,
            out["Dif_Servicios"].fillna(0)<0
        ],
        ["No ejecutado","No planificado","Exacto","Sobre planificado","Bajo planificado"],
        default="NA"
    )
    return enrich_time(out)

def agg_error_metrics(df: pd.DataFrame) -> dict:
    d = df.copy()
    mape = d["APE"].mean()*100 if "APE" in d and d["APE"].notna().any() else np.nan
    mae  = d["AE"].mean() if "AE" in d and d["AE"].notna().any() else np.nan
    fbias= (d["Bias"].sum()/d["Servicios_Reales"].sum()*100) if "Servicios_Reales" in d and d["Servicios_Reales"].sum()!=0 else np.nan
    return {"MAPE_%": mape, "MAE": mae, "ForecastBias_%": fbias}

# ==========================
# Agregados
# ==========================
def aggregate_nacional(df: pd.DataFrame)->pd.DataFrame:
    if df.empty: return pd.DataFrame()
    g = df.groupby("HoraStr", as_index=False)[["Servicios_Planificados","Servicios_Reales","Moviles_Planificados","Moviles_Reales","Dif_Servicios","Dif_Moviles"]].sum()
    return g.sort_values("HoraStr")

def aggregate_bases(df: pd.DataFrame)->pd.DataFrame:
    if df.empty: return pd.DataFrame()
    g = df.groupby("Base", as_index=False)[["Servicios_Planificados","Servicios_Reales"]].sum()
    g["abs_diff"] = (g["Servicios_Reales"]-g["Servicios_Planificados"]).abs()
    g["Desvio_%"] = np.where(g["Servicios_Planificados"]>0, (g["Servicios_Reales"]-g["Servicios_Planificados"])/g["Servicios_Planificados"]*100, np.nan)
    return g.sort_values("abs_diff", ascending=False)

def aggregate_franja(df: pd.DataFrame)->pd.DataFrame:
    if df.empty: return pd.DataFrame()
    return df.groupby(["Franja","HoraStr"], as_index=False)[["Servicios_Planificados","Servicios_Reales","Dif_Servicios","Dif_Moviles"]].sum().sort_values(["Franja","HoraStr"])

def aggregate_cat(df: pd.DataFrame)->pd.DataFrame:
    if df.empty or "CAT" not in df.columns: return pd.DataFrame()
    return df.groupby("CAT", as_index=False)[["Servicios_Planificados","Servicios_Reales","Dif_Servicios","Dif_Moviles"]].sum().sort_values("Dif_Servicios", ascending=False)

# ==========================
# Forecast simple (perfil horario + EWMA)
# ==========================
def _hidx(hhmm: str)->int:
    try: return int(hhmm[:2])
    except: return -1

def forecast_next_hours(df: pd.DataFrame, target_col="Servicios_Reales", horizon: int=6, today: Optional[pd.Timestamp]=None)->pd.DataFrame:
    if df.empty or target_col not in df: return pd.DataFrame()
    d = df.copy(); d["Fecha_dt"]=pd.to_datetime(d["Fecha"])
    if today is None: today = d["Fecha_dt"].max()
    cur = pd.to_datetime(today).date()
    d_today = d[d["Fecha"].eq(cur)]
    if d_today.empty: d_today = d[d["Fecha"].eq(d["Fecha"].max())]
    obs = d_today.groupby("HoraStr", as_index=False)[target_col].sum()
    obs["h"] = obs["HoraStr"].map(_hidx)
    obs = obs[(obs["h"]>=0)&(obs["h"]<=23)].drop_duplicates("h").set_index("h").sort_index()

    hist = d[d["Fecha"]<cur]
    if not hist.empty:
        hprof = hist.assign(h=hist["HoraStr"].map(_hidx)).groupby("h")[target_col].median()
    else:
        hprof = pd.Series(dtype=float)

    if hprof.empty and "Servicios_Planificados" in d_today:
        tmp = d_today.groupby("HoraStr", as_index=False)["Servicios_Planificados"].sum()
        tmp["h"]=tmp["HoraStr"].map(_hidx)
        hprof = tmp.set_index("h")["Servicios_Planificados"]

    if hprof.empty:
        base = float(obs[target_col].median() if not obs.empty else 0.0)
        hprof = pd.Series({h:base for h in range(24)})

    ewma = None
    if not obs.empty:
        vals = obs[target_col].astype(float)
        ewma = vals.ewm(alpha=0.5).mean().iloc[-1]

    last_h = int(obs.index.max()) if len(obs)>0 else -1
    fut = [h for h in range(last_h+1, min(24, last_h+1+horizon))]
    preds = []
    for h in fut:
        base = float(hprof.get(h, hprof.median()))
        pred = 0.6*base + 0.4*float(ewma) if ewma is not None else base
        preds.append({"HoraStr": f"{h:02d}:00", "Prediccion": max(pred,0.0), "Metodo": "Perfil+EWMA" if ewma is not None else "Perfil"})
    return pd.DataFrame(preds)

# ==========================
# Export Excel
# ==========================
def to_excel_bytes(book: Dict[str,pd.DataFrame], fname="reporte_plan_vs_real.xlsx")->tuple[bytes,str]:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as wr:
        for sheet, df in book.items():
            if df is None or df.empty: continue
            df.to_excel(wr, index=False, sheet_name=sheet[:31])
    return buf.getvalue(), fname

# ==========================
# Persistencia (opcional)
# ==========================
DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
MERG_CSV = DATA_DIR/"merged.csv"
def save_csv(df: pd.DataFrame, path: Path): df.to_csv(path, index=False, encoding="utf-8")
def load_csv(path: Path) -> Optional[pd.DataFrame]: return pd.read_csv(path, encoding="utf-8") if path.exists() else None
