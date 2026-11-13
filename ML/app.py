"""
MedCare - Ung dung Du doan Nguy co Benh (Streamlit)
=====================================================
Tab 1: Dai thao duong (T2D)
Tab 2: Benh tim mach (CVD)

Moi tab gom:
  Tang 1: nhap lieu 1 lan -> du doan -> luu lich su
  Tang 2: chi hoat dong khi co >= 3 lich su Tang 1
          -> tu dong ghep 3 ban ghi gan nhat -> du doan tuong lai (R16)
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# Constants
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

T2D_FEATS = [
    "RAGENDER","AGEY_B","HEIGHT","WEIGHT","PMBMI","PMWAIST",
    "SMOKEV","SMOKEN","VGACTX","MDACTX","LTACTX",
    "DRINK","DRINKD","DRINKN",
    "SLEEP","SLEEPE","SLEEPFAL","SLEEPSN","SLEEPBR",
    "HIBP","HIBPS","HIBPE","ARTHR","ARTHRE","ARTHRS",
    "DRUGS","BPMED","HIGHCHOL","CHOLMED","MENOPAUSE","ITOT",
    "FE_WtHR","FE_BMI_Waist","FE_Weekly_Drinks","FE_Smoking_Status",
    "FE_Physical_Activity_Score","FE_Sleep_Disturbance_Index","FE_Log_Income",
]

CVD_FEATS = [
    "RAGENDER","AGEY_B","HEIGHT","WEIGHT","PMBMI","PMWAIST",
    "BPSYS","BPDIA","BPPULS",
    "SMOKEV","SMOKEN","VGACTX","MDACTX","LTACTX",
    "DRINK","DRINKD","DRINKN",
    "SLEEP","SLEEPE","SLEEPFAL","SLEEPSN","SLEEPBR",
    "HIBP","HIBPS","HIBPE","ARTHR","ARTHRE","ARTHRS",
    "DRUGS","BPMED","HIGHCHOL","IS_SOB","SOB","CHOLMED","MENOPAUSE","ITOT",
    "FE_WtHR","FE_BMI_Waist","FE_Weekly_Drinks","FE_Smoking_Status",
    "FE_Physical_Activity_Score","FE_Sleep_Disturbance_Index","FE_Log_Income",
    "FE_Pulse_Pressure","FE_MAP","FE_RPP",
]

# (label, min, max, step, default, help)
META = {
    "RAGENDER":     ("Gioi tinh (1=Nam 2=Nu)", 1, 2, 1, 2, ""),
    "AGEY_B":       ("Tuoi", 21, 110, 1, 66, ""),
    "HEIGHT":       ("Chieu cao (m)", 1.0, 2.5, 0.01, 1.68, ""),
    "WEIGHT":       ("Can nang (kg)", 20.0, 200.0, 0.5, 75.0, ""),
    "PMBMI":        ("BMI", 9.0, 80.0, 0.1, 28.0, ""),
    "PMWAIST":      ("Vong eo (cm)", 30, 200, 1, 90, ""),
    "BPSYS":        ("HA tam thu (mmHg)", 60, 250, 1, 120, ""),
    "BPDIA":        ("HA tam truong (mmHg)", 30, 150, 1, 80, ""),
    "BPPULS":       ("Nhip tim (bpm)", 30, 200, 1, 75, ""),
    "SMOKEV":       ("Tung hut thuoc (0/1)", 0, 1, 1, 0, ""),
    "SMOKEN":       ("Dang hut thuoc (0/1)", 0, 1, 1, 0, ""),
    "VGACTX":       ("HĐ manh (1-5)", 1, 5, 1, 4, "1=Thuong xuyen 5=Khong bao gio"),
    "MDACTX":       ("HĐ vua (1-5)", 1, 5, 1, 3, ""),
    "LTACTX":       ("HĐ nhe (1-5)", 1, 5, 1, 3, ""),
    "DRINK":        ("Co uong ruou (0/1)", 0, 1, 1, 0, ""),
    "DRINKD":       ("Ngay uong/tuan", 0, 7, 1, 0, ""),
    "DRINKN":       ("Ly/ngay uong", 0, 30, 1, 0, ""),
    "SLEEP":        ("Kho ngu (0/1)", 0, 1, 1, 0, ""),
    "SLEEPE":       ("Met sang som (0/1)", 0, 1, 1, 0, ""),
    "SLEEPFAL":     ("Buon ngu ngay (1-3)", 1, 3, 1, 2, ""),
    "SLEEPSN":      ("Ngay (0-3)", 0, 3, 1, 0, ""),
    "SLEEPBR":      ("Ngung tho ngu (0-3)", 0, 3, 1, 0, ""),
    "HIBP":         ("HA cao trang thai (0-6)", 0, 6, 1, 0, ""),
    "HIBPS":        ("DT HA cao hien tai (0/1)", 0, 1, 1, 0, ""),
    "HIBPE":        ("Tung DT HA cao (0/1)", 0, 1, 1, 0, ""),
    "ARTHR":        ("Viem khop trang thai (0-6)", 0, 6, 1, 0, ""),
    "ARTHRE":       ("Tung DT viem khop (0/1)", 0, 1, 1, 0, ""),
    "ARTHRS":       ("Dang DT viem khop (0/1)", 0, 1, 1, 0, ""),
    "DRUGS":        ("Dung thuoc trang thai (0-6)", 0, 6, 1, 0, ""),
    "BPMED":        ("Thuoc HA (1-6)", 1, 6, 1, 1, ""),
    "HIGHCHOL":     ("Cholesterol cao (0-6)", 0, 6, 1, 0, ""),
    "IS_SOB":       ("Kho tho (0/1)", 0, 1, 1, 0, ""),
    "SOB":          ("Kho tho leo cau thang (0/1)", 0, 1, 1, 0, ""),
    "CHOLMED":      ("Thuoc Cholesterol (1-5)", 1, 5, 1, 1, ""),
    "MENOPAUSE":    ("Man kinh (0/1)", 0, 1, 1, 0, ""),
    "ITOT":         ("Thu nhap (USD)", 0, 8000000, 1000, 50000, ""),
    "FE_WtHR":                   ("WtHR", 0.1, 1.5, 0.001, 0.24, "Tu dong tinh"),
    "FE_BMI_Waist":              ("BMI x Vong eo", 250.0, 6000.0, 1.0, 1200.0, "Tu dong tinh"),
    "FE_Weekly_Drinks":          ("Ly ruou/tuan", 0, 200, 1, 0, "Tu dong tinh"),
    "FE_Smoking_Status":         ("Trang thai hut thuoc", 0, 2, 1, 0, "Tu dong tinh"),
    "FE_Physical_Activity_Score":("Diem hoat dong", 6, 30, 1, 15, "Tu dong tinh"),
    "FE_Sleep_Disturbance_Index":("Chi so roi loan GN", 3, 10, 1, 5, "Tu dong tinh"),
    "FE_Log_Income":             ("Log Thu nhap", 0.0, 16.0, 0.01, 10.5, "Tu dong tinh"),
    "FE_Pulse_Pressure":         ("Ap luc mach", 0, 200, 1, 40, "Tu dong tinh"),
    "FE_MAP":                    ("HA trung binh MAP", 40, 200, 1, 93, "Tu dong tinh"),
    "FE_RPP":                    ("RPP = SBP x HR", 2000, 50000, 100, 9000, "Tu dong tinh"),
}

GROUPS = {
    "Nhan khau hoc":       ["RAGENDER","AGEY_B"],
    "The trang":           ["HEIGHT","WEIGHT","PMBMI","PMWAIST"],
    "Huyet ap / Tim":      ["BPSYS","BPDIA","BPPULS"],
    "Thuoc la":            ["SMOKEV","SMOKEN"],
    "Hoat dong the chat":  ["VGACTX","MDACTX","LTACTX"],
    "Ruou bia":            ["DRINK","DRINKD","DRINKN"],
    "Giac ngu":            ["SLEEP","SLEEPE","SLEEPFAL","SLEEPSN","SLEEPBR"],
    "Benh ly / Thuoc":     ["HIBP","HIBPS","HIBPE","ARTHR","ARTHRE","ARTHRS",
                            "DRUGS","BPMED","HIGHCHOL","IS_SOB","SOB","CHOLMED","MENOPAUSE"],
    "Kinh te":             ["ITOT"],
    # "Dac trung tong hop":  ["FE_WtHR","FE_BMI_Waist","FE_Weekly_Drinks",
    #                         "FE_Smoking_Status","FE_Physical_Activity_Score",
    #                         "FE_Sleep_Disturbance_Index","FE_Log_Income",
    #                         "FE_Pulse_Pressure","FE_MAP","FE_RPP"],
}

EXPANDED_GROUPS = {"Nhan khau hoc","The trang","Huyet ap / Tim"}


# ============================================================
# Core helpers
# ============================================================
@st.cache_resource
def load_all_models():
    m = {}
    for dis in ["t2d","cvd"]:
        for algo in ["xgboost","lightgbm","catboost"]:
            for sfx in ["","_ft"]:
                k = f"{dis}{sfx}_{algo}"
                try:
                    m[k] = joblib.load(f"{MODEL_DIR}\\{k}.pkl")
                except Exception:
                    m[k] = None
    return m


def fe_from_row(row: dict) -> dict:
    h    = float(row.get("HEIGHT", 1.68))
    wc   = float(row.get("PMWAIST", 90))
    bmi  = float(row.get("PMBMI", 28))
    dd   = float(row.get("DRINKD", 0))
    dn   = float(row.get("DRINKN", 0))
    dk   = float(row.get("DRINK", 0))
    sv   = float(row.get("SMOKEV", 0))
    sn   = float(row.get("SMOKEN", 0))
    vg   = float(row.get("VGACTX", 3))
    md   = float(row.get("MDACTX", 3))
    lt   = float(row.get("LTACTX", 3))
    sl   = float(row.get("SLEEP", 0))
    sfl  = float(row.get("SLEEPFAL", 1))
    ssn  = float(row.get("SLEEPSN", 0))
    sbr  = float(row.get("SLEEPBR", 0))
    it   = max(float(row.get("ITOT", 0)), 0.0)
    sys_ = float(row.get("BPSYS", 120))
    dia  = float(row.get("BPDIA", 80))
    pls  = float(row.get("BPPULS", 75))
    meno = float(row.get("MENOPAUSE", 0))
    age  = float(row.get("AGEY_B", 60))
    return {
        "FE_WtHR":                   wc / (h * 100) if h > 0 else 0.0,
        "FE_BMI_Waist":              bmi * wc,
        "FE_Weekly_Drinks":          dd * dn if dk else 0.0,
        "FE_Smoking_Status":         float(2 if sn else (1 if sv else 0)),
        "FE_Physical_Activity_Score":(6-vg)*3 + (6-md)*2 + (6-lt),
        "FE_Sleep_Disturbance_Index": sl + sfl + ssn + sbr,
        "FE_Log_Income":             np.log1p(it),
        "FE_Menopause_Age":          age if meno == 1 else 0.0,
        "FE_Pulse_Pressure":         sys_ - dia,
        "FE_MAP":                    dia + (sys_ - dia) / 3.0,
        "FE_RPP":                    sys_ * pls,
    }


def do_predict(model, X_df):
    prob  = float(model.predict_proba(X_df)[0, 1])
    label = int(prob >= 0.5)
    return label, prob


def build_wide(hist: list, disease: str) -> pd.DataFrame:
    recs  = hist[-3:]
    waves = ["R13","R14","R15"]
    if disease == "T2D":
        base = ["AGEY_B","HEIGHT","WEIGHT","PMBMI","PMWAIST",
                "SMOKEV","SMOKEN","VGACTX","MDACTX","LTACTX",
                "DRINK","DRINKD","DRINKN","SLEEP","SLEEPE",
                "SLEEPFAL","SLEEPSN","SLEEPBR","HIBP","HIBPS","HIBPE",
                "ARTHR","ARTHRE","ARTHRS","DRUGS","BPMED",
                "HIGHCHOL","CHOLMED","MENOPAUSE"]
        fe_c = ["FE_WtHR","FE_BMI_Waist","FE_Weekly_Drinks",
                "FE_Smoking_Status","FE_Physical_Activity_Score",
                "FE_Sleep_Disturbance_Index","FE_Log_Income","FE_Menopause_Age"]
    else:
        base = ["AGEY_B","HEIGHT","WEIGHT","PMBMI","PMWAIST",
                "BPSYS","BPDIA","BPPULS","SMOKEV","SMOKEN",
                "VGACTX","MDACTX","LTACTX","DRINK","DRINKD","DRINKN",
                "SLEEP","SLEEPE","SLEEPFAL","SLEEPSN","SLEEPBR",
                "HIBP","HIBPS","HIBPE","ARTHR","ARTHRE","ARTHRS",
                "DRUGS","BPMED","HIGHCHOL","IS_SOB","SOB","CHOLMED","MENOPAUSE"]
        fe_c = ["FE_WtHR","FE_BMI_Waist","FE_Weekly_Drinks",
                "FE_Smoking_Status","FE_Physical_Activity_Score",
                "FE_Sleep_Disturbance_Index","FE_Log_Income","FE_Menopause_Age",
                "FE_Pulse_Pressure","FE_MAP","FE_RPP"]
    wide = {"RAGENDER": float(recs[0]["inputs"].get("RAGENDER", 2.0))}
    for rec, wave in zip(recs, waves):
        inp  = rec["inputs"]
        wnum = wave[1:]
        for col in base:
            wide[f"{wave}{col}"] = float(inp.get(col, 0.0))
        wide[f"H{wnum}ITOT"] = float(inp.get("ITOT", 0.0))
        for fe in fe_c:
            wide[f"{wave}_{fe}"] = float(inp.get(fe, 0.0))
    return pd.DataFrame([wide])


# ============================================================
# UI helpers
# ============================================================
def input_section(feats: list, kp: str) -> dict:
    row = {}
    for gname, gcols in GROUPS.items():
        active = [f for f in gcols if f in feats]
        if not active:
            continue
        with st.expander(gname, expanded=(gname in EXPANDED_GROUPS)):
            cols = st.columns(2)
            for j, feat in enumerate(active):
                m = META.get(feat, (feat, 0.0, 100.0, 1.0, 0.0, ""))
                lbl, mn, mx, step, dflt, hlp = m
                wk = f"{kp}__{feat}"
                if isinstance(step, int) and isinstance(dflt, (int, float)) and step == int(step):
                    val = cols[j%2].number_input(
                        lbl, min_value=int(mn), max_value=int(mx),
                        value=int(dflt), step=int(step), help=hlp, key=wk)
                else:
                    val = cols[j%2].number_input(
                        lbl, min_value=float(mn), max_value=float(mx),
                        value=float(dflt), step=float(step),
                        format="%.4f", help=hlp, key=wk)
                row[feat] = val
    return row


def result_card(label: int, prob: float):
    pct = f"{prob*100:.1f}%"
    if label == 1:
        st.error(f"**NGUY CO CAO** — Xac suat: **{pct}**")
    else:
        st.success(f"**NGUY CO THAP** — Xac suat: **{pct}**")
    st.progress(min(prob, 1.0))
    c1, c2, c3 = st.columns(3)
    c1.metric("Du doan", "CO benh" if label==1 else "KHONG benh")
    c2.metric("Xac suat", pct)
    c3.metric("Nguong", "0.50")


def history_table(hist: list):
    if not hist:
        st.info("Chua co lich su du doan.")
        return
    rows = [{
        "Lan": i+1,
        "Ket qua": "CO" if h["label"]==1 else "KHONG",
        "Xac suat": f"{h['prob']*100:.1f}%",
        "Model": h["model"],
        "Tuoi": h["inputs"].get("AGEY_B",""),
        "BMI": round(h["inputs"].get("PMBMI", 0), 1),
        "Vong eo": h["inputs"].get("PMWAIST",""),
    } for i, h in enumerate(hist)]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ============================================================
# Disease tab renderer
# ============================================================
def disease_tab(disease: str, models: dict):
    dkey  = disease.lower()
    hkey  = f"hist_{dkey}"
    if hkey not in st.session_state:
        st.session_state[hkey] = []

    feats = T2D_FEATS if disease == "T2D" else CVD_FEATS
    dname = "Dai Thao Duong" if disease == "T2D" else "Benh Tim Mach"

    # ---------- Tier 1 ----------
    st.subheader(f"Tang 1 — Danh gia nguy co {dname} hien tai")
    st.caption("Du doan nguy co tai mot lan kham dua tren cac chi so do duoc.")

    with st.form(f"frm_{dkey}_t1"):
        row = input_section(feats, f"{dkey}_t1")
        # auto-compute FE and merge
        fe_vals = fe_from_row(row)
        for fk, fv in fe_vals.items():
            if fk in feats:
                row[fk] = fv

        model_sel = st.selectbox(
            "Chon mo hinh (Tang 1)",
            ["XGBoost", "LightGBM", "CatBoost"],
            key=f"sel_{dkey}_t1"
        )
        go = st.form_submit_button("Du doan Tang 1", use_container_width=True)

    if go:
        mkey = f"{dkey}_{model_sel.lower()}"
        mdl  = models.get(mkey)
        if mdl is None:
            st.error(f"Khong tim thay model: {mkey}.pkl")
        else:
            X = pd.DataFrame([{f: row.get(f, np.nan) for f in feats}])
            lbl, prb = do_predict(mdl, X)
            st.session_state[hkey].append({
                "label": lbl, "prob": prb,
                "model": model_sel, "inputs": dict(row),
            })
            st.divider()
            st.markdown("### Ket qua Tang 1")
            result_card(lbl, prb)

    # ---------- History ----------
    st.divider()
    hist = st.session_state[hkey]
    n    = len(hist)
    ca, cb = st.columns([5, 1])
    ca.markdown(f"#### Lich su Tang 1 ({n} lan)")
    if cb.button("Xoa lich su", key=f"clr_{dkey}"):
        st.session_state[hkey] = []
        st.rerun()
    history_table(hist)

    # ---------- Tier 2 ----------
    st.divider()
    st.subheader(f"Tang 2 — Du bao nguy co {dname} tuong lai (R16)")
    st.caption("Can it nhat 3 lan kham o Tang 1. Tang 2 su dung 3 lan gan nhat.")

    if n < 3:
        st.warning(f"Can them {3-n} lan kham nua de kich hoat Tang 2. ({n}/3)")
        st.progress(n / 3)
        return

    st.success(f"Da du {n} lich su — Tang 2 san sang.")
    model_sel2 = st.selectbox(
        "Chon mo hinh (Tang 2)",
        ["XGBoost", "LightGBM", "CatBoost"],
        key=f"sel_{dkey}_t2"
    )
    if st.button("Du doan Tang 2", key=f"btn_{dkey}_t2", use_container_width=True):
        mkey2 = f"{dkey}_ft_{model_sel2.lower()}"
        mdl2  = models.get(mkey2)
        if mdl2 is None:
            st.error(f"Khong tim thay model: {mkey2}.pkl")
        else:
            with st.spinner("Dang xay dung du lieu wide-format..."):
                X2 = build_wide(hist, disease)
            lbl2, prb2 = do_predict(mdl2, X2)
            st.markdown("### Ket qua Tang 2 (Tuong lai)")
            result_card(lbl2, prb2)
            with st.expander("Du lieu dau vao Tang 2 (wide-format)"):
                st.dataframe(
                    X2.T.rename(columns={0: "Gia tri"}),
                    use_container_width=True
                )


# ============================================================
# Main
# ============================================================
def main():
    st.set_page_config(
        page_title="MedCare — Du doan Nguy co Benh",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0f0f23 0%,#1a1a3e 50%,#0d1b2a 100%);min-height:100vh;}
[data-testid="stHeader"]{background:rgba(15,15,35,.96);border-bottom:1px solid rgba(99,102,241,.3);}
.app-title{text-align:center;font-size:2.4rem;font-weight:700;
  background:linear-gradient(135deg,#818cf8,#c084fc,#38bdf8);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  padding:1rem 0 .3rem;}
.app-sub{text-align:center;color:#94a3b8;font-size:.95rem;margin-bottom:1.5rem;}
div[data-testid="stForm"]{
  background:rgba(255,255,255,.04);border:1px solid rgba(99,102,241,.22);
  border-radius:16px;padding:1.5rem;backdrop-filter:blur(8px);}
details{background:rgba(255,255,255,.03)!important;
  border:1px solid rgba(148,163,184,.15)!important;border-radius:12px!important;margin-bottom:.5rem;}
.stButton>button{
  background:linear-gradient(135deg,#6366f1,#8b5cf6)!important;
  color:#fff!important;border:none!important;border-radius:10px!important;
  font-weight:600!important;transition:all .18s ease;padding:.55rem 1.2rem;}
.stButton>button:hover{transform:translateY(-2px);
  box-shadow:0 8px 24px rgba(99,102,241,.45)!important;}
[data-testid="stMetricValue"]{font-size:1.35rem!important;font-weight:700!important;}
h2,h3,h4,h5{color:#e2e8f0!important;}
p,label,li{color:#cbd5e1!important;}
.stCaption{color:#94a3b8!important;}
</style>
""", unsafe_allow_html=True)

    st.markdown('<div class="app-title">🏥 MedCare — AI Du Doan Nguy Co Benh Man Tinh</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="app-sub">He thong hoc may hai tang · XGBoost · LightGBM · CatBoost </div>',
        unsafe_allow_html=True)

    with st.spinner("Dang tai mo hinh..."):
        models = load_all_models()

    with st.expander("Trang thai mo hinh", expanded=False):
        status = [{"Model": k, "Trang thai": "OK" if v else "Khong tim thay"}
                  for k, v in models.items()]
        st.dataframe(pd.DataFrame(status), use_container_width=True, hide_index=True)

    st.divider()

    tab1, tab2 = st.tabs(["🩸 Dai Thao Duong (T2D)", "❤️ Benh Tim Mach (CVD)"])
    with tab1:
        disease_tab("T2D", models)
    with tab2:
        disease_tab("CVD", models)


if __name__ == "__main__":
    if not st.runtime.exists():
        import sys
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
        sys.exit(stcli.main())
    else:
        main()
