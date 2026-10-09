"""Streamlit interface — Bayesian uncertainty decomposition for early concept-drift warning, validated by XAI.

Run locally:   streamlit run streamlit_app.py
Deploy:        push the repo to GitHub -> share.streamlit.io -> New app -> main file: streamlit_app.py
"""
from __future__ import annotations

import copy
import json
import os
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

st.set_page_config(page_title="Avertizare timpurie a devierii · Bayesian Drift Early Warning",
                   page_icon=":material/monitoring:", layout="wide")

from bdx import data as D  # noqa: E402
from bdx import drift as DR  # noqa: E402
from bdx import ui as UI  # noqa: E402
from bdx.models import MODEL_LABELS  # noqa: E402
from bdx.uncertainty import SIGNAL_LABELS  # noqa: E402

st.html(UI.CSS)

# ---------------------------------------------------------------- language (Romanian by default)
LANG = st.sidebar.segmented_control("Limba / Language", ["ro", "en"], default="ro", key="lang",
                                    format_func={"ro": "Română", "en": "English"}.get) or "ro"
RO = LANG == "ro"


def T(en: str, ro: str) -> str:
    """Return the text in the selected interface language."""
    return ro if RO else en


def N(n) -> str:
    """Thousands separator: 2 860 in Romanian, 2,860 in English."""
    return f"{int(n):,}".replace(",", "\u202f") if RO else f"{int(n):,}"


MODEL_LABELS_RO = {"mcd": "MC-Dropout MLP (aproximare bayesiană)", "ens": "Ansamblu de rețele (aproximare bayesiană)",
                   "bsvm": "SVM RBF bootstrap (frecventist)", "rf": "Pădure aleatoare (frecventist)"}
SIGNAL_LABELS_RO = {"conf": "1 - max prob (referință MSP)", "tu": "Entropia totală TU", "au": "Aleatorică AU",
                    "eu": "Epistemică EU (informația mutuală)", "lr": "Raportul log(EU/AU)  [propus]"}
MLAB = MODEL_LABELS_RO if RO else MODEL_LABELS
SLAB = SIGNAL_LABELS_RO if RO else SIGNAL_LABELS
SNAME = UI.SIG_NAME_RO if RO else UI.SIG_NAME


@st.cache_data(show_spinner=False)
def base_cfg():
    return yaml.safe_load(open(os.path.join(ROOT, "config", "default.yaml")))


@st.cache_resource(show_spinner=False)
def cached_tab(key):
    return D.load_dataset(key)


@st.cache_resource(show_spinner=False, max_entries=8)
def run_cached(dataset, drift, model, seed, cfg_json, upload_bytes=None, target=None, drop_json="[]"):
    from bdx.experiment import run_one
    cfg = json.loads(cfg_json)
    tab = None
    if dataset == "upload":
        import io
        tab = D.load_csv(io.BytesIO(upload_bytes), target, drop=json.loads(drop_json), name="upload")
        if not tab.numeric_idx:
            tab.numeric_idx = list(range(tab.X.shape[1]))
    elif dataset in D.SPECS:
        tab = cached_tab(dataset)
    return run_one(dataset if dataset != "upload" else "upload", drift, model, seed, cfg, tab=tab, keep_traces=True,
                   stream=DR.build_stream(tab, drift, seed, cfg) if dataset == "upload" else None)


# ============================================================================ sidebar
cfg0 = base_cfg()
st.sidebar.header(T("Set up an experiment", "Configurează un experiment"))
st.sidebar.html('<div class="bdx-side-note">' + T(
    "Pick data, a kind of drift and a model, then run. "
    "The drift starts at record 2&nbsp;500 and is complete by record 4&nbsp;000.",
    "Alege datele, tipul de deviere și modelul, apoi rulează. "
    "Devierea începe la înregistrarea 2&nbsp;500 și este completă la înregistrarea 4&nbsp;000.") + '</div>')
ds_names = {"credit": T("Credit default (finance, UCI)", "Neplata creditului (finanțe, UCI)"),
            "telco": T("Telco churn (customers, IBM)", "Pierderea clienților telecom (IBM)"),
            "digits": T("Digits 8×8 rotation (MNIST-like)", "Cifre 8×8 cu rotație (tip MNIST)"),
            "sea": T("SEA synthetic (real drift)", "SEA sintetic (deviere reală)"),
            "elec2": T("Elec2 electricity (natural drift)", "Elec2 electricitate (deviere naturală)"),
            "upload": T("Upload your own CSV", "Încarcă propriul fișier CSV")}
dataset = st.sidebar.selectbox(T("Dataset", "Setul de date"), list(ds_names), format_func=ds_names.get)
upload_bytes, target, drop_cols, upload_ok = None, None, [], True
if dataset == "upload":
    up = st.sidebar.file_uploader(T("CSV file (one row per record, one label column)",
                                    "Fișier CSV (un rând pe înregistrare, o coloană-etichetă)"), type=["csv"])
    if up is not None:
        upload_bytes = up.getvalue()
        import io
        df_up = pd.read_csv(io.BytesIO(upload_bytes))
        cols = list(df_up.columns)
        target = st.sidebar.selectbox(T("Label column (what the model predicts)",
                                        "Coloana-etichetă (ce prezice modelul)"), cols, index=len(cols) - 1)
        # columns that would only add noise: identifiers and free text (almost one value per row)
        guess = [c for c in cols if c != target and (c.lower() in ("id", "index", "unnamed: 0") or c.lower().endswith("id")
                 or (not pd.api.types.is_numeric_dtype(df_up[c]) and df_up[c].nunique() > 0.5 * len(df_up)))]
        drop_cols = st.sidebar.multiselect(T("Columns to ignore (IDs, dates, free text)",
                                             "Coloane ignorate (ID-uri, date calendaristice, text liber)"),
                                           [c for c in cols if c != target], default=guess)
        n_cls = df_up[target].nunique()
        st.sidebar.caption(T(f"{N(len(df_up))} rows · {len(cols) - 1 - len(drop_cols)} input columns · "
                             f"{n_cls} label classes",
                             f"{N(len(df_up))} rânduri · {len(cols) - 1 - len(drop_cols)} coloane de intrare · "
                             f"{n_cls} clase"))
        if n_cls < 2 or n_cls > 20:
            upload_ok = False
            st.sidebar.error(T("The label column must have 2–20 classes (classification). "
                               "For a numeric target, bin it into classes first.",
                               "Coloana-etichetă trebuie să aibă 2–20 de clase (clasificare). "
                               "O țintă numerică trebuie mai întâi împărțită în clase."))
        elif len(df_up) < 1000:
            st.sidebar.warning(T("Fewer than 1 000 rows: the run works, but results will be noisy.",
                                 "Mai puțin de 1 000 de rânduri: rularea funcționează, dar rezultatele vor fi zgomotoase."))

allowed = {"digits": ["covariate", "none"], "sea": ["real", "none"], "elec2": ["natural"]}.get(
    dataset, ["covariate", "sensor", "real", "none"])
drift_help = {"covariate": T("Covariate: new kinds of records arrive", "Covariabile: apar înregistrări de tip nou"),
              "sensor": T("Sensor: two columns shift (a broken pipeline)", "Senzor: două coloane se deplasează (flux defect)"),
              "real": T("Real: same inputs, labels change meaning", "Reală: aceleași intrări, etichetele își schimbă sensul"),
              "none": T("None: nothing changes (false-alarm check)", "Niciuna: nimic nu se schimbă (test de alarme false)"),
              "natural": T("Natural: real time-ordered stream", "Naturală: flux real, ordonat în timp")}
drift = st.sidebar.selectbox(T("Drift type", "Tipul devierii"), allowed, format_func=drift_help.get)
model = st.sidebar.selectbox(T("Probabilistic model", "Modelul probabilistic"), ["mcd", "ens", "bsvm", "rf"],
                             format_func=MLAB.get)
seed = st.sidebar.number_input(T("Seed", "Sămânța (seed)"), 0, 999, 0)

with st.sidebar.expander(T("Advanced settings", "Setări avansate")):
    cfg = copy.deepcopy(cfg0)
    cfg["stream_len"] = st.slider(T("Stream length", "Lungimea fluxului"), 2000, 10000, cfg0["stream_len"], 500)
    cfg["t0"] = st.slider(T("Drift onset t0", "Începutul devierii t0"), 500, cfg["stream_len"] - 500, min(cfg0["t0"], cfg["stream_len"] - 500), 100)
    cfg["width"] = st.slider(T("Ramp width (0 = abrupt)", "Lățimea rampei (0 = bruscă)"), 0, 3000, cfg0["width"], 100)
    cfg["sensor_delta"] = st.slider(T("Sensor shift δ (σ units)", "Deplasarea senzorului δ (în σ)"), 0.5, 5.0, cfg0["sensor_delta"], 0.5)
    cfg["flip_rho"] = st.slider(T("Real drift: flip probability ρ", "Deviere reală: probabilitatea inversării ρ"), 0.1, 1.0, cfg0["flip_rho"], 0.1)
    cfg["model"]["T"] = st.slider(T("MC-Dropout samples T", "Eșantioane MC-Dropout T"), 5, 100, cfg0["model"]["T"], 5)
    cfg["model"]["dropout"] = st.slider(T("Dropout rate p", "Rata dropout p"), 0.05, 0.5, cfg0["model"]["dropout"], 0.05)
    cfg["detectors"]["target_far"] = st.slider(T("Target false-alarm probability", "Probabilitatea-țintă de alarmă falsă"), 0.01, 0.3,
                                               cfg0["detectors"]["target_far"], 0.01)
    cfg["detectors"]["label_delay"] = st.slider(T("Label delay for supervised detectors",
                                                  "Întârzierea etichetelor pentru detectoarele supervizate"), 0, 2000, 0, 50)
run_btn = st.sidebar.button(T("Run experiment", "Rulează experimentul"), type="primary", width="stretch")

# ============================================================================ header
st.html(UI.hero_html(RES, LANG))

tabs = st.tabs([T("The idea", "Ideea"), T("Run an experiment", "Rulează un experiment"),
                T("Explain the alarm", "Explică alarma"), T("Research results", "Rezultatele cercetării"),
                T("Theory", "Teorie"), T("Reproduce", "Reproducere")])

# ============================================================================ overview
with tabs[0]:
    c1, c2 = st.columns([1, 1.05], gap="large")
    with c1:
        st.subheader(T("Why it matters", "De ce contează"))
        st.html(T("""<div class="bdx-prose">
<p>A model trained on last year's clients keeps predicting when a new kind of client appears. Its accuracy
slowly falls, but nobody can measure that until the true outcomes arrive, often months later. Classic drift
detectors watch the error rate, so they always wait for those labels.</p>
<p>This project asks the model itself. Many slightly different versions of the model vote on every record.
When they all hesitate, the case is simply hard: <b>aleatoric</b> uncertainty (AU). When they disagree with
each other, the model has not seen anything like it: <b>epistemic</b> uncertainty (EU). A rising
<b>log(EU/AU)</b> means the world has moved away from the training data.</p></div>""",
                  """<div class="bdx-prose">
<p>Un model antrenat pe clienții de anul trecut continuă să facă predicții și atunci când apare un tip nou de
client. Acuratețea lui scade treptat, dar nimeni nu o poate măsura până nu sosesc rezultatele reale, adesea după
luni de zile. Detectoarele clasice de deviere urmăresc rata de eroare, deci așteaptă mereu aceste etichete.</p>
<p>Proiectul întreabă chiar modelul. Mai multe versiuni ușor diferite ale modelului votează pentru fiecare
înregistrare. Când toate ezită, cazul este pur și simplu dificil: incertitudine <b>aleatorică</b> (AU). Când nu
sunt de acord între ele, modelul nu a mai văzut ceva asemănător: incertitudine <b>epistemică</b> (EU). Creșterea
raportului <b>log(EU/AU)</b> înseamnă că lumea s-a îndepărtat de datele de antrenare.</p></div>"""))
    with c2:
        st.subheader(T("What was tested, and what came out", "Ce s-a testat și ce a ieșit"))
        st.html(UI.hypotheses_html(LANG))
    st.subheader(T("How one experiment runs", "Cum se desfășoară un experiment"))
    st.graphviz_chart(UI.pipeline_dot(LANG), width="stretch")
    st.html('<p class="bdx-note">' + T(
        'Start in the sidebar: choose a dataset (or upload a CSV), a drift and a model, '
        'then open <b>Run an experiment</b>. After that, <b>Explain the alarm</b> shows which columns made the '
        'model unsure.',
        'Începe din bara laterală: alege un set de date (sau încarcă un CSV), o deviere și un model, apoi deschide '
        '<b>Rulează un experiment</b>. După aceea, <b>Explică alarma</b> arată ce coloane au făcut modelul '
        'nesigur.') + '</p>')

# ============================================================================ live experiment
with tabs[1]:
    if dataset == "upload" and (upload_bytes is None or not upload_ok):
        st.warning(T("Upload a CSV in the sidebar first.", "Încarcă mai întâi un fișier CSV în bara laterală.")
                   if upload_bytes is None else
                   T("Choose a label column with 2–20 classes in the sidebar.",
                     "Alege în bara laterală o coloană-etichetă cu 2–20 de clase."))
    elif run_btn or st.session_state.get("last_key"):
        key = (dataset, drift, model, int(seed), json.dumps(cfg, sort_keys=True), target, json.dumps(sorted(drop_cols)))
        if run_btn:
            st.session_state["last_key"] = key
        key = st.session_state["last_key"]
        with st.spinner(T("Training the model and running all detectors …",
                          "Se antrenează modelul și rulează toate detectoarele …")):
            out = run_cached(key[0], key[1], key[2], key[3], key[4], upload_bytes if key[0] == "upload" else None,
                             key[5], key[6])
        st.session_state["last_out"] = out
        tr = out["traces"]
        stt = tr["stream"]
        rows = pd.DataFrame(out["rows"])
        t0, t_acc = tr["t0"], tr["t_acc"]
        L = len(tr["err"])
        roll = lambda x, w=200: pd.Series(x).rolling(w, min_periods=20).mean()

        fa = rows.set_index("detector")
        if t0 is not None:
            post = stt.p >= 1
            acc_post = 1 - tr["err"][post].mean() if post.any() else None
            first_lr = fa.loc["PH[lr]", "first_alarm"]
            first_err = fa.loc[["DDM[err]", "ADWIN[err]", "PH[err]"], "first_alarm"].min()
            if pd.notna(first_lr) and t_acc is not None:
                d_ = t_acc - int(first_lr)
                lead = T(f"{abs(d_)} records {'before' if d_ >= 0 else 'after'} the visible drop",
                         f"cu {abs(d_)} înregistrări {'înainte de' if d_ >= 0 else 'după'} scăderea vizibilă")
            else:
                lead = (T("no visible drop to compare with", "nu există o scădere vizibilă pentru comparație")
                        if pd.notna(first_lr) else T("never fired", "nu s-a declanșat"))
            st.html(UI.readout_html([
                (T("Accuracy before the drift", "Acuratețea înainte de deviere"), f"{tr['acc_ref']:.3f}",
                 T("measured on calibration data", "măsurată pe datele de calibrare"), ""),
                (T("Accuracy after the drift", "Acuratețea după deviere"),
                 f"{acc_post:.3f}" if acc_post is not None else "—",
                 (T(f"{acc_post - tr['acc_ref']:+.3f} compared with before", f"{acc_post - tr['acc_ref']:+.3f} față de înainte")
                  if acc_post is not None else ""), "ink"),
                (T("Accuracy drop visible at", "Scăderea acurateței vizibilă la"),
                 f"t = {t_acc}" if t_acc is not None else T("never", "niciodată"),
                 T("needs the true labels", "necesită etichetele reale"), "drop" if t_acc is not None else ""),
                (T("Our alarm, log(EU/AU)", "Alarma noastră, log(EU/AU)"),
                 f"t = {int(first_lr)}" if pd.notna(first_lr) else T("none", "niciuna"), lead, "coral"),
            ]))
            if pd.notna(first_err) and pd.notna(first_lr):
                gap = int(first_err) - int(first_lr)
                st.html('<p class="bdx-note">' + T(
                    f'The first error-based alarm (which needs labels) came at t = {int(first_err)}, '
                    f'{abs(gap)} records {"after" if gap >= 0 else "before"} ours.',
                    f'Prima alarmă bazată pe eroare (care necesită etichete) a venit la t = {int(first_err)}, '
                    f'cu {abs(gap)} înregistrări {"după" if gap >= 0 else "înaintea"} alarmei noastre.') + '</p>')
        else:
            st.html(UI.readout_html([
                (T("Accuracy before", "Acuratețea înainte"), f"{tr['acc_ref']:.3f}",
                 T("calibration data", "datele de calibrare"), ""),
                (T("Accuracy on the stream", "Acuratețea pe flux"), f"{1 - tr['err'].mean():.3f}", "", ""),
                (T("Stream length", "Lungimea fluxului"), N(L), T("records", "înregistrări"), ""),
                (T("Alarms of our detector", "Alarmele detectorului nostru"), str(len(tr["alarms"]["PH[lr]"])),
                 T("every alarm here is false", "aici orice alarmă este falsă") if stt.drift_type == "none"
                 else T("no known drift time", "momentul devierii nu este cunoscut"), "coral"),
            ]))

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            row_heights=[0.24, 0.24, 0.22, 0.30],
                            subplot_titles=(T("Accuracy over the last 300 records (needs labels)",
                                              "Acuratețea pe ultimele 300 de înregistrări (necesită etichete)"),
                                            T("Aleatoric vs epistemic uncertainty", "Incertitudinea aleatorică vs epistemică"),
                                            T("log(EU/AU), the proposed signal", "log(EU/AU), semnalul propus"),
                                            T("Alarms of each detector", "Alarmele fiecărui detector")))
        fig.add_trace(go.Scatter(y=roll(1 - tr["err"], 300), name=T("accuracy", "acuratețe"), line=dict(color=UI.INK, width=2)), 1, 1)
        if tr["acc_thr"] is not None:
            fig.add_hline(y=tr["acc_thr"], line=dict(color=UI.DROP, dash="dot", width=1.3), row=1, col=1)
        for k in ["eu", "au"]:
            fig.add_trace(go.Scatter(y=roll(tr["signals"][k]), name=SNAME[k],
                                     line=dict(color=UI.SIG[k], width=2.2)), 2, 1)
        fig.add_trace(go.Scatter(y=roll(tr["signals"]["lr"]), name="log(EU/AU)",
                                 line=dict(color=UI.SIG["lr"], width=2.8)), 3, 1)
        dets = ["DDM[err]", "ADWIN[err]", "PH[err]", "KS[x]", "EM-GMM[au,eu]", "PH[conf]", "PH[tu]", "PH[eu]",
                "ADWIN[lr]", "BOCPD[lr]", "PH[lr]"]
        for d in dets:
            a = tr["alarms"].get(d, [])
            fig.add_trace(go.Scatter(x=a, y=[d] * len(a), mode="markers",
                                     marker=dict(symbol="triangle-down", size=12, color=UI.detector_color(d),
                                                 line=dict(color="white", width=1)),
                                     name=d, showlegend=False, hovertemplate=f"{d}<br>t=%{{x}}<extra></extra>"), 4, 1)
        for r in range(1, 5):
            if t0 is not None:
                fig.add_vrect(x0=t0, x1=t0 + max(stt.width, 1), fillcolor=UI.RAMP, line_width=0, layer="below",
                              row=r, col=1)
                fig.add_vline(x=t0, line=dict(color="#9C97B5", dash="dash", width=1.2), row=r, col=1)
            if t_acc is not None:
                fig.add_vline(x=t_acc, line=dict(color=UI.DROP, width=1.4), row=r, col=1)
        UI.style_fig(fig, height=880)
        fig.update_layout(hovermode="x unified", legend=dict(y=1.06))
        st.plotly_chart(fig, width="stretch")
        st.html('<p class="bdx-note">' + T(
            'Dashed line: the drift begins. Shaded band: the drift is ramping in. '
            'Amber line: the accuracy drop becomes visible. Triangles: alarms; coloured ones use no labels, '
            'dark grey ones need the true labels.',
            'Linia punctată: începe devierea. Banda umbrită: devierea crește treptat. '
            'Linia chihlimbarie: scăderea acurateței devine vizibilă. Triunghiurile: alarme; cele colorate nu '
            'folosesc etichete, cele gri închis necesită etichetele reale.') + '</p>')

        fmt = lambda v: "—" if pd.isna(v) else N(v)
        show = pd.DataFrame({
            "Detector": rows["detector"],
            T("Needs labels?", "Necesită etichete?"): rows["family"].map(
                {"supervised (labels)": T("yes", "da"), "uncertainty (label-free)": T("no", "nu"),
                 "input space": T("no (raw inputs)", "nu (intrări brute)")}),
            T("First alarm after the drift began", "Prima alarmă după începutul devierii"): rows["first_alarm"].map(fmt),
            T("Delay (records)", "Întârziere (înregistrări)"): rows["delay"].map(fmt),
            T("Lead over the visible drop", "Avans față de scăderea vizibilă"):
                rows["lead"].map(lambda v: "—" if pd.isna(v) else ("+" if v > 0 else "−" if v < 0 else "") + N(abs(v))),
            T("False alarms", "Alarme false"): rows["n_false_alarms"].map(fmt),
        })
        st.subheader(T("Every detector, side by side", "Toate detectoarele, unul lângă altul"))
        st.dataframe(show, width="stretch", hide_index=True)
        if out["signal_rows"]:
            sr = pd.DataFrame(out["signal_rows"])[["signal", "auroc", "cohen_d", "mean_pre", "mean_post"]]
            sr["signal"] = sr["signal"].map(SLAB)
            sr = sr.rename(columns={"signal": T("Signal", "Semnal"), "auroc": T("Separation (AUROC)", "Separare (AUROC)"),
                                    "cohen_d": T("Effect size (Cohen d)", "Mărimea efectului (Cohen d)"),
                                    "mean_pre": T("Mean before", "Media înainte"), "mean_post": T("Mean after", "Media după")})
            st.subheader(T("How well each signal separates drifted from normal records",
                           "Cât de bine separă fiecare semnal înregistrările deviate de cele normale"))
            st.html('<p class="bdx-note">' + T("AUROC: 0.5 means no information, 1 means perfect separation.",
                                                "AUROC: 0.5 înseamnă nicio informație, 1 înseamnă separare perfectă.")
                    + '</p>')
            st.dataframe(sr.round(3), width="stretch", hide_index=True)
        st.download_button(T("Download the scorecard (CSV)", "Descarcă tabelul de rezultate (CSV)"),
                           rows.to_csv(index=False), "scorecard.csv")
    else:
        st.info(T("Choose a dataset, a drift and a model in the sidebar, then press **Run experiment**. "
                  "A run takes 10–60 seconds.",
                  "Alege în bara laterală un set de date, o deviere și un model, apoi apasă **Rulează experimentul**. "
                  "O rulare durează 10–60 de secunde."))

# ============================================================================ XAI
with tabs[2]:
    out = st.session_state.get("last_out")
    if not out or out["traces"]["t0"] is None:
        st.info(T("Run an experiment with a drift (covariate, sensor or real) first; this tab explains its alarm.",
                  "Rulează mai întâi un experiment cu deviere (covariabile, senzor sau reală); acest tab îi explică alarma."))
    else:
        from bdx import xai as XAI
        tr = out["traces"]
        stt = tr["stream"]
        cand = [d for d in tr["alarms"] if not d.endswith("[err]")]
        c1, c2, c3 = st.columns(3)
        det = c1.selectbox(T("Alarm that defines the explained window", "Alarma care definește fereastra explicată"),
                           cand, index=cand.index("PH[lr]"))
        n_exp = c2.slider(T("Samples per window", "Eșantioane pe fereastră"), 10, 60, 25, 5)
        use_lime = c3.checkbox(T("Include LIME (slower)", "Include LIME (mai lent)"), value=True)
        al = [a for a in tr["alarms"][det] if a >= stt.t0]
        alarm = al[0] if al else stt.t0 + stt.width // 2
        st.write(T(f"Explaining the {n_exp} highest-log(EU/AU) samples in **[{alarm}, {alarm + 300})** "
                   f"({'first alarm of ' + det if al else 'no alarm — mid-ramp fallback'}) against two pre-drift windows.",
                   f"Se explică cele {n_exp} eșantioane cu cel mai mare log(EU/AU) din **[{alarm}, {alarm + 300})** "
                   f"({'prima alarmă a ' + det if al else 'nicio alarmă — se folosește mijlocul rampei'}) "
                   f"față de două ferestre de dinaintea devierii."))
        if st.button(T("Compute explanations", "Calculează explicațiile"), type="primary"):
            xcfg = copy.deepcopy(base_cfg())
            xcfg["xai"]["n_explain"] = n_exp
            with st.spinner(T("Computing KernelSHAP, GradientSHAP and LIME (1–3 minutes)",
                              "Se calculează KernelSHAP, GradientSHAP și LIME (1–3 minute)")):
                res, imps = XAI.explain_run(tr["model"], stt, tr["signals"], alarm, xcfg, use_lime=use_lime)
            st.session_state["xai"] = (res, imps)
        if "xai" in st.session_state:
            res, imps = st.session_state["xai"]
            names = stt.feature_names
            gt = set(stt.gt_features)
            ex_names = {
                "SHAP_EU": T("KernelSHAP of epistemic uncertainty (agnostic)",
                             "KernelSHAP pe incertitudinea epistemică (agnostic)"),
                "LIME_EU": T("LIME of epistemic uncertainty", "LIME pe incertitudinea epistemică"),
                "SHAP_pred": T("KernelSHAP of prediction (agnostic)", "KernelSHAP pe predicție (agnostic)"),
                "GradSHAP_pred": T("Gradient SHAP of prediction (model-specific)",
                                   "Gradient SHAP pe predicție (specific modelului)"),
                "LIME_pred": T("LIME of prediction", "LIME pe predicție")}
            ex = st.selectbox(T("Explainer", "Metoda de explicare"), list(imps), format_func=lambda e: ex_names.get(e, e))
            v = imps[ex]
            order = np.argsort(v["delta"])[::-1][:15]
            fig = go.Figure(go.Bar(x=v["delta"][order][::-1], y=[names[i] + ("  ★" if i in gt else "") for i in order][::-1],
                                   orientation="h", marker_color=[UI.CORAL if i in gt else "#B8B2CC" for i in order][::-1],
                                   marker_line_width=0, hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
            UI.style_fig(fig, height=500)
            fig.update_layout(title=dict(text=T("Gain in importance at the alarm (★ and coral: the columns we really "
                                                "changed)", "Creșterea importanței la alarmă (★ și coral: coloanele "
                                                "modificate cu adevărat)"), font=dict(family=UI.FONT_HEAD, size=16)),
                              bargap=0.35)
            st.plotly_chart(fig, width="stretch")
            k1, k2, k3 = st.columns(3)
            k1.metric("Precision@k", f"{res[ex + '_prec_at_k']:.2f}",
                      help=T(f"chance = {res['chance_prec_at_k']:.2f}", f"hazard = {res['chance_prec_at_k']:.2f}"))
            k2.metric(T("Ground-truth rank score", "Scorul de rang al trăsăturilor deviate"), f"{res[ex + '_gt_rank']:.2f}",
                      help=T("1 = drifted features on top, 0.5 = chance",
                             "1 = trăsăturile deviate pe primele locuri, 0.5 = hazard"))
            k3.metric(T("Rank stability ρ (pre→alarm)", "Stabilitatea rangurilor ρ (înainte→alarmă)"),
                      f"{res[ex + '_rho_drift']:.2f}",
                      delta=T(f"{res[ex + '_rho_drift'] - res[ex + '_rho_null']:+.2f} vs pre→pre",
                              f"{res[ex + '_rho_drift'] - res[ex + '_rho_null']:+.2f} față de înainte→înainte"))
            with st.expander(T("All XAI metrics", "Toate metricile XAI")):
                st.json({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})

# ============================================================================ research results
with tabs[3]:
    hyp = os.path.join(RES, "summary", "hypotheses.md")
    if not os.path.exists(hyp):
        st.warning(T("No precomputed results found. Run `python -m bdx.run_all && python -m bdx.analyze`.",
                     "Nu există rezultate calculate. Rulează `python -m bdx.run_all && python -m bdx.analyze`."))
    else:
        st.html('<p class="bdx-prose">' + T(
            'Results of the full experiment: every dataset and drift type, 3 pre-registered '
            'models plus the random forest added afterwards, 5 seeds each. Regenerate them with '
            '<code>python -m bdx.run_all</code> and <code>python -m bdx.analyze</code>.',
            'Rezultatele experimentului complet: toate seturile de date și tipurile de deviere, 3 modele stabilite '
            'dinainte plus pădurea aleatoare adăugată ulterior, câte 5 semințe fiecare. Se regenerează cu '
            '<code>python -m bdx.run_all</code> și <code>python -m bdx.analyze</code>.') + '</p>')
        figs = [("fig1_example_trace.png", T("One run: the label-free alarm and the visible accuracy drop",
                                             "O rulare: alarma fără etichete și scăderea vizibilă a acurateței")),
                ("fig2_signal_auroc.png", T("H2 and H3: how well each signal separates drifted data",
                                            "H2 și H3: cât de bine separă fiecare semnal datele deviate")),
                ("fig3_tradeoff.png", T("H1: early warnings against false alarms",
                                        "H1: avertizările timpurii față de alarmele false")),
                ("fig4_meta_forest.png", T("Meta-analysis across datasets", "Meta-analiza pe seturile de date")),
                ("fig5_xai.png", T("H4: do explanations find the drifted columns?",
                                   "H4: găsesc explicațiile coloanele deviate?")),
                ("fig7_bayes_vs_freq.png", T("H5: Bayesian vs frequentist models", "H5: modele bayesiene vs frecventiste")),
                ("fig6_elec2_leadlag.png", T("Real electricity stream: does uncertainty lead the error?",
                                             "Fluxul real de electricitate: anticipează incertitudinea eroarea?"))]
        fig_dir = os.path.join(FIG, "ro") if RO and os.path.isdir(os.path.join(FIG, "ro")) else FIG
        for i in range(0, len(figs), 2):
            cols = st.columns(2)
            for c, (f, cap) in zip(cols, figs[i:i + 2]):
                p = os.path.join(fig_dir, f)
                if not os.path.exists(p):
                    p = os.path.join(FIG, f)
                if os.path.exists(p):
                    c.image(p, caption=cap, width="stretch")
        with st.expander(T("Every hypothesis test, with its numbers",
                           "Toate testele ipotezelor, cu valorile lor (raport generat automat, în engleză)"),
                         expanded=False):
            st.markdown(open(hyp).read())
        sdir = os.path.join(RES, "summary")
        with st.expander(T("All summary tables", "Toate tabelele de sinteză")):
            for f in sorted(os.listdir(sdir)):
                if f.endswith(".csv"):
                    st.markdown(f"**{f}**")
                    st.dataframe(pd.read_csv(os.path.join(sdir, f)), width="stretch", hide_index=True)

# ============================================================================ theory
with tabs[4]:
    _, mid, _ = st.columns([0.08, 1, 0.25])
    mid.markdown(T(r"""
### 1 · From likelihood to posterior
A classifier defines a likelihood $p(y\mid x,\theta)$. **Maximum likelihood** picks
$\hat\theta_{MLE}=\arg\max_\theta\sum_i\log p(y_i\mid x_i,\theta)$ — minimising cross-entropy *is* MLE.
Adding L2 weight decay $\lambda\|\theta\|^2$ is **MAP** with a Gaussian prior $\theta\sim\mathcal N(0,\sigma^2 I)$.
The **Bayesian** answer keeps the whole posterior $p(\theta\mid\mathcal D)\propto p(\mathcal D\mid\theta)p(\theta)$
and predicts with $p(y\mid x,\mathcal D)=\int p(y\mid x,\theta)\,p(\theta\mid\mathcal D)\,d\theta\approx\frac1T\sum_t p(y\mid x,\theta_t)$.

### 2 · MC-Dropout = approximate variational inference
Gal & Ghahramani (2016): training with dropout minimises a KL divergence to the posterior for a
Bernoulli variational family $q(\theta)$. Keeping dropout *on* at test time samples $\theta_t\sim q(\theta)$.

### 3 · Decomposition
$$\underbrace{\mathcal H\Big[\tfrac1T\sum_t p_t\Big]}_{\text{total}}=\underbrace{\tfrac1T\sum_t\mathcal H[p_t]}_{\text{aleatoric}}+\underbrace{\mathcal I(y;\theta\mid x,\mathcal D)}_{\text{epistemic}}$$
EU (mutual information) is large when sampled models **disagree** — typical of inputs unlike the training data.
AU is large when every model agrees the classes **overlap**. Proposed signal: $\log(EU+\epsilon)-\log(AU+\epsilon)$.

### 4 · Fisher (frequentist) vs Bayes
Frequentist uncertainty comes from the *sampling distribution* of an estimator — here, the bootstrap:
refit an SVM on resampled data and look at the spread. A **random forest** does the same with trees: each tree
grows on its own bootstrap sample, and their disagreement is the epistemic part. Bayesian uncertainty is a
*posterior over parameters*.
Both produce a set of predictive distributions, so the same decomposition applies, but they mean different things.

### 5 · Why this cannot see every drift
EU and AU are functions of $x$ only. If $P(x)$ stays fixed and only $P(y\mid x)$ changes (real concept drift),
the distribution of any label-free signal is **unchanged** — no detector on it can do better than chance.
That is a falsifiable boundary, tested as H3.

### 6 · Detectors
* **Page-Hinkley**: cumulative sum $m_t=\sum(x_i-\bar x_i-\delta)$; alarm when $m_t-\min m>\lambda$.
* **ADWIN**: adaptive window; cuts when two sub-window means differ beyond a Hoeffding bound.
* **BOCPD** (Bayesian): exact recursive posterior over the run length $r_t$ with a Normal-Inverse-Gamma prior;
  alarm when $P(r_t<4\mid x_{1:t})$ is high.
* **EM-GMM**: mixture density of (log AU, log EU) fitted by Expectation-Maximisation; alarm when a batch is unlikely.

### 7 · XAI
SHAP gives each feature its Shapley value — its average marginal contribution over all coalitions.
LIME fits a local linear surrogate around one sample. We explain **EU itself**: if the alarm is about a real
input shift, the features whose attribution grows most should be the ones that drifted.

### 8 · Statistics on dependent data
Samples in a stream are autocorrelated, so tests use **independent runs (seeds)** as units, the **moving-block
bootstrap** inside a stream, **TOST** to show equivalence to chance, Holm correction, and a **DerSimonian-Laird**
random-effects meta-analysis across datasets.
""",
r"""
### 1 · De la verosimilitate la distribuția aposteriori
Un clasificator definește o verosimilitate $p(y\mid x,\theta)$. **Estimarea de verosimilitate maximă** (MLE) alege
$\hat\theta_{MLE}=\arg\max_\theta\sum_i\log p(y_i\mid x_i,\theta)$ — minimizarea entropiei încrucișate *este* MLE.
Adăugarea penalizării L2 $\lambda\|\theta\|^2$ este **MAP** cu un apriori gaussian $\theta\sim\mathcal N(0,\sigma^2 I)$.
Răspunsul **bayesian** păstrează întreaga distribuție aposteriori $p(\theta\mid\mathcal D)\propto p(\mathcal D\mid\theta)p(\theta)$
și prezice cu $p(y\mid x,\mathcal D)=\int p(y\mid x,\theta)\,p(\theta\mid\mathcal D)\,d\theta\approx\frac1T\sum_t p(y\mid x,\theta_t)$.

### 2 · MC-Dropout = inferență variațională aproximativă
Gal & Ghahramani (2016): antrenarea cu dropout minimizează o divergență KL față de distribuția aposteriori, pentru o
familie variațională Bernoulli $q(\theta)$. Lăsarea dropout-ului *activ* la predicție extrage $\theta_t\sim q(\theta)$.

### 3 · Descompunerea
$$\underbrace{\mathcal H\Big[\tfrac1T\sum_t p_t\Big]}_{\text{totală}}=\underbrace{\tfrac1T\sum_t\mathcal H[p_t]}_{\text{aleatorică}}+\underbrace{\mathcal I(y;\theta\mid x,\mathcal D)}_{\text{epistemică}}$$
EU (informația mutuală) este mare când modelele extrase **nu sunt de acord** — tipic pentru intrări diferite de datele
de antrenare. AU este mare când toate modelele sunt de acord că clasele **se suprapun**.
Semnalul propus: $\log(EU+\epsilon)-\log(AU+\epsilon)$.

### 4 · Fisher (frecventist) vs Bayes
Incertitudinea frecventistă provine din *distribuția de eșantionare* a unui estimator — aici, bootstrap-ul:
reantrenăm un SVM pe date re-eșantionate și privim dispersia. O **pădure aleatoare** face același lucru cu arbori:
fiecare arbore crește pe propriul eșantion bootstrap, iar dezacordul lor este partea epistemică. Incertitudinea
bayesiană este o *distribuție aposteriori asupra parametrilor*. Ambele produc o mulțime de distribuții predictive,
deci se aplică aceeași descompunere, dar înseamnă lucruri diferite.

### 5 · De ce nu poate vedea orice deviere
EU și AU sunt funcții numai de $x$. Dacă $P(x)$ rămâne fix și se schimbă doar $P(y\mid x)$ (deviere conceptuală reală),
distribuția oricărui semnal fără etichete **nu se schimbă** — niciun detector aplicat pe el nu poate fi mai bun decât
hazardul. Aceasta este o limită care poate fi infirmată, testată ca H3.

### 6 · Detectoarele
* **Page-Hinkley**: suma cumulată $m_t=\sum(x_i-\bar x_i-\delta)$; alarmă când $m_t-\min m>\lambda$.
* **ADWIN**: fereastră adaptivă; o taie când mediile a două sub-ferestre diferă peste o margine Hoeffding.
* **BOCPD** (bayesian): distribuția aposteriori exactă, recursivă, a lungimii secvenței $r_t$, cu apriori
  Normal-Invers-Gamma; alarmă când $P(r_t<4\mid x_{1:t})$ este mare.
* **EM-GMM**: densitatea de amestec a perechii (log AU, log EU), estimată prin algoritmul Expectation-Maximisation;
  alarmă când un lot de date este puțin probabil.

### 7 · XAI (inteligență artificială explicabilă)
SHAP atribuie fiecărei trăsături valoarea Shapley — contribuția sa marginală medie peste toate coalițiile.
LIME potrivește un model liniar local în jurul unui eșantion. Noi explicăm **chiar EU**: dacă alarma se datorează
unei schimbări reale a intrărilor, trăsăturile a căror atribuire crește cel mai mult ar trebui să fie cele deviate.

### 8 · Statistică pe date dependente
Eșantioanele dintr-un flux sunt autocorelate, deci testele folosesc ca unități **rulările independente (semințele)**,
**bootstrap-ul pe blocuri mobile** în interiorul unui flux, **TOST** pentru a arăta echivalența cu hazardul, corecția
Holm și o meta-analiză cu efecte aleatoare **DerSimonian-Laird** între seturile de date.
"""))

# ============================================================================ reproducibility
with tabs[5]:
    st.markdown(T("Every number in the paper comes from `config/default.yaml` and the seeds in it. "
                  "Data files are verified by SHA-256.",
                  "Fiecare număr din articol provine din `config/default.yaml` și din semințele definite acolo. "
                  "Fișierele de date sunt verificate prin SHA-256."))
    st.markdown("""
```bash
git clone <your-repo-url> && cd <project-folder>
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -m pytest -q                         # unit tests / teste unitare
python -m bdx.run_all                       # full grid -> results/*.csv  (--quick: 5-10 min)
python -m bdx.analyze                       # tables, tests, figures / tabele, teste, figuri
streamlit run streamlit_app.py
```
""")
    rows = [{T("key", "cheie"): k, T("dataset", "set de date"): v.name, T("file", "fișier"): v.file, "sha256": v.sha256,
             T("licence", "licență"): v.licence, T("citation", "citare"): v.citation} for k, v in D.SPECS.items()]
    st.subheader(T("Data provenance", "Proveniența datelor"))
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    mp = os.path.join(RES, "manifest.json")
    if os.path.exists(mp):
        with st.expander(T("Run manifest (versions, config hash)",
                           "Manifestul rulării (versiuni, amprenta configurației)")):
            st.json(json.load(open(mp)))
    st.download_button(T("Download config/default.yaml", "Descarcă config/default.yaml"), open(os.path.join(ROOT, "config", "default.yaml")).read(),
                       "default.yaml")
