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

st.set_page_config(page_title="Bayesian Drift Early Warning", page_icon=":material/monitoring:", layout="wide")

from bdx import data as D  # noqa: E402
from bdx import drift as DR  # noqa: E402
from bdx import ui as UI  # noqa: E402
from bdx.models import MODEL_LABELS  # noqa: E402
from bdx.uncertainty import SIGNAL_LABELS  # noqa: E402

st.html(UI.CSS)


@st.cache_data(show_spinner=False)
def base_cfg():
    return yaml.safe_load(open(os.path.join(ROOT, "config", "default.yaml")))


@st.cache_resource(show_spinner=False)
def cached_tab(key):
    return D.load_dataset(key)


@st.cache_resource(show_spinner="Training the model and running all detectors …", max_entries=8)
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
st.sidebar.header("Set up an experiment")
st.sidebar.html('<div class="bdx-side-note">Pick data, a kind of drift and a model, then run. '
                'The drift starts at record 2&nbsp;500 and is complete by record 4&nbsp;000.</div>')
ds_names = {"credit": "Credit default (finance, UCI)", "telco": "Telco churn (customers, IBM)",
            "digits": "Digits 8×8 rotation (MNIST-like)", "sea": "SEA synthetic (real drift)",
            "elec2": "Elec2 electricity (natural drift)", "upload": "Upload your own CSV"}
dataset = st.sidebar.selectbox("Dataset", list(ds_names), format_func=ds_names.get)
upload_bytes, target, drop_cols, upload_ok = None, None, [], True
if dataset == "upload":
    up = st.sidebar.file_uploader("CSV file (one row per record, one label column)", type=["csv"])
    if up is not None:
        upload_bytes = up.getvalue()
        import io
        df_up = pd.read_csv(io.BytesIO(upload_bytes))
        cols = list(df_up.columns)
        target = st.sidebar.selectbox("Label column (what the model predicts)", cols, index=len(cols) - 1)
        # columns that would only add noise: identifiers and free text (almost one value per row)
        guess = [c for c in cols if c != target and (c.lower() in ("id", "index", "unnamed: 0") or c.lower().endswith("id")
                 or (not pd.api.types.is_numeric_dtype(df_up[c]) and df_up[c].nunique() > 0.5 * len(df_up)))]
        drop_cols = st.sidebar.multiselect("Columns to ignore (IDs, dates, free text)",
                                           [c for c in cols if c != target], default=guess)
        n_cls = df_up[target].nunique()
        st.sidebar.caption(f"{len(df_up):,} rows · {len(cols) - 1 - len(drop_cols)} input columns · "
                           f"{n_cls} label classes")
        if n_cls < 2 or n_cls > 20:
            upload_ok = False
            st.sidebar.error("The label column must have 2–20 classes (classification). "
                             "For a numeric target, bin it into classes first.")
        elif len(df_up) < 1000:
            st.sidebar.warning("Fewer than 1 000 rows: the run works, but results will be noisy.")

allowed = {"digits": ["covariate", "none"], "sea": ["real", "none"], "elec2": ["natural"]}.get(
    dataset, ["covariate", "sensor", "real", "none"])
drift_help = {"covariate": "Covariate: new kinds of records arrive",
              "sensor": "Sensor: two columns shift (a broken pipeline)",
              "real": "Real: same inputs, labels change meaning",
              "none": "None: nothing changes (false-alarm check)",
              "natural": "Natural: real time-ordered stream"}
drift = st.sidebar.selectbox("Drift type", allowed, format_func=drift_help.get)
model = st.sidebar.selectbox("Probabilistic model", ["mcd", "ens", "bsvm", "rf"], format_func=MODEL_LABELS.get)
seed = st.sidebar.number_input("Seed", 0, 999, 0)

with st.sidebar.expander("Advanced settings"):
    cfg = copy.deepcopy(cfg0)
    cfg["stream_len"] = st.slider("Stream length", 2000, 10000, cfg0["stream_len"], 500)
    cfg["t0"] = st.slider("Drift onset t0", 500, cfg["stream_len"] - 500, min(cfg0["t0"], cfg["stream_len"] - 500), 100)
    cfg["width"] = st.slider("Ramp width (0 = abrupt)", 0, 3000, cfg0["width"], 100)
    cfg["sensor_delta"] = st.slider("Sensor shift δ (σ units)", 0.5, 5.0, cfg0["sensor_delta"], 0.5)
    cfg["flip_rho"] = st.slider("Real drift: flip probability ρ", 0.1, 1.0, cfg0["flip_rho"], 0.1)
    cfg["model"]["T"] = st.slider("MC-Dropout samples T", 5, 100, cfg0["model"]["T"], 5)
    cfg["model"]["dropout"] = st.slider("Dropout rate p", 0.05, 0.5, cfg0["model"]["dropout"], 0.05)
    cfg["detectors"]["target_far"] = st.slider("Target false-alarm probability", 0.01, 0.3,
                                               cfg0["detectors"]["target_far"], 0.01)
    cfg["detectors"]["label_delay"] = st.slider("Label delay for supervised detectors", 0, 2000, 0, 50)
run_btn = st.sidebar.button("Run experiment", type="primary", width="stretch")

# ============================================================================ header
st.html(UI.hero_html(RES))

tabs = st.tabs(["The idea", "Run an experiment", "Explain the alarm", "Research results", "Theory", "Reproduce"])

# ============================================================================ overview
with tabs[0]:
    c1, c2 = st.columns([1, 1.05], gap="large")
    with c1:
        st.subheader("Why it matters")
        st.html("""<div class="bdx-prose">
<p>A model trained on last year's clients keeps predicting when a new kind of client appears. Its accuracy
slowly falls, but nobody can measure that until the true outcomes arrive, often months later. Classic drift
detectors watch the error rate, so they always wait for those labels.</p>
<p>This project asks the model itself. Many slightly different versions of the model vote on every record.
When they all hesitate, the case is simply hard: <b>aleatoric</b> uncertainty (AU). When they disagree with
each other, the model has not seen anything like it: <b>epistemic</b> uncertainty (EU). A rising
<b>log(EU/AU)</b> means the world has moved away from the training data.</p></div>""")
    with c2:
        st.subheader("What was tested, and what came out")
        st.html(UI.hypotheses_html())
    st.subheader("How one experiment runs")
    st.graphviz_chart(UI.PIPELINE_DOT, width="stretch")
    st.html('<p class="bdx-note">Start in the sidebar: choose a dataset (or upload a CSV), a drift and a model, '
            'then open <b>Run an experiment</b>. After that, <b>Explain the alarm</b> shows which columns made the '
            'model unsure.</p>')

# ============================================================================ live experiment
with tabs[1]:
    if dataset == "upload" and (upload_bytes is None or not upload_ok):
        st.warning("Upload a CSV in the sidebar first." if upload_bytes is None else
                   "Choose a label column with 2–20 classes in the sidebar.")
    elif run_btn or st.session_state.get("last_key"):
        key = (dataset, drift, model, int(seed), json.dumps(cfg, sort_keys=True), target, json.dumps(sorted(drop_cols)))
        if run_btn:
            st.session_state["last_key"] = key
        key = st.session_state["last_key"]
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
                lead = f"{abs(d_)} records {'before' if d_ >= 0 else 'after'} the visible drop"
            else:
                lead = "no visible drop to compare with" if pd.notna(first_lr) else "never fired"
            st.html(UI.readout_html([
                ("Accuracy before the drift", f"{tr['acc_ref']:.3f}", "measured on calibration data", ""),
                ("Accuracy after the drift", f"{acc_post:.3f}" if acc_post is not None else "—",
                 f"{acc_post - tr['acc_ref']:+.3f} compared with before" if acc_post is not None else "", "ink"),
                ("Accuracy drop visible at", f"t = {t_acc}" if t_acc is not None else "never",
                 "needs the true labels", "drop" if t_acc is not None else ""),
                ("Our alarm, log(EU/AU)", f"t = {int(first_lr)}" if pd.notna(first_lr) else "none", lead, "coral"),
            ]))
            if pd.notna(first_err) and pd.notna(first_lr):
                gap = int(first_err) - int(first_lr)
                st.html(f'<p class="bdx-note">The first error-based alarm (which needs labels) came at t = {int(first_err)}, '
                        f'{abs(gap)} records {"after" if gap >= 0 else "before"} ours.</p>')
        else:
            st.html(UI.readout_html([
                ("Accuracy before", f"{tr['acc_ref']:.3f}", "calibration data", ""),
                ("Accuracy on the stream", f"{1 - tr['err'].mean():.3f}", "", ""),
                ("Stream length", f"{L:,}", "records", ""),
                ("Alarms of our detector", str(len(tr["alarms"]["PH[lr]"])), "every alarm here is false"
                 if stt.drift_type == "none" else "no known drift time", "coral"),
            ]))

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            row_heights=[0.24, 0.24, 0.22, 0.30],
                            subplot_titles=("Accuracy over the last 300 records (needs labels)",
                                            "Aleatoric vs epistemic uncertainty", "log(EU/AU), the proposed signal",
                                            "Alarms of each detector"))
        fig.add_trace(go.Scatter(y=roll(1 - tr["err"], 300), name="accuracy", line=dict(color=UI.INK, width=2)), 1, 1)
        if tr["acc_thr"] is not None:
            fig.add_hline(y=tr["acc_thr"], line=dict(color=UI.DROP, dash="dot", width=1.3), row=1, col=1)
        for k in ["eu", "au"]:
            fig.add_trace(go.Scatter(y=roll(tr["signals"][k]), name=UI.SIG_NAME[k],
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
        st.html('<p class="bdx-note">Dashed line: the drift begins. Shaded band: the drift is ramping in. '
                'Amber line: the accuracy drop becomes visible. Triangles: alarms; coloured ones use no labels, '
                'dark grey ones need the true labels.</p>')

        fmt = lambda v: "—" if pd.isna(v) else f"{int(v):,}"
        show = pd.DataFrame({
            "Detector": rows["detector"],
            "Needs labels?": rows["family"].map({"supervised (labels)": "yes", "uncertainty (label-free)": "no",
                                                 "input space": "no (raw inputs)"}),
            "First alarm after the drift began": rows["first_alarm"].map(fmt),
            "Delay (records)": rows["delay"].map(fmt),
            "Lead over the visible drop": rows["lead"].map(lambda v: "—" if pd.isna(v) else f"{int(v):+,}"),
            "False alarms": rows["n_false_alarms"].map(fmt),
        })
        st.subheader("Every detector, side by side")
        st.dataframe(show, width="stretch", hide_index=True)
        if out["signal_rows"]:
            sr = pd.DataFrame(out["signal_rows"])[["signal", "auroc", "cohen_d", "mean_pre", "mean_post"]]
            sr["signal"] = sr["signal"].map(SIGNAL_LABELS)
            sr = sr.rename(columns={"signal": "Signal", "auroc": "Separation (AUROC)", "cohen_d": "Effect size (Cohen d)",
                                    "mean_pre": "Mean before", "mean_post": "Mean after"})
            st.subheader("How well each signal separates drifted from normal records")
            st.html('<p class="bdx-note">AUROC: 0.5 means no information, 1 means perfect separation.</p>')
            st.dataframe(sr.round(3), width="stretch", hide_index=True)
        st.download_button("Download the scorecard (CSV)", rows.to_csv(index=False), "scorecard.csv")
    else:
        st.info("Choose a dataset, a drift and a model in the sidebar, then press **Run experiment**. "
                "A run takes 10–60 seconds.")

# ============================================================================ XAI
with tabs[2]:
    out = st.session_state.get("last_out")
    if not out or out["traces"]["t0"] is None:
        st.info("Run an experiment with a drift (covariate, sensor or real) first; this tab explains its alarm.")
    else:
        from bdx import xai as XAI
        tr = out["traces"]
        stt = tr["stream"]
        cand = [d for d in tr["alarms"] if not d.endswith("[err]")]
        c1, c2, c3 = st.columns(3)
        det = c1.selectbox("Alarm that defines the explained window", cand, index=cand.index("PH[lr]"))
        n_exp = c2.slider("Samples per window", 10, 60, 25, 5)
        use_lime = c3.checkbox("Include LIME (slower)", value=True)
        al = [a for a in tr["alarms"][det] if a >= stt.t0]
        alarm = al[0] if al else stt.t0 + stt.width // 2
        st.write(f"Explaining the {n_exp} highest-log(EU/AU) samples in **[{alarm}, {alarm + 300})** "
                 f"({'first alarm of ' + det if al else 'no alarm — mid-ramp fallback'}) against two pre-drift windows.")
        if st.button("Compute explanations", type="primary"):
            xcfg = copy.deepcopy(base_cfg())
            xcfg["xai"]["n_explain"] = n_exp
            with st.spinner("Computing KernelSHAP, GradientSHAP and LIME (1–3 minutes)"):
                res, imps = XAI.explain_run(tr["model"], stt, tr["signals"], alarm, xcfg, use_lime=use_lime)
            st.session_state["xai"] = (res, imps)
        if "xai" in st.session_state:
            res, imps = st.session_state["xai"]
            names = stt.feature_names
            gt = set(stt.gt_features)
            ex = st.selectbox("Explainer", list(imps), format_func=lambda e: {
                "SHAP_EU": "KernelSHAP of epistemic uncertainty (agnostic)", "LIME_EU": "LIME of epistemic uncertainty",
                "SHAP_pred": "KernelSHAP of prediction (agnostic)", "GradSHAP_pred": "Gradient SHAP of prediction (model-specific)",
                "LIME_pred": "LIME of prediction"}.get(e, e))
            v = imps[ex]
            order = np.argsort(v["delta"])[::-1][:15]
            fig = go.Figure(go.Bar(x=v["delta"][order][::-1], y=[names[i] + ("  ★" if i in gt else "") for i in order][::-1],
                                   orientation="h", marker_color=[UI.CORAL if i in gt else "#B8B2CC" for i in order][::-1],
                                   marker_line_width=0, hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
            UI.style_fig(fig, height=500)
            fig.update_layout(title=dict(text="Gain in importance at the alarm (★ and coral: the columns we really "
                                              "changed)", font=dict(family=UI.FONT_HEAD, size=16)),
                              bargap=0.35)
            st.plotly_chart(fig, width="stretch")
            k1, k2, k3 = st.columns(3)
            k1.metric("Precision@k", f"{res[ex + '_prec_at_k']:.2f}", help=f"chance = {res['chance_prec_at_k']:.2f}")
            k2.metric("Ground-truth rank score", f"{res[ex + '_gt_rank']:.2f}", help="1 = drifted features on top, 0.5 = chance")
            k3.metric("Rank stability ρ (pre→alarm)", f"{res[ex + '_rho_drift']:.2f}",
                      delta=f"{res[ex + '_rho_drift'] - res[ex + '_rho_null']:+.2f} vs pre→pre")
            with st.expander("All XAI metrics"):
                st.json({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})

# ============================================================================ research results
with tabs[3]:
    hyp = os.path.join(RES, "summary", "hypotheses.md")
    if not os.path.exists(hyp):
        st.warning("No precomputed results found. Run `python -m bdx.run_all && python -m bdx.analyze`.")
    else:
        st.html('<p class="bdx-prose">Results of the full experiment: every dataset and drift type, 3 pre-registered '
                'models plus the random forest added afterwards, 5 seeds each. Regenerate them with '
                '<code>python -m bdx.run_all</code> and <code>python -m bdx.analyze</code>.</p>')
        figs = [("fig1_example_trace.png", "One run: the label-free alarm and the visible accuracy drop"),
                ("fig2_signal_auroc.png", "H2 and H3: how well each signal separates drifted data"),
                ("fig3_tradeoff.png", "H1: early warnings against false alarms"),
                ("fig4_meta_forest.png", "Meta-analysis across datasets"),
                ("fig5_xai.png", "H4: do explanations find the drifted columns?"),
                ("fig7_bayes_vs_freq.png", "H5: Bayesian vs frequentist models"),
                ("fig6_elec2_leadlag.png", "Real electricity stream: does uncertainty lead the error?")]
        for i in range(0, len(figs), 2):
            cols = st.columns(2)
            for c, (f, cap) in zip(cols, figs[i:i + 2]):
                p = os.path.join(FIG, f)
                if os.path.exists(p):
                    c.image(p, caption=cap, width="stretch")
        with st.expander("Every hypothesis test, with its numbers", expanded=False):
            st.markdown(open(hyp).read())
        sdir = os.path.join(RES, "summary")
        with st.expander("All summary tables"):
            for f in sorted(os.listdir(sdir)):
                if f.endswith(".csv"):
                    st.markdown(f"**{f}**")
                    st.dataframe(pd.read_csv(os.path.join(sdir, f)), width="stretch", hide_index=True)

# ============================================================================ theory
with tabs[4]:
    _, mid, _ = st.columns([0.08, 1, 0.25])
    mid.markdown(r"""
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
""")

# ============================================================================ reproducibility
with tabs[5]:
    st.markdown("""
```bash
git clone <your-repo-url> && cd bayes-drift-xai
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pytest -q                                   # unit tests of the from-scratch components
python -m bdx.run_all                       # full grid -> results/*.csv   (python -m bdx.run_all --quick for a 5-min check)
python -m bdx.analyze                       # tables, hypothesis tests, figures
streamlit run streamlit_app.py
```
Every number in the paper comes from `config/default.yaml` + the seeds in it. Data files are verified by SHA-256.
""")
    rows = [dict(key=k, dataset=v.name, file=v.file, sha256=v.sha256, licence=v.licence, citation=v.citation)
            for k, v in D.SPECS.items()]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    mp = os.path.join(RES, "manifest.json")
    if os.path.exists(mp):
        with st.expander("Run manifest (versions, config hash)"):
            st.json(json.load(open(mp)))
    st.download_button("Download config/default.yaml", open(os.path.join(ROOT, "config", "default.yaml")).read(),
                       "default.yaml")
