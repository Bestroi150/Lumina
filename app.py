import datetime
import io
import os
import re
import csv
import logging
import warnings
logging.getLogger("kraken").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Polygonizer failed")
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import json
from groq import Groq
from langdetect import detect, DetectorFactory
from dotenv import load_dotenv

load_dotenv()

from lib.display_utils import (
    display_baselines,
    display_baselines_with_text,
    prepare_segments,
    open_image,
)
from lib.kraken_utils import (
    load_model_seg,
    load_model_rec,
    segment_image,
    recognize_text,
)

DetectorFactory.seed = 0

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Lumina Multimodality",
    page_icon="✨",
)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "wizard_step": 1,
    "results_history": [],
    "combined_text": "",
    "geolocation_history": [],
    "processed_df": None,
    "query_log": [],
    "ner_entities": [],
    "ocr_pages": [],
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Step indicator ── */
.step-bar {
    display: flex;
    align-items: flex-start;
    margin: 1.25rem 0 2rem 0;
}
.step-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    flex: 1;
    position: relative;
}
.step-connector {
    position: absolute;
    top: 17px;
    left: 50%;
    width: 100%;
    height: 2px;
    background: #d0d5dd;
    z-index: 0;
}
.step-connector.done-conn { background: #12b76a; }
.step-circle {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.82rem;
    border: 2.5px solid #d0d5dd;
    background: #fff;
    color: #98a2b3;
    z-index: 1;
    position: relative;
}
.step-circle.done   { background: #12b76a; border-color: #12b76a; color: #fff; }
.step-circle.active { background: #6941c6; border-color: #6941c6; color: #fff; }
.step-label {
    font-size: 0.72rem;
    color: #98a2b3;
    margin-top: 6px;
    text-align: center;
    line-height: 1.3;
}
.step-label.active-lbl { color: #6941c6; font-weight: 700; }
.step-label.done-lbl   { color: #027a48; }
/* ── Section card ── */
.wizard-section {
    background: #f9fafb;
    border: 1px solid #e4e7ec;
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.4rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── STATIC ASSETS ─────────────────────────────────────────────────────────────
def get_real_path(path: str) -> str:
    return os.path.join(os.path.dirname(__file__), path)


def write_temporary_model(file_path: str, uploaded_file) -> None:
    full_path = get_real_path(file_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "wb") as fh:
        fh.write(uploaded_file.getbuffer())


@st.cache_resource(show_spinner=False)
def get_seg_model():
    return load_model_seg(get_real_path("data/default/blla.mlmodel"))

MODEL_SEG_DEFAULT = get_seg_model()

HARDCODED_MODELS = [
    {
        "name": "catmus-medieval-160.mlmodel",
        "path": "models/catmus-medieval-160.mlmodel",
        "language": "Latin",
        "meta": None,
    },
    {
        "name": "catmus-tiny.mlmodel",
        "path": "models/catmus-tiny.mlmodel",
        "language": "Latin",
        "meta": None,
    },
    {
        "name": "e-NDP_V7.mlmodel",
        "path": "models/e-NDP_V7.mlmodel",
        "language": "French",
        "meta": None,
    },
    {
        "name": "lectaurep_base.mlmodel",
        "path": "models/lectaurep_base.mlmodel",
        "language": "French",
        "meta": None,
    },
    {
        "name": "german_handwriting.mlmodel",
        "path": "models/german_handwriting.mlmodel",
        "language": "German",
        "meta": None,
    },
    {
        "name": "McCATMuS_nfd_nofix_V1.mlmodel",
        "path": "models/McCATMuS_nfd_nofix_V1.mlmodel",
        "language": "Multilang",
        "meta": (
            "Chagué, A. (2024). McCATMuS - Transcription model for handwritten, "
            "printed and typewritten documents from the 16th century to the 21st century. "
            "Zenodo. https://doi.org/10.5281/zenodo.13788177"
        ),
    },
]

# ── GROQ GEOCODING ────────────────────────────────────────────────────────────
@st.cache_resource
def get_groq_client():
    api_key = os.environ.get("GROQ_API") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


_GEO_SYSTEM_MSG = {
    "role": "system",
    "content": (
        "You are a skilled geographist. When provided with a location query, respond only with a "
        "JSON object containing three keys: 'lat' (latitude in decimal degrees, WGS 84), "
        "'lon' (longitude in decimal degrees, WGS 84), and 'url' which is a valid link to "
        "OpenStreetMap in the format "
        "'https://www.openstreetmap.org/#map=14/<lat>/<lon>&layers=H'. "
        "Do not include any additional text."
    ),
}


def get_coordinates(query: str) -> str:
    client = get_groq_client()
    if client is None:
        return json.dumps(
            {"error": "GROQ API key not configured. Set GROQ_API or GROQ_API_KEY in your .env file."}
        )
    try:
        response = client.chat.completions.create(
            messages=[_GEO_SYSTEM_MSG, {"role": "user", "content": query}],
            model="openai/gpt-oss-20b",
            temperature=0.0,
        )
        return response.choices[0].message.content
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── FRENCH NER ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_ner_model():
    """Load the spaCy French NER model. Returns (model, error_message)."""
    try:
        import spacy  # noqa: PLC0415

        nlp = spacy.load("fr_core_news_lg")
        return nlp, None
    except ModuleNotFoundError:
        return None, "spaCy is not installed. Run: `pip install spacy`"
    except OSError:
        return None, (
            "French model **`fr_core_news_lg`** not found. "
            "Run: `python -m spacy download fr_core_news_lg`"
        )


def run_french_ner(text: str):
    """Return (list_of_entity_dicts, error_string_or_None)."""
    nlp, err = load_ner_model()
    if err:
        return [], err
    doc = nlp(text)
    entities = [
        {
            "text": ent.text,
            "label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
        }
        for ent in doc.ents
        if ent.label_ in ("LOC", "GPE", "MISC")
    ]
    return entities, None


# ── DATA MINING ───────────────────────────────────────────────────────────────
def _parse_line(line: str):
    num_part, _sep, text = line.partition(":")
    return num_part.strip(), text.strip()


def _tokenize(text: str):
    return re.findall(r"\b\w+\b", text)


def _detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"


def _extract_context(tokens, index: int, before: int = 5, after: int = 6) -> str:
    start = max(0, index - before)
    end = min(len(tokens), index + after + 1)
    return " ".join(tokens[start:end])


def process_text_data(text_content: str) -> pd.DataFrame:
    rows = []
    for line_counter, line in enumerate(text_content.splitlines(), start=1):
        if not line.strip():
            continue
        num_part, sep, text_part = line.partition(":")
        # Treat as structured (line_number: text) only when the prefix before ":"
        # is short and purely numeric — otherwise use the full line as text.
        if sep and len(num_part.strip()) <= 6 and num_part.strip().replace(".", "").isdigit():
            line_number = num_part.strip()
            text = text_part.strip()
        else:
            line_number = str(line_counter)
            text = line.strip()
        if not text:
            continue
        tokens = _tokenize(text)
        for i, token in enumerate(tokens):
            rows.append(
                {
                    "lemma": token.lower(),
                    "line": line_number,
                    "context": _extract_context(tokens, i),
                }
            )
    return pd.DataFrame(rows, columns=["lemma", "line", "context"])


# ── STEP INDICATOR ────────────────────────────────────────────────────────────
_STEP_LABELS = ["Transcription", "Data Mining", "NER", "Geolocation", "Export"]


def render_step_indicator(current: int) -> None:
    n = len(_STEP_LABELS)
    parts = ['<div class="step-bar">']
    for i, label in enumerate(_STEP_LABELS, start=1):
        is_done = i < current
        is_active = i == current
        circle_cls = "done" if is_done else ("active" if is_active else "")
        label_cls = "done-lbl" if is_done else ("active-lbl" if is_active else "")
        icon = "&#10003;" if is_done else str(i)
        connector = (
            '<div class="step-connector {}"></div>'.format("done-conn" if is_done else "")
            if i < n
            else ""
        )
        parts.append(
            '<div class="step-item">'
            + connector
            + '<div class="step-circle {}">{}</div>'.format(circle_cls, icon)
            + '<div class="step-label {}">{}</div>'.format(label_cls, label)
            + "</div>"
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


# ── NAVIGATION BUTTONS ────────────────────────────────────────────────────────
def nav_buttons(
    *,
    show_back: bool = True,
    show_next: bool = True,
    next_label: str = "Next \u2192",
    next_disabled: bool = False,
) -> None:
    step = st.session_state["wizard_step"]
    st.markdown("---")
    cols = st.columns([1, 8, 1])
    if show_back:
        with cols[0]:
            if st.button("\u2190 Back", use_container_width=True, key="back_{}".format(step)):
                st.session_state["wizard_step"] -= 1
                st.rerun()
    if show_next:
        with cols[2]:
            if st.button(
                next_label,
                use_container_width=True,
                disabled=next_disabled,
                key="next_{}".format(step),
                type="primary",
            ):
                st.session_state["wizard_step"] += 1
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — TRANSCRIPTION
# ══════════════════════════════════════════════════════════════════════════════
def render_step1() -> None:
    st.subheader("Step 1 — OCR / HTR Transcription")
    st.markdown(
        "Upload manuscript or printed page images. The Kraken engine will segment "
        "and transcribe each page. You may edit the output before continuing."
    )

    # ── Sidebar: HTR model selection ──────────────────────────────────────────
    with st.sidebar:
        st.header("HTR Configuration")
        st.markdown("---")
        recognition_model_file = st.file_uploader(
            "Upload custom recognition model (.mlmodel)", type=["mlmodel"]
        )
        selected_model = st.selectbox(
            "Or select a built-in model",
            options=HARDCODED_MODELS,
            format_func=lambda m: "{} ({})".format(m["name"], m["language"]),
        )
        if recognition_model_file is not None:
            write_temporary_model("tmp/model_rec_temp.mlmodel", recognition_model_file)
            model_rec_path = get_real_path("tmp/model_rec_temp.mlmodel")
            st.success("Custom model loaded.")
        else:
            model_rec_path = get_real_path(selected_model["path"])
            st.info("Using: {} ({})".format(selected_model["name"], selected_model["language"]))
            if selected_model["meta"]:
                st.caption(selected_model["meta"])



    # ── Image uploader ────────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload page images (JPG / PNG — up to 5 pages):",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="step1_uploader",
    )
    if uploaded_files and len(uploaded_files) > 5:
        st.warning("Only the first 5 pages will be processed.")
        uploaded_files = uploaded_files[:5]

    if not uploaded_files:
        st.info("Upload at least one image to enable transcription.")

    if uploaded_files:
        if st.button("🚀 Run Transcription", type="primary", key="btn_run_htr"):
            # Clear stale text-area edits from any previous run
            for _i in range(10):
                st.session_state.pop("edited_{}".format(_i), None)
            st.session_state["ocr_pages"] = []
            st.session_state["results_history"] = []

            model_rec = load_model_rec(model_rec_path)
            progress = st.progress(0, text="Starting…")

            for idx, file in enumerate(uploaded_files):
                n_files = len(uploaded_files)
                progress.progress(
                    idx / n_files,
                    text="Page {}/{} — segmenting…".format(idx + 1, n_files),
                )
                image = open_image(file)
                baseline_seg = segment_image(image, MODEL_SEG_DEFAULT)
                baselines, boundaries = prepare_segments(baseline_seg)

                progress.progress(
                    (idx + 0.6) / n_files,
                    text="Page {}/{} — recognising text…".format(idx + 1, n_files),
                )
                pred = recognize_text(model_rec, image, baseline_seg)
                lines = [r.prediction.strip() for r in pred]
                lines_indexed = ["{}: {}".format(i, l) for i, l in enumerate(lines)]

                st.session_state["ocr_pages"].append(
                    {
                        "idx": idx,
                        "image": image,
                        "baselines": baselines,
                        "boundaries": boundaries,
                        "lines": lines,
                        "lines_indexed": lines_indexed,
                    }
                )

            progress.progress(1.0, text="Done!")
            st.success(
                "✅ Transcription complete — {} page(s) processed.".format(
                    len(uploaded_files)
                )
            )

    # ── Display results (persists across reruns) ─────────────────────────────
    if st.session_state["ocr_pages"]:
        all_texts = []
        for page_data in st.session_state["ocr_pages"]:
            idx = page_data["idx"]
            image = page_data["image"]
            baselines = page_data["baselines"]
            boundaries = page_data["boundaries"]
            lines = page_data["lines"]
            lines_indexed = page_data["lines_indexed"]

            with st.expander("Page {}".format(idx + 1), expanded=True):
                col_seg, col_txt, col_idx, col_vis = st.columns(4)

                with col_seg:
                    st.markdown("**Segmentation**")
                    fig_seg, fig_idx_view = display_baselines(image, baselines, boundaries)
                    st.pyplot(fig_seg)

                with col_txt:
                    st.markdown("**Transcription** *(editable)*")
                    edited_text = st.text_area(
                        "Edit text:",
                        value="\n".join(lines),
                        height=400,
                        key="edited_{}".format(idx),
                        label_visibility="collapsed",
                    )
                    all_texts.append(edited_text)

                with col_idx:
                    st.markdown("**Segmentation (indexed)**")
                    st.pyplot(fig_idx_view)

                with col_vis:
                    st.markdown("**Text overlay**")
                    st.pyplot(display_baselines_with_text(image, baselines, lines))

                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(
                        "💾 Download transcription (page)",
                        edited_text,
                        file_name="page_{}.txt".format(idx + 1),
                        key="dl_page_{}".format(idx),
                    )
                with col_dl2:
                    st.download_button(
                        "💾 Download indexed text (page)",
                        "\n".join(lines_indexed),
                        file_name="page_{}_indexed.txt".format(idx + 1),
                        key="dl_idx_{}".format(idx),
                    )

        # Keep combined_text and results_history in sync with current edits
        st.session_state["combined_text"] = "\n\n".join(all_texts)
        st.session_state["results_history"] = [
            {"page": i + 1, "text": t} for i, t in enumerate(all_texts)
        ]
        st.info("**{} page(s) transcribed** — edit above then proceed.".format(
            len(st.session_state["ocr_pages"])
        ))

    nav_buttons(show_back=False, show_next=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DATA MINING
# ══════════════════════════════════════════════════════════════════════════════
def render_step2() -> None:
    st.subheader("Step 2 — Data Mining & Text Analysis")
    st.markdown(
        "Tokenise the transcribed text and search for specific terms. "
        "The text from Step 1 is pre-loaded below — you may also upload a separate file."
    )

    initial_text = st.session_state.get("combined_text", "")
    uploaded_txt = st.file_uploader(
        "Upload a text file (optional override):", type=["txt"], key="step2_uploader"
    )
    if uploaded_txt is not None:
        file_text = uploaded_txt.getvalue().decode("utf-8")
        # Only overwrite when a *new* file arrives (avoid re-setting every rerun).
        if file_text != st.session_state.get("_step2_last_upload", ""):
            st.session_state["_step2_last_upload"] = file_text
            # Pre-set the keyed widget value so the text_area picks it up.
            st.session_state["step2_text"] = file_text

    text_input = st.text_area(
        "Text to analyse:",
        value=initial_text,
        height=180,
        key="step2_text",
    )

    if st.button("⚙️ Process text", type="primary", key="btn_process_text"):
        if not text_input.strip():
            st.warning("Provide some text before processing.")
        else:
            with st.spinner("Tokenising…"):
                st.session_state["processed_df"] = process_text_data(text_input)
                st.session_state["query_log"] = []
            st.success(
                "✅ Processing complete — {} tokens extracted.".format(
                    len(st.session_state["processed_df"])
                )
            )

    df = st.session_state["processed_df"]
    if df is not None and not df.empty:
        st.markdown("---")
        with st.expander("Token table preview — {} rows".format(len(df)), expanded=False):
            st.dataframe(df.head(100), use_container_width=True)

        st.markdown("**Lemma Search**")
        search_term = st.text_input("Enter a lemma to search:", key="step2_search")
        if st.button("Search", key="btn_search"):
            mask = df["lemma"].str.lower() == search_term.lower()
            df_results = df[mask].copy()
            if not df_results.empty:
                st.success("{} match(es) found for **'{}'**.".format(len(df_results), search_term))
                st.dataframe(df_results, use_container_width=True)
                st.session_state["query_log"].extend(
                    [{"query": search_term, **row} for row in df_results.to_dict("records")]
                )
                st.download_button(
                    "💾 Download search results (CSV)",
                    df_results.to_csv(index=False).encode("utf-8"),
                    file_name="search_results.csv",
                    mime="text/csv",
                )
            else:
                st.info("No matches found for '{}'.".format(search_term))

        if st.session_state["query_log"]:
            st.markdown("---")
            if st.button("📊 Plot query statistics", key="btn_plot"):
                log_df = pd.DataFrame(st.session_state["query_log"])
                log_df = log_df[log_df["lemma"].str.strip() != ""]
                if log_df.empty:
                    st.warning("No successful searches to plot yet.")
                else:
                    lemma_stats = log_df["lemma"].value_counts().reset_index()
                    lemma_stats.columns = ["Lemma", "Frequency"]
                    fig_pie = px.pie(
                        lemma_stats,
                        names="Lemma",
                        values="Frequency",
                        title="Lemma Frequency Distribution",
                        hole=0.3,
                    )
                    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                    st.plotly_chart(fig_pie, use_container_width=True)

                    chapter_stats = (
                        log_df.groupby(["line", "lemma"]).size().reset_index(name="Mentions")
                    )
                    fig_bar = px.bar(
                        chapter_stats,
                        x="line",
                        y="Mentions",
                        color="lemma",
                        title="Line-wise Lemma Mentions",
                        labels={"line": "Line", "Mentions": "Mentions"},
                    )
                    fig_bar.update_layout(barmode="stack")
                    st.plotly_chart(fig_bar, use_container_width=True)

    nav_buttons(show_back=True, show_next=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FRENCH NER
# ══════════════════════════════════════════════════════════════════════════════
def render_step3() -> None:
    st.subheader("Step 3 — French Named Entity Recognition")
    st.markdown(
        "Extract **place names and geographic entities** (`LOC` / `MISC`) "
        "from your text using a French NLP model (`fr_core_news_lg`). "
        "Detected locations are forwarded automatically to the Geolocation step."
    )

    _nlp, _ner_err = load_ner_model()
    if _ner_err:
        st.error(
            "**NER model unavailable:** {}  \n"
            "Install the model and restart the app to use this step.".format(_ner_err)
        )

    initial_text = st.session_state.get("combined_text", "")
    ner_text = st.text_area(
        "Text to analyse (pre-filled from Step 1):",
        value=initial_text,
        height=200,
        key="step3_text",
    )

    if st.button(
        "🔍 Run NER",
        type="primary",
        key="btn_run_ner",
        disabled=(_ner_err is not None),
    ):
        if not ner_text.strip():
            st.warning("Provide some text before running NER.")
        else:
            with st.spinner("Running Named Entity Recognition…"):
                entities, err = run_french_ner(ner_text)
            if err:
                st.error(err)
            elif not entities:
                st.info("No place / location entities (`LOC` / `MISC`) detected in the text.")
                st.session_state["ner_entities"] = []
            else:
                st.session_state["ner_entities"] = entities
                unique_count = len({e["text"] for e in entities})
                st.success(
                    "✅ **{} mention(s)** of places/locations detected ({} unique).".format(
                        len(entities), unique_count
                    )
                )

    if st.session_state["ner_entities"]:
        df_ner = pd.DataFrame(st.session_state["ner_entities"])
        unique_places = sorted({e["text"] for e in st.session_state["ner_entities"]})

        col_table, col_chips = st.columns([3, 2])
        with col_table:
            st.markdown("**All entity mentions ({})**".format(len(df_ner)))
            st.dataframe(df_ner, use_container_width=True)
        with col_chips:
            st.markdown("**Unique places ({})**".format(len(unique_places)))
            chips_html = " ".join(
                '<span style="background:#ede9fe;color:#5b21b6;padding:3px 10px;'
                'border-radius:999px;font-size:0.82rem;display:inline-block;'
                'margin:3px 2px">{}</span>'.format(p)
                for p in unique_places
            )
            st.markdown(chips_html, unsafe_allow_html=True)

        with st.expander("Preview entity highlights in source text", expanded=False):
            highlighted = ner_text
            for place in sorted(unique_places, key=len, reverse=True):
                highlighted = highlighted.replace(
                    place,
                    '<mark style="background:#d1fae5;border-radius:3px;'
                    'padding:0 3px"><b>{}</b></mark>'.format(place),
                )
            st.markdown(
                '<div style="line-height:1.8;font-size:0.9rem">{}</div>'.format(highlighted),
                unsafe_allow_html=True,
            )

        st.info(
            "These {} place(s) will be pre-loaded as suggestions in the Geolocation step.".format(
                len(unique_places)
            )
        )

    nav_buttons(show_back=True, show_next=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — GEOLOCATION
# ══════════════════════════════════════════════════════════════════════════════
def render_step4() -> None:
    st.subheader("Step 4 — Geolocation")
    st.markdown(
        "Geocode place names to geographic coordinates using the GROQ AI service. "
        "Locations detected by NER are pre-loaded — select which ones to geocode automatically, "
        "or use the manual form for custom queries."
    )

    if get_groq_client() is None:
        st.warning(
            "GROQ API key not configured. "
            "Add `GROQ_API=<your_key>` to your `.env` file and restart the app."
        )

    # ── Auto-geocode from NER ─────────────────────────────────────────────────
    ner_suggestions = sorted(
        {e["text"].strip() for e in st.session_state["ner_entities"] if e["text"].strip()}
    )

    if ner_suggestions:
        st.markdown("#### Entities from NER")
        selected = st.multiselect(
            "Select places to geocode:",
            options=ner_suggestions,
            default=ner_suggestions[: min(5, len(ner_suggestions))],
            key="geo_multiselect",
        )
        if selected:
            if st.button("🌍 Geocode selected entities", type="primary", key="btn_batch_geo"):
                progress_bar = st.progress(0, text="Geocoding…")
                newly_added = 0
                for i, place in enumerate(selected):
                    progress_bar.progress(
                        (i + 1) / len(selected), text="Geocoding: {}".format(place)
                    )
                    raw = get_coordinates(place)
                    try:
                        coords = json.loads(raw)
                        lat = coords.get("lat")
                        lon = coords.get("lon")
                        url = coords.get("url", "")
                        if lat is not None and lon is not None:
                            st.session_state["geolocation_history"].append(
                                {
                                    "query": place,
                                    "coordinates": "{}, {}".format(lat, lon),
                                    "url": url,
                                }
                            )
                            newly_added += 1
                        else:
                            st.warning("Could not geocode '{}': {}".format(place, raw))
                    except Exception as exc:
                        st.warning("Error geocoding '{}': {}".format(place, exc))
                progress_bar.empty()
                if newly_added:
                    st.success("✅ {} location(s) geocoded.".format(newly_added))

    # ── Manual form ───────────────────────────────────────────────────────────
    st.markdown("#### Manual query")
    with st.form(key="geo_manual_form", clear_on_submit=True):
        user_query = st.text_input(
            "Location query:", placeholder="E.g., Notre-Dame de Paris"
        )
        submitted = st.form_submit_button("Get Coordinates")
        if submitted and user_query:
            with st.spinner("Fetching coordinates…"):
                raw = get_coordinates(user_query)
            try:
                coords = json.loads(raw)
                lat = coords.get("lat")
                lon = coords.get("lon")
                url = coords.get("url", "")
                if lat is None or lon is None:
                    st.error("Invalid response: " + raw)
                else:
                    st.success("Coordinates: {}, {}".format(lat, lon))
                    if url:
                        st.markdown("[Open in OpenStreetMap]({})".format(url))
                    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))
                    st.session_state["geolocation_history"].append(
                        {
                            "query": user_query,
                            "coordinates": "{}, {}".format(lat, lon),
                            "url": url,
                        }
                    )
            except Exception as exc:
                st.error("Error parsing response: {}".format(exc))

    # ── History & map ─────────────────────────────────────────────────────────
    if st.session_state["geolocation_history"]:
        st.markdown("---")
        st.markdown(
            "#### Query history — {} entries".format(len(st.session_state["geolocation_history"]))
        )
        df_geo = pd.DataFrame(st.session_state["geolocation_history"])
        st.dataframe(df_geo, use_container_width=True)

        try:
            split = df_geo["coordinates"].str.split(", ", expand=True).astype(float)
            st.map(pd.DataFrame({"lat": split[0], "lon": split[1]}))
        except Exception:
            pass

        if st.button("🗑️ Clear history", key="btn_clear_geo"):
            st.session_state["geolocation_history"] = []
            st.rerun()

    nav_buttons(show_back=True, show_next=True, next_label="Proceed to Export \u2192")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — EXPORT
# ══════════════════════════════════════════════════════════════════════════════
def render_step5() -> None:
    st.subheader("Step 5 — Export Results")
    st.markdown(
        "Download all data produced during this session. "
        "Each section corresponds to one processing stage."
    )

    any_export = False

    # Transcription ────────────────────────────────────────────────────────────
    if st.session_state["results_history"]:
        any_export = True
        st.markdown("### 📄 Transcription")
        all_text = "\n\n".join(
            "=== Page {} ===\n{}".format(e["page"], e["text"])
            for e in st.session_state["results_history"]
        )
        st.download_button(
            "💾 Download all pages (combined .txt)",
            all_text,
            file_name="transcription_all_pages.txt",
            mime="text/plain",
            use_container_width=True,
            key="dl_export_transcription",
        )

    # Token table ──────────────────────────────────────────────────────────────
    df_tokens = st.session_state["processed_df"]
    if df_tokens is not None and not df_tokens.empty:
        any_export = True
        st.markdown("### 🔤 Token Table (Data Mining)")
        st.download_button(
            "💾 Download token table (.csv)",
            df_tokens.to_csv(index=False).encode("utf-8"),
            file_name="token_table.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_export_tokens",
        )

    # NER entities ─────────────────────────────────────────────────────────────
    if st.session_state["ner_entities"]:
        any_export = True
        st.markdown("### 🗺️ Named Entities — Places & Locations (NER)")
        df_ner = pd.DataFrame(st.session_state["ner_entities"])
        st.download_button(
            "💾 Download NER entities (.csv)",
            df_ner.to_csv(index=False).encode("utf-8"),
            file_name="ner_entities.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_export_ner",
        )

    # Geolocation ──────────────────────────────────────────────────────────────
    if st.session_state["geolocation_history"]:
        any_export = True
        st.markdown("### 📍 Geolocation Data")
        df_geo = pd.DataFrame(st.session_state["geolocation_history"])
        st.download_button(
            "💾 Download geolocation data (.csv)",
            df_geo.to_csv(index=False).encode("utf-8"),
            file_name="geolocation_history.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_export_geo",
        )

    if not any_export:
        st.info(
            "No data to export yet. Complete earlier steps and return here to download your results."
        )
    else:
        st.success("✅ All outputs are ready for download.")

    nav_buttons(show_back=True, show_next=False)


# ══════════════════════════════════════════════════════════════════════════════
# SHELL
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.image("images/logo.png")
st.sidebar.markdown("---")
st.sidebar.markdown("**Lumina Multimodality**  \nKraken HTR · Data Mining · NER · Geocoding")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Step {} / {}** — {}".format(
        st.session_state["wizard_step"],
        len(_STEP_LABELS),
        _STEP_LABELS[st.session_state["wizard_step"] - 1],
    )
)

st.title("✨ Lumina Multimodality")
render_step_indicator(st.session_state["wizard_step"])
st.markdown("---")

_step = st.session_state["wizard_step"]
if _step == 1:
    render_step1()
elif _step == 2:
    render_step2()
elif _step == 3:
    render_step3()
elif _step == 4:
    render_step4()
elif _step == 5:
    render_step5()
