import datetime
import os
import re
import csv
import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq
from langdetect import detect, DetectorFactory

# Custom library imports
from lib.display_utils import (
    display_baselines,
    display_baselines_with_text,
    prepare_segments,
    open_image
)
from lib.kraken_utils import (
    load_model_seg,
    load_model_rec,
    segment_image,
    recognize_text
)

# For consistent language detection outcomes.
DetectorFactory.seed = 0

# --- PAGE / SESSION INITIALIZATION ---
st.set_page_config(layout="wide")

if "results_history" not in st.session_state:
    st.session_state["results_history"] = []

if "geolocation_history" not in st.session_state:
    st.session_state["geolocation_history"] = []


# === PATH / FILE HELPERS ===
def get_real_path(path: str) -> str:
    return os.path.join(os.path.dirname(__file__), path)


def write_temporary_model(file_path, custom_model_loaded):
    full_path = get_real_path(file_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "wb") as file:
        file.write(custom_model_loaded.getbuffer())


# === GROQ INITIALIZATION ===
def get_groq_client():
    """
    Create and return a Groq client.
    Looks for API key in:
      1. Streamlit secrets: GROQ_API_KEY
      2. Environment variable: GROQ_API_KEY
    """
    api_key = None

    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    try:
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")
        return None


client = get_groq_client()

# The model must output exactly a JSON object containing:
#    - 'lat': Latitude in decimal degrees (WGS 84)
#    - 'lon': Longitude in decimal degrees (WGS 84)
#    - 'url': A link to OpenStreetMap in the format:
#      "https://www.openstreetmap.org/#map=14/<lat>/<lon>&layers=H"
system_message = {
    "role": "system",
    "content": (
        "You are a skilled geographer. When provided with a location query, "
        "respond only with a valid JSON object containing exactly three keys: "
        "'lat' (latitude in decimal degrees, WGS 84), "
        "'lon' (longitude in decimal degrees, WGS 84), and "
        "'url' which must be a valid OpenStreetMap link in the format "
        "'https://www.openstreetmap.org/#map=14/<lat>/<lon>&layers=H'. "
        "Do not include markdown, code fences, or any extra text."
    )
}


def get_coordinates(query):
    """
    Uses the Groq model to get coordinates and an OpenStreetMap URL from a location query.
    Returns a JSON string.
    """
    if client is None:
        return json.dumps({
            "error": "Groq client is not configured. Set GROQ_API_KEY in Streamlit secrets or environment variables."
        })

    try:
        messages = [
            system_message,
            {"role": "user", "content": query}
        ]

        response = client.chat.completions.create(
            messages=messages,
            model="moonshotai/kimi-k2-instruct-0905",
            temperature=0.0
        )

        reply = response.choices[0].message.content.strip()
        return reply

    except Exception as e:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] Error getting coordinates: {str(e)}")
        return json.dumps({"error": str(e)})


# === DATA MINING FUNCTIONS ===
def parse_line(line):
    num_part, sep, text = line.partition(":")
    line_number = num_part.strip()
    return line_number, text.strip()


def tokenize(text):
    tokens = re.findall(r"\b\w+\b", text)
    return tokens


def detect_language(text):
    try:
        lang = detect(text)
    except Exception:
        lang = "unknown"
    return lang


def lemmatize(token, lang_code="default"):
    return token.lower()


def extract_context(tokens, index, window_before=5, window_after=6):
    start = max(0, index - window_before)
    end = min(len(tokens), index + window_after + 1)
    context_tokens = tokens[start:end]
    return " ".join(context_tokens)


def process_text_file(input_filepath, output_csv_filepath):
    results = []
    with open(input_filepath, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            line_number, text = parse_line(line)
            lang_detected = detect_language(text)
            tokens = tokenize(text)
            for i, token in enumerate(tokens):
                lemma = lemmatize(token, lang_code=lang_detected)
                context = extract_context(tokens, i)
                results.append((lemma, line_number, context))

    with open(output_csv_filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["lemma", "line", "context"])
        writer.writerows(results)

    st.success(f"Processing complete. Output written to {output_csv_filepath}")


def search_lemma(search_term, csv_filepath):
    matches = []
    with open(csv_filepath, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["lemma"].lower() == search_term.lower():
                matches.append(row)
    return matches


def log_query(query, results, query_csv_filepath):
    try:
        with open(query_csv_filepath, "r", newline="", encoding="utf-8") as csvfile:
            pass
        header_exists = True
    except FileNotFoundError:
        header_exists = False

    with open(query_csv_filepath, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not header_exists:
            writer.writerow(["query", "lemma", "line", "context"])
        if results:
            for res in results:
                writer.writerow([query, res["lemma"], res["line"], res["context"]])
        else:
            writer.writerow([query, "", "", ""])


def plot_statistics_plotly(query_csv_filepath):
    try:
        results_df = pd.read_csv(query_csv_filepath)
    except FileNotFoundError:
        st.warning("No query log CSV file found. No graph will be produced.")
        return

    results_df = results_df[results_df["lemma"].fillna("").str.strip() != ""]
    if results_df.empty:
        st.warning("The query log CSV file is empty. No graph will be produced.")
        return

    lemma_stats = results_df["lemma"].value_counts().reset_index()
    lemma_stats.columns = ["Lemma", "Frequency"]

    fig_pie = px.pie(
        lemma_stats,
        names="Lemma",
        values="Frequency",
        title="Lemma Frequency Distribution",
        hole=0.3
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

    chapter_stats = results_df.groupby(["line", "lemma"]).size().reset_index(name="Mentions")
    fig_bar = px.bar(
        chapter_stats,
        x="line",
        y="Mentions",
        color="lemma",
        title="Chapter-wise Lemma Mentions",
        labels={"line": "Book/Chapter", "Mentions": "Mentions"}
    )
    fig_bar.update_layout(barmode="stack")
    st.plotly_chart(fig_bar, use_container_width=True)


# === MODEL LOADING ===
def load_model_seg_cache(model_path):
    return load_model_seg(model_path)


def load_model_rec_cache(model_path):
    return load_model_rec(model_path)


MODEL_SEG_DEFAULT = load_model_seg_cache(get_real_path("data/default/blla.mlmodel"))

hardcoded_models = [
    {
        "name": "catmus-medieval-160.mlmodel",
        "path": "models/catmus-medieval-160.mlmodel",
        "language": "Latin",
        "meta": "null"
    },
    {
        "name": "catmus-tiny.mlmodel",
        "path": "models/catmus-tiny.mlmodel",
        "language": "Latin",
        "meta": "null"
    },
    {
        "name": "e-NDP_V7.mlmodel",
        "path": "models/e-NDP_V7.mlmodel",
        "language": "French",
        "meta": "null"
    },
    {
        "name": "lectaurep_base.mlmodel",
        "path": "models/lectaurep_base.mlmodel",
        "language": "French",
        "meta": "null"
    },
    {
        "name": "german_handwriting.mlmodel",
        "path": "models/german_handwriting.mlmodel",
        "language": "German",
        "meta": "null"
    },
    {
        "name": "McCATMuS_nfd_nofix_V1.mlmodel",
        "path": "models/McCATMuS_nfd_nofix_V1.mlmodel",
        "language": "Multilang",
        "meta": "Chagué, A. (2024). McCATMuS - Transcription model for handwritten, printed and typewritten documents from the 16th century to the 21st century. Zenodo. https://doi.org/10.5281/zenodo.13788177"
    }
]


# === USER INTERFACE AND NAVIGATION ===
st.title("Lumina Multimodality")
st.markdown("Kraken-based OCR/HTR Tool, NLP Text Mining, and Geocoding")

st.sidebar.image("images/logo.png")
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Select Page", ["Transcription", "Data Mining", "Geolocation"])


# --- SHARED SIDEBAR FOR TRANSCRIPTION ---
if app_mode == "Transcription":
    st.sidebar.header("HTR Configuration")
    st.sidebar.markdown("---")

    recognition_model_file = st.sidebar.file_uploader(
        "Upload Recognition Model (.mlmodel)",
        type=["mlmodel"]
    )

    selected_model = st.sidebar.selectbox(
        "Or select a built-in recognition model",
        options=hardcoded_models,
        format_func=lambda model: f"{model['name']} ({model['language']}) ({model['meta']})"
    )

    if recognition_model_file is not None:
        write_temporary_model("tmp/model_rec_temp.mlmodel", recognition_model_file)
        model_rec_path = get_real_path("tmp/model_rec_temp.mlmodel")
        st.sidebar.success("Recognition model loaded from upload!")
    else:
        model_rec_path = get_real_path(selected_model["path"])
        st.sidebar.info(
            f"Using built-in model: {selected_model['name']} ({selected_model['language']})"
        )
        st.sidebar.info(f"{selected_model['meta']}")


# --- PAGE: Transcription ---
if app_mode == "Transcription":
    st.subheader("OCR/HTR Results")

    uploaded_files = st.file_uploader(
        "Upload your page images:",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if len(uploaded_files) > 5:
            st.warning("For now, only the first 5 pages will be processed.")
            uploaded_files = uploaded_files[:5]
    else:
        st.info("Please upload at least one page image.")

    if uploaded_files and st.button("🚀 Run Prediction"):
        with st.spinner("Loading models..."):
            model_rec = load_model_rec_cache(model_rec_path)
            model_seg = MODEL_SEG_DEFAULT
            st.success("✅ Models loaded!")

        for idx, file in enumerate(uploaded_files):
            with st.expander(f"Page {idx + 1} - Preview & Prediction"):
                image = open_image(file)
                st.image(image, width=300, caption=f"Page {idx + 1}")

                col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1])

                with col2:
                    st.markdown("## ✂️ Segmentation")
                    st.markdown("---")
                    with st.spinner("⚙️ Segmenting image..."):
                        baseline_seg = segment_image(image, model_seg)
                        baselines, boundaries = prepare_segments(baseline_seg)
                    fig1, fig2 = display_baselines(image, baselines, boundaries)
                    st.pyplot(fig1)

                with col3:
                    st.markdown("## ✍️ Text")
                    st.markdown("---")
                    with st.spinner("⚙️ Recognizing text..."):
                        pred = recognize_text(model_rec, image, baseline_seg)
                        lines = [record.prediction.strip() for record in pred]
                        lines_with_idx = [f"{i}: {line}" for i, line in enumerate(lines)]

                    edited_text = st.text_area(
                        label="",
                        value="\n".join(lines),
                        height=570,
                        key=f"editable_text_{idx}"
                    )

                with col4:
                    st.markdown("## ✂️ Segmentation (Index)")
                    st.markdown("---")
                    st.pyplot(fig2)

                with col5:
                    st.markdown("## ✏️ Text (Index)")
                    st.markdown("---")
                    indexed_text = "\n".join(lines_with_idx)
                    st.text_area(
                        label="",
                        value=indexed_text,
                        height=570,
                        key=f"indexed_text_{idx}"
                    )

                with col6:
                    st.markdown("## 🔎 Text (Image)")
                    st.markdown("---")
                    st.pyplot(display_baselines_with_text(image, baselines, lines))

                st.markdown("---")

                st.download_button(
                    "💾 Download Edited Transcription (this page)",
                    edited_text,
                    file_name=f"prediction_page_{idx + 1}_edited.txt"
                )

                st.download_button(
                    "💾 Download Transcription Index (this page)",
                    indexed_text,
                    file_name=f"prediction_page_{idx + 1}_index.txt"
                )


# --- PAGE: Data Mining ---
elif app_mode == "Data Mining":
    st.subheader("Data Mining")
    st.write("Upload a structured text file (formatted as 'line_number: text') for analysis.")

    uploaded_text_file = st.file_uploader("Upload Input Text File", type=["txt"])

    if uploaded_text_file is not None:
        input_txt = uploaded_text_file.getvalue().decode("utf-8")
        temp_input_path = "temp_input.txt"

        with open(temp_input_path, "w", encoding="utf-8") as f:
            f.write(input_txt)

        st.success("Input file uploaded successfully.")

        if st.button("Process Text File"):
            process_text_file(temp_input_path, "temp_output.csv")

        st.markdown("---")
        st.write("Search for a lemma in the processed data:")
        search_term = st.text_input("Enter a lemma to search:")

        if st.button("Search") and search_term:
            if os.path.exists("temp_output.csv"):
                results = search_lemma(search_term, "temp_output.csv")
                if results:
                    df_results = pd.DataFrame(results)
                    st.write("Search results:")
                    st.table(df_results)

                    csv_results = df_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Search Results as CSV",
                        csv_results,
                        file_name="search_results.csv",
                        mime="text/csv"
                    )
                    log_query(search_term, results, "temp_query.csv")
                else:
                    st.write(f"No matches found for lemma '{search_term}'.")
            else:
                st.warning("Please process the text file first.")

        st.markdown("---")

        if st.button("Plot Query Statistics"):
            if os.path.exists("temp_query.csv"):
                plot_statistics_plotly("temp_query.csv")
            else:
                st.warning("No query log found. Please perform a search first.")


# --- PAGE: Geolocation ---
elif app_mode == "Geolocation":
    st.subheader("Geolocation - Geographic Coordinate Finder")
    st.markdown(
        "Enter a location query to receive its geographic coordinates along with an OpenStreetMap link. "
        "You can also export your queries as a CSV file."
    )

    if client is None:
        st.warning(
            "Groq is not configured. Add GROQ_API_KEY to Streamlit secrets or environment variables "
            "to enable geolocation."
        )

    with st.form(key="coordinate_form", clear_on_submit=True):
        user_query = st.text_input(
            "Enter your location query:",
            placeholder="E.g., 'Eiffel Tower, Paris'"
        )
        submit_button = st.form_submit_button("Get Coordinates")

        if submit_button and user_query:
            result = get_coordinates(user_query)

            try:
                coords = json.loads(result)

                if "error" in coords:
                    st.error(f"Error: {coords['error']}")
                else:
                    lat = coords.get("lat")
                    lon = coords.get("lon")
                    osm_url = coords.get("url")

                    if lat is None or lon is None or osm_url is None:
                        st.error(
                            "The response did not contain valid 'lat', 'lon', and 'url' keys.\n\n"
                            f"Received: {result}"
                        )
                    else:
                        # Ensure numeric values
                        lat = float(lat)
                        lon = float(lon)

                        st.success(f"Coordinates: Latitude {lat}, Longitude {lon}")
                        st.markdown(f"[Open in OpenStreetMap]({osm_url})")

                        df_map = pd.DataFrame({"lat": [lat], "lon": [lon]})
                        st.map(df_map)

                        st.session_state.geolocation_history.append({
                            "query": user_query,
                            "coordinates": f"{lat}, {lon}",
                            "url": osm_url
                        })

            except Exception as e:
                st.error(f"An error occurred while parsing the response: {e}")
                st.code(result, language="json")

    # Display geolocation history if available.
    if st.session_state.geolocation_history:
        st.subheader("Geolocation Query History")
        df_geo_history = pd.DataFrame(st.session_state.geolocation_history)
        st.dataframe(df_geo_history)
        
        csv_geo = df_geo_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Export Query History as CSV",
            data=csv_geo,
            file_name="geolocation_query_history.csv",
            mime="text/csv"
        )
