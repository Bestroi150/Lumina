<table>
  <tr>
    <td>
      <img 
        src="https://github.com/Bestroi150/Lumina/blob/main/images/logo.png" 
        alt="lumina logo" 
        width="240"
        width="400"
      />
    </td>
    <td>
      <h1 style="margin:0;">Lumina Multimodality</h1>
      <p style="margin:0;">
        A <strong>Streamlit web application</strong> that integrates 
        <strong>OCR/HTR transcription, NLP text mining, and AI-powered geolocation</strong> 
        into a single interface.The application is built on the <strong>Kraken OCR engine</strong> and uses <strong>Groq LLMs</strong> for AI-assisted geolocation.
      </p>
    </td>
  </tr>
</table>

# Lumina Multimodality

A Streamlit web application that combines OCR/HTR (Handwriting Text Recognition), NLP text mining, Named Entity Recognition, and AI-powered geolocation into a single five-step pipeline. Built on top of the [Kraken](https://kraken.re) engine, spaCy NER, and Groq LLMs.

---

## Features

### 📖 Step 1 — Transcription (OCR/HTR)
- Upload one or more manuscript/document page images (`.jpg`, `.jpeg`, `.png`, up to 5 pages)
- Automatic **baseline segmentation** using the built-in BLLA model (native 1800 px resolution)
- **Text recognition** using a selection of pre-trained HTR models (or upload your own `.mlmodel`)
- Side-by-side view: original image, segmentation map, recognized text, indexed text, and text overlaid on image
- Editable transcription text area per page
- Download individual page transcriptions or a **combined all-pages file**

### 🔍 Step 2 — Data Mining
- Automatic **tokenization**, **language detection**, and **lemmatization** of the transcribed text
- Upload a separate `.txt` file to override the transcription output
- Search for a lemma and view all occurrences with context
- Export search results as CSV
- Interactive **frequency distribution pie chart** and **chapter-wise bar chart** (Plotly)

### 🏷️ Step 3 — Named Entity Recognition (NER)
- Extract **place names and geographic entities** (`LOC`, `GPE`, `MISC`) from the text
- Powered by **spaCy `fr_core_news_lg`** (French large model with word vectors)
- Detected locations are forwarded automatically to the Geolocation step
- Exportable entity table and unique-place chips view

### 🌍 Step 4 — Geolocation
- AI-powered coordinate resolution via the **Groq API** (returns lat/lon + OpenStreetMap link)
- Auto-geocode all NER-detected places, or enter a manual location query
- Interactive map rendered directly in the app
- Persistent session history with CSV export and one-click clear

### 📦 Step 5 — Export
- Combined download of all pipeline outputs (transcription, tokens, NER entities, geolocation history)

---

## Built-in HTR Models

| Model | Language | Notes |
|---|---|---|
| `catmus-medieval-160.mlmodel` | Latin | CATMuS medieval prints |
| `catmus-tiny.mlmodel` | Latin | CATMuS tiny (fast) |
| `e-NDP_V7.mlmodel` | French | Notre-Dame de Paris chapter registers (1326–1504) |
| `lectaurep_base.mlmodel` | French | Administrative handwriting (19th–20th c.) |
| `german_handwriting.mlmodel` | German | German handwriting |
| `McCATMuS_nfd_nofix_V1.mlmodel` | Multilingual | Handwritten, printed & typewritten docs, 16th–21st c. |

You can also upload a custom `.mlmodel` file via the sidebar.

---

## Limitations

- **CPU-only inference.** Kraken segmentation and recognition run on CPU; processing a single page can take tens of seconds depending on image size and model complexity. GPU (CUDA) is not currently used.
- **NER accuracy.** The spaCy `fr_core_news_lg` model is a general-purpose French model. It may miss locations embedded in named expressions (e.g. "Belgrade" inside "Traité de Belgrade" may be tagged as `MISC` rather than `LOC`). Entities of type `MISC` are included to mitigate this, but some noise is expected.
- **French-only NER.** The NER step uses a French-language spaCy model. For other languages, a different model would need to be loaded.
- **Geolocation requires a Groq API key.** Without it, Steps 1–3 work normally but Step 4 is unavailable.
- **5-page limit.** Only the first 5 uploaded images are processed per session to keep processing times manageable.
- **Recognition models are not retrained.** The bundled `.mlmodel` files were trained with an older Kraken version (legacy polygon extractor). The app forces the new polygon extractor at load time, which improves speed but may marginally affect accuracy on some models.

---

## Installation

### Prerequisites
- Python **3.12** (recommended; the `.python-version` file pins this for Streamlit Cloud)
- A [Groq API key](https://console.groq.com/keys) (required for the Geolocation step)

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Lumina.git
cd Lumina
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download fr_core_news_lg
```

### 4. Configure your API key

Create a `.env` file in the project root:
```
GROQ_API=your_groq_api_key_here
```

> The app also accepts the `GROQ_API_KEY` environment variable as a fallback.  
> Without a key the Transcription, Data Mining, and NER steps work normally; only the Geolocation step requires it.

---

## Running the App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project Structure

```
Lumina/
├── app.py                   # Main Streamlit application
├── requirements.txt
├── .env                     # API keys (not committed)
├── data/
│   └── default/
│       └── blla.mlmodel     # Default segmentation model
├── models/                  # Built-in HTR recognition models
│   ├── catmus-medieval-160.mlmodel
│   ├── catmus-tiny.mlmodel
│   ├── e-NDP_V7.mlmodel
│   ├── lectaurep_base.mlmodel
│   ├── german_handwriting.mlmodel
│   └── McCATMuS_nfd_nofix_V1.mlmodel
├── images/
│   └── logo.png
└── lib/
    ├── __init__.py
    ├── constants.py         # Model metadata and config
    ├── display_utils.py     # Matplotlib baseline visualization
    └── kraken_utils.py      # Kraken segmentation & recognition wrappers
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `kraken` | OCR/HTR engine (segmentation + recognition) |
| `groq` | LLM API client for geolocation |
| `pandas` / `numpy` | Data processing |
| `plotly` | Interactive charts |
| `langdetect` | Language detection for text mining |
| `matplotlib` | Baseline visualization overlays |
| `python-dotenv` | `.env` file support |

---

## Security Notes

- **Never commit your `.env` file or API keys** to version control. Add `.env` to your `.gitignore`.
- The Groq API key is loaded lazily — the app starts normally even without it.
