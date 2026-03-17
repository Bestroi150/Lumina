<table>
  <tr>
    <td>
      <img 
        src="https://github.com/Bestroi150/Lumina/blob/main/images/logo.png" 
        alt="lumina logo" 
        width="240"
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

---

## Overview

Lumina provides a unified workflow for working with historical and textual data:

* **Transcribe manuscript images** using OCR/HTR models
* **Analyze extracted text** using NLP tools
* **Locate geographic references** using AI-based coordinate resolution

The goal is to simplify the pipeline from **image → text → analysis → location**.

---

# Features

## 📖 Transcription (OCR / HTR)

* Upload one or more manuscript or document images (`.jpg`, `.jpeg`, `.png`)
* Automatic **baseline segmentation** using the built-in **BLLA model**
* **Text recognition** using selectable pre-trained HTR models
* **Side-by-side visualization** including:

  * original image
  * segmentation map
  * recognized text
  * indexed text
  * text overlay on the image
* Editable transcription area for each page
* Export:

  * individual page transcriptions
  * combined transcription for all pages

---

## 🔍 Data Mining

* Upload a structured text file in the format:

```
line_number: text content
```

Features include:

* Automatic **tokenization**
* **Language detection**
* **Lemmatization**
* Search by **lemma**
* Display **occurrences with context**
* Export search results as **CSV**

Visualizations:

* **Word frequency pie chart**
* **Chapter-wise distribution bar chart**

(All charts rendered with Plotly)

---

## 🌍 Geolocation

Resolve geographic references using an AI model.

Features:

* Enter a location query (example: *"Notre-Dame de Paris"*)
* AI resolves **latitude and longitude**
* Direct **OpenStreetMap link**
* Interactive map inside the app
* Session history:

  * export results as CSV
  * clear results with one click

---

# Built-in HTR Models

| Model                           | Language     | Description                                                               |
| ------------------------------- | ------------ | ------------------------------------------------------------------------- |
| `catmus-medieval-160.mlmodel`   | Latin        | CATMuS medieval prints                                                    |
| `catmus-tiny.mlmodel`           | Latin        | Lightweight CATMuS model (fast inference)                                 |
| `e-NDP_V7.mlmodel`              | French       | Notre-Dame de Paris chapter registers (1326–1504)                         |
| `lectaurep_base.mlmodel`        | French       | Administrative handwriting (19th–20th century)                            |
| `german_handwriting.mlmodel`    | German       | Historical German handwriting                                             |
| `McCATMuS_nfd_nofix_V1.mlmodel` | Multilingual | Mixed handwritten, printed, and typewritten documents (16th–21st century) |

You may also **upload custom `.mlmodel` files** via the sidebar.

---

# Installation

## Prerequisites

* Python **3.10 – 3.12**
* A **Groq API key**

Create one here:

[https://console.groq.com/keys](https://console.groq.com/keys)

(The API key is only required for the **Geolocation** page.)

---

## 1. Clone the repository

```bash
git clone https://github.com/your-username/Lumina.git
cd Lumina
```

---

## 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure your API key

Create a `.env` file in the project root:

```
GROQ_API=your_groq_api_key_here
```

The application also accepts the fallback variable:

```
GROQ_API_KEY
```

If no key is provided:

* **Transcription** and **Data Mining** still work
* **Geolocation** will be disabled

---

# Running the Application

Start the Streamlit server:

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

# Project Structure

```
Lumina/
│
├── app.py                   # Main Streamlit application
├── requirements.txt
│
├── data/
│   └── default/
│       └── blla.mlmodel     # Default segmentation model
│
├── models/                  # Built-in HTR recognition models
│   ├── catmus-medieval-160.mlmodel
│   ├── catmus-tiny.mlmodel
│   ├── e-NDP_V7.mlmodel
│   ├── lectaurep_base.mlmodel
│   ├── german_handwriting.mlmodel
│   └── McCATMuS_nfd_nofix_V1.mlmodel
│
├── images/
│   └── logo.png
│
└── lib/
    ├── __init__.py
    ├── constants.py         # Model metadata and configuration
    ├── display_utils.py     # Matplotlib baseline visualization
    └── kraken_utils.py      # Kraken segmentation & recognition wrappers
```

---

# Dependencies

| Package            | Purpose                      |
| ------------------ | ---------------------------- |
| `streamlit`        | Web application framework    |
| `kraken`           | OCR/HTR engine               |
| `groq`             | LLM API client               |
| `pandas` / `numpy` | Data processing              |
| `plotly`           | Interactive visualizations   |
| `langdetect`       | Language detection           |
| `matplotlib`       | Segmentation visualization   |
| `python-dotenv`    | `.env` configuration support |

---

# License

This project is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) License. See [LICENSE](https://github.com/Bestroi150/Lumina/blob/main/LICENSE) file for details.
