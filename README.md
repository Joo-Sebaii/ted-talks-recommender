# 🎤 TED Talks Recommendation System

**Content-based recommender** that suggests relevant TED Talks based on your interests using **TF-IDF + Cosine Similarity + Pearson Correlation**.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Features

- Clean modular project structure (not a single Jupyter notebook)
- Text preprocessing (stopwords removal + punctuation cleaning)
- TF-IDF vectorization
- Dual similarity scoring:
  - **Cosine similarity** (angle-based)
  - **Pearson correlation** (linear relationship)
- Combined ranking of top-N most relevant talks
- Easy to extend (add new similarity metrics, evaluation, UI, etc.)

## 🖼️ Example Output

Query:  
`"Climate change and impact on health and carbon footprint"`

Recommended talks (example):

| main_speaker | details |
|--------------|--------|
| Al Gore | ... climate change health impacts carbon emissions ... |
| Johan Rockström | ... planetary boundaries climate health connection ... |
| Christiana Figueres | ... Paris agreement health co-benefits ... |

## 📂 Project Structure
```
ted-talks-recommender/
├── data/
│   └── tedx_dataset.csv               # original dataset
├── src/
│   ├── init.py
│   ├── preprocessing.py               # data loading & text cleaning
│   ├── model.py                       # TF-IDF + similarity computation
│   └── utils.py                       # helper functions
├── recommend.py                       # main recommendation script
├── README.md
└── requirements.txt                   (recommended)
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- pandas, scikit-learn, nltk, scipy

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/ted-talks-recommender.git
cd ted-talks-recommender

# (Recommended) Create virtual environment
python -m venv venv
source venv/bin/activate    # Linux / macOS
venv\Scripts\activate       # Windows

# Install dependencies
pip install pandas scikit-learn nltk scipy
```

### 3. Download NLTK data (only needed once)
```python
import nltk
nltk.download('stopwords')
```

### 4. Run the recommender
```bash
python recommend.py
```
Or modify the query directly in `recommend.py`:
```python
query = [
    "Climate change and impact on health and carbon footprint",
    # "machine learning ethics bias fairness",
    # "future of education technology children",
]
```

## 🛠️ How It Works (Pipeline)

**Load & Preprocess**
- Read CSV → keep main_speaker, title, details
- Merge title + details
- Lowercase → remove stopwords → remove punctuation

**Vectorization**
- TfidfVectorizer on cleaned details column

**Similarity Calculation**
- Transform user query to TF-IDF vector
- Compute cosine similarity for each document
- Compute Pearson correlation for each document
- Sort primarily by cosine, secondarily by Pearson
- Return top-N results

## 🔧 Possible Improvements (Future Work)

- Add more preprocessing (lemmatization, stemming)
- Try sentence transformers / BERT embeddings
- Add evaluation metrics (precision@K, NDCG, user study)
- Create a simple Streamlit / Gradio web interface
- Support multi-query / query expansion
- Add speaker / event / year filters

## 📄 License
MIT License  
Feel free to use this code for learning, personal projects, or portfolios.  
Made with ❤️ by Youssef Mohammed  
Happy recommending! 🎬

