# BiasScope

# 🧭 BiasScope — AI-Powered Media Bias & Sentiment Analyzer

**BiasScope** is an interactive **Streamlit web app** that analyzes **news articles** and **Reddit discussions** to detect **media bias**, **sentiment**, and **credibility**.  
It combines **real-time data scraping**, **AI summarization using Ollama**, and a **Machine Learning model** for a comprehensive analysis of public discourse.

---

## 🚀 Features

### 📰 News Analysis
- Fetches live news from **NewsAPI** based on topic and user preferences.
- Displays article **titles, sources, publication dates**, and **hyperlinked URLs**.
- Sort by **latest**, **popular**, or **relevant** stories.

### 💬 Reddit Analysis
- Collects posts from specific **subreddits** or topic keywords.
- Includes **post titles, subreddit names, upvotes, and comment scores**.
- Sort options: **Top**, **Hot**, or **Best** posts for more relevant insights.

### 🧠 AI Summarization
- Uses **Ollama’s Mistral model** to generate 500–1000 word structured summaries.
- Runs **fully offline** (local LLM), ensuring **privacy** and **speed**.
- Highlights **positive, negative, and neutral viewpoints** within articles.

### 📊 Machine Learning Sentiment Model
- Implements a **TF-IDF + Logistic Regression** classifier using **scikit-learn**.
- Trains on summarized news and Reddit text to predict **Positive**, **Negative**, or **Neutral** sentiment.
- Visualizes performance via **accuracy** and **confusion matrix**.

### 🧩 Bias Detection
- Merges ML predictions with summarization to indicate **source bias**.
- Detects opinion-heavy vs. factual reporting patterns.
- Enables comparison across media ecosystems.

### 🖥️ Smart Interface
- Built with **Streamlit**, styled via **custom CSS and fonts**.
- Clean, responsive design — centered input fields and modern typography.
- Sort and filter features for focused analysis.

---

## 🧠 How the Machine Learning Works

The project’s ML component uses **Logistic Regression**, chosen for its:
- Interpretability and transparency (ideal for bias detection).
- Efficiency on small-to-medium datasets.
- Ability to model binary/multiclass sentiment outcomes.

**Pipeline Overview:**
1. Extracts text summaries from news and Reddit content.
2. Converts text to numeric vectors using **TF-IDF**.
3. Splits data into **training** and **testing** sets.
4. Trains a **Logistic Regression model** to classify sentiment polarity.
5. Evaluates using accuracy, precision, recall, and confusion matrix.

---

## ⚙️ Tech Stack

| Component | Technology Used |
|------------|-----------------|
| **Frontend** | Streamlit, HTML, CSS |
| **Backend** | Python, APIs (Reddit, NewsAPI), Ollama |
| **ML** | scikit-learn, pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **AI Model** | Ollama Mistral (local LLM) |

---

## 🧱 Project Workflow

1. **Input a topic** — e.g., *Artificial Intelligence*, *Elections*, *Climate Change*.
2. **BiasScope** fetches articles and Reddit discussions in real-time.
3. **Ollama (Mistral)** summarizes each piece into structured, detailed summaries.
4. The **ML sentiment model** classifies the tone of each summary.
5. Results are displayed interactively in the web UI with sorting and filtering options.

---

## 📊 Example Outputs

| Source | Title | Sentiment | Bias Indicator |
|:--|:--|:--|:--|
| The Verge | "AI Regulation in 2025: What’s Next?" | Neutral | Balanced |
| Reddit (r/technology) | "OpenAI’s new model just dropped!" | Positive | User-driven |
| NY Times | "Concerns rise over AI unemployment" | Negative | Slight Bias |

---

## 🧩 Future Enhancements

- Integrate **transformer-based models (DistilBERT or RoBERTa)** for advanced sentiment detection.  
- Add **credibility scoring** using known media reliability datasets.  
- Implement **historical bias tracking** for evolving topics.  
- Extend to **Twitter/X API** for wider social media sentiment coverage.

---

## 🧑‍💻 Installation & Setup

### Prerequisites
- Python 3.9+
- Ollama installed and model pulled (e.g. `ollama pull mistral`)
- NewsAPI key (free at [newsapi.org](https://newsapi.org))

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<yourusername>/BiasScope.git
cd BiasScope
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Start Ollama Server
```bash
ollama serve
```
### 4️⃣ Run the App
```bash
streamlit run app.py
```
