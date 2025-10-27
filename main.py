import streamlit as st
from streamlit.components.v1 import html as st_html
import pandas as pd
import os
import re
import subprocess
import time
import praw
import prawcore
import requests
from collections import Counter
from transformers import pipeline
import matplotlib.pyplot as plt
import base64
from pathlib import Path
import time
import socket
from dotenv import load_dotenv
load_dotenv()


# ML imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


# ----------------------
# Config (move to env or config file for production)
# ----------------------
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
OLLAMA_MODEL = "mistral"
HF_SUMMARIZER_MODEL = os.getenv("HF_SUMMARIZER_MODEL") or "facebook/bart-large-cnn"
REDDIT_LIMIT = 15
NEWSAPI_PAGE_SIZE = 25

# ----------------------
# Helper utilities
# ----------------------

def slugify(text: str) -> str:
    s = re.sub(r'[^A-Za-z0-9]+', '_', text.strip().lower())
    return s.strip('_')[:60]


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"[^A-Za-z0-9\s\.,!\?:;\-']+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_ollama_running(host="127.0.0.1", port=11434):
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False
    
# Start Ollama only if not running (prevents multiple servers on Streamlit refresh)
if not is_ollama_running():
    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # short sleep to let server initialize; keep small to avoid long blocking
        time.sleep(5)
    except Exception as e:
        # don't crash app; just warn user
        st.warning(f"Could not start ollama serve automatically: {e}")

# ----------------------
# Reddit scraping (robust with retries)
# ----------------------

def scrape_reddit(query):
    """Scrape Reddit for posts related to the query"""
    
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT
    )

    posts = []
    try:
        subreddit = reddit.subreddit(query)
        for post in subreddit.top(limit=10):
            posts.append({
                "title": post.title,
                "text": post.selftext,
                "url": post.url,
                "score": post.score,
                "comments": post.num_comments
            })
    except Exception as e:
        st.warning(f"⚠️ Subreddit fetch failed for '{query}', falling back to r/all. Error: {e}")
        try:
            subreddit = reddit.subreddit("all")
            for post in subreddit.search(query, limit=10):
                posts.append({
                    "title": post.title,
                    "text": post.selftext,
                    "url": post.url,
                    "score": post.score,
                    "comments": post.num_comments
                })
        except Exception as e2:
            st.error(f"⚠️ Fallback search also failed: {e2}")

    if not posts:
        st.warning("⚠️ No Reddit posts found.")
        return pd.DataFrame(columns=["title", "text", "url", "score", "comments", "raw_text"])

    df = pd.DataFrame(posts)
    df["raw_text"] = df["title"].fillna("") + ". " + df["text"].fillna("")
    return df


# ----------------------
# NewsAPI scraper
# ----------------------
BASE_URL = "https://newsapi.org/v2/everything"

def scrape_news(query: str, sort_by: str = "publishedAt", domains: str = None, sources: str = None):
    st.info(f"Starting NewsAPI scrape for: {query}")
    if NEWSAPI_KEY in (None, "", "YOUR_NEWSAPI_KEY"):
        st.error("NewsAPI key not set")
        return pd.DataFrame([])

    params = {
        "q": query,
        "apiKey": NEWSAPI_KEY,
        "language": "en",
        "sortBy": sort_by,
        "pageSize": NEWSAPI_PAGE_SIZE
    }
    if domains:
        params["domains"] = domains
    if sources:
        params["sources"] = sources

    try:
        r = requests.get(BASE_URL, params=params, timeout=30)
        r.raise_for_status()
    except Exception as e:
        st.error(f"NewsAPI request failed: {e}")
        return pd.DataFrame([])

    data = r.json()
    if data.get("status") != "ok":
        st.error(f"NewsAPI error: {data.get('message', 'unknown')}")
        return pd.DataFrame([])

    rows = []
    for a in data.get("articles", []):
        rows.append({
            "source": a.get("source", {}).get("name", ""),
            "author": a.get("author", ""),
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "url": a.get("url", ""),
            "publishedAt": a.get("publishedAt", ""),
            "content": a.get("content") or ""
        })
    return pd.DataFrame(rows)


# ----------------------
# Ollama summarization (subprocess)
# ----------------------

import subprocess, time, streamlit as st

def generate_ollama_summary(query: str, model: str = "mistral", timeout: int = 300):
    """
    Generate a structured, pointwise summary for a topic using Ollama.
    Shows a loading spinner while generating and displays only final clean text.
    """

    prompt = f"""
    Generate a 500–1000 word structured, pointwise and detailed summary for the topic: {query}.
    Include positive, negative and neutral viewpoints. 
    Use short paragraphs or bullet points for clarity.
    """

    try:
        st.info("🧠 Generating AI summary using Ollama... please wait.")
        output_placeholder = st.empty()

        with st.spinner("Thinking... this may take 10–30 seconds depending on your model."):
            start = time.time()
            process = subprocess.Popen(
                ["ollama", "run", model],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line-buffered output
            )

            process.stdin.write(prompt)
            process.stdin.close()

            output_lines = []
            # Collect streamed output
            for line in iter(process.stdout.readline, ''):
                output_lines.append(line)
                # Optional: live preview every few seconds
                if len(output_lines) % 10 == 0:
                    output_placeholder.text("⏳ Generating summary...")

                if time.time() - start > timeout:
                    process.kill()
                    st.warning("⚠️ Ollama generation timed out after 300 seconds.")
                    break

            process.wait()
            stderr = process.stderr.read().strip()
            if process.returncode != 0:
                # st.warning(f"Ollama returned error ({process.returncode}): {stderr}")
                pass

        summary = "".join(output_lines).strip()
        if not summary:
            summary = stderr or "⚠️ No output received from Ollama."

        # Clean display
        output_placeholder.empty()
        # st.markdown("### 🧾 AI Summary")
        # st.markdown(f"<div style='background-color:#111;padding:16px;border-radius:10px;color:#e0e0e0;'>{summary}</div>", unsafe_allow_html=True)

        return summary

    except Exception as e:
        st.error(f"❌ Ollama call failed: {e}")
        return f"Error: {e}"



# ----------------------
# HuggingFace summarizer loader + summarization helper
# ----------------------

def torch_available():
    try:
        import torch
        return True
    except Exception:
        return False


def load_hf_summarizer(model_name=HF_SUMMARIZER_MODEL):
    try:
        device = 0 if torch_available() else -1
        summarizer = pipeline("summarization", model=model_name, device=device)
        return summarizer
    except Exception as e:
        st.warning(f"Could not load HF summarizer: {e}")
        return None


def summarize_texts(texts, summarizer, min_words=30):
    out = []
    if summarizer is None:
        return texts
    for t in texts:
        cleaned = clean_text(t)
        if len(cleaned.split()) < min_words:
            out.append(cleaned)
            continue
        try:
            s = summarizer(cleaned[:1500], max_length=250, min_length=40, do_sample=False)
            out.append(s[0]["summary_text"])
        except Exception:
            out.append(cleaned)
    return out


# ----------------------
# Simple descriptive-word extractor (no spaCy required)
# ----------------------
STOPWORDS = set(["the","and","a","an","in","on","for","is","it","of","to","this","that","with","as","are","was","be","by","from","at"])
ADJ_SUFFIXES = ("ive","ous","ful","less","able","ible","al","ic","ant","ent","ary","ish","like")

def get_top_descriptive_words(texts, top_n=12):
    cnt = Counter()
    for t in texts:
        t_clean = re.sub(r"[^A-Za-z0-9\s]", " ", t.lower())
        for w in t_clean.split():
            if w in STOPWORDS or len(w) < 4:
                continue
            if w.endswith(ADJ_SUFFIXES):
                cnt[w] += 1
    return cnt.most_common(top_n)


# ----------------------
# Sentiment (HF) and bias score
# ----------------------

def add_sentiment(df, text_col="ai_summary", sentiment_pipe=None):
    if sentiment_pipe is None:
        try:
            sentiment_pipe = pipeline("sentiment-analysis")
        except Exception as e:
            st.warning(f"Could not load sentiment model: {e}")
            df['sentiment'] = 'NEUTRAL'
            df['sentiment_score'] = 0.0
            df['bias_score'] = 0
            return df

    labels = []
    scores = []
    for t in df[text_col].fillna(""):
        if not t.strip():
            labels.append("NEUTRAL")
            scores.append(0.0)
            continue
        try:
            out = sentiment_pipe(t[:512])[0]
            lab = out.get('label', 'NEUTRAL')
            labels.append(lab)
            scores.append(out.get('score', 0.0))
        except Exception:
            labels.append("NEUTRAL")
            scores.append(0.0)
    
    df['sentiment'] = labels
    df['sentiment_score'] = scores
    
    # Handle various label formats from different models
    mapping = {
        "POSITIVE": 1, "NEGATIVE": -1, "NEUTRAL": 0,
        "LABEL_2": 1, "LABEL_0": -1, "LABEL_1": 0,
        "POS": 1, "NEG": -1
    }
    df['bias_score'] = df['sentiment'].map(mapping).fillna(0)
    return df


# ----------------------
# Streamlit UI + Orchestration
# ----------------------

def render_background():
    """Renders animated dot grid background that covers entire page"""
    
    # Complete HTML with inline CSS and JavaScript
    background_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            html, body {
                width: 100%;
                height: 100%;
                overflow: hidden;
            }
            
            #dots {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                display: block;
                background: #000000;
            }
        </style>
    </head>
    <body>
        <canvas id="dots"></canvas>
        
        <script>
        (function(){
          const canvas = document.getElementById('dots');
          if (!canvas) return;
          
          const ctx = canvas.getContext('2d');
          let dpr = window.devicePixelRatio || 1;
          let w, h;
          const spacing = 36;
          const dotR = 2;
          const floatAmp = 3;
          const floatSpeed = 0.005;
          let dots = [];
          let animationId;

          function resize(){
            w = window.innerWidth;
            h = window.innerHeight;
            canvas.width = w * dpr;
            canvas.height = h * dpr;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            initDots();
          }

          function initDots(){
            dots = [];
            for (let x = spacing/2; x < w; x += spacing) {
              for (let y = spacing/2; y < h; y += spacing) {
                dots.push({
                  x: x + (Math.random() * 6 - 3),
                  y: y + (Math.random() * 6 - 3),
                  phase: Math.random() * Math.PI * 2,
                  r: dotR * (0.7 + Math.random() * 0.8)
                });
              }
            }
          }

          function draw(t){
            ctx.clearRect(0, 0, w, h);
            ctx.fillStyle = '#2f2f2f';
            for (const d of dots) {
              const floatY = Math.sin(d.phase + t * floatSpeed) * floatAmp;
              ctx.beginPath();
              ctx.arc(d.x, d.y + floatY, d.r, 0, Math.PI * 2);
              ctx.fill();
            }
            animationId = requestAnimationFrame(draw);
          }

          window.addEventListener('resize', resize);
          resize();
          requestAnimationFrame(draw);
        })();
        </script>
    </body>
    </html>
    """
    
    # Inject CSS to style Streamlit's containers
    st.markdown("""
        <style>
        /* Remove Streamlit's default padding and make background transparent */
        .main > div {
            padding-top: 2rem;
        }
        
        .stApp {
            background: transparent;
        }
        
        .main {
            background: transparent;
        }
        
        /* Position the iframe background */
        iframe[title="render_background.render_background"] {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            z-index: -1 !important;
            border: none !important;
            pointer-events: none !important;
        }
        
        /* Make content visible on dark background */
        .stTextInput label, .stMarkdown, p, h1, h2, h3 {
            color: #f0f0f0 !important;
        }
        
        /* Style the text input */
        .stTextInput input {
            background-color: rgba(47, 47, 47, 0.8) !important;
            color: #f0f0f0 !important;
            border: 1px solid #444 !important;
        }
        
        /* Button styling */
        .stButton button {
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: #f0f0f0 !important;
            border: 1px solid #666 !important;
        }
        
        .stButton button:hover {
            background-color: rgba(255, 255, 255, 0.2) !important;
            border-color: #888 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Render the background with proper height
    st.components.v1.html(background_html, height=0, scrolling=False)


def run_biasscope_pipeline(topic: str):
    # This function ties together scraping, summarization and local HF summarization
    slug = slugify(topic)

    # 1) Reddit
    reddit_df = scrape_reddit(topic)
    # FIXED: Changed 'body' to 'text' to match the actual column name
    reddit_df['raw_text'] = (reddit_df.get('title','').fillna('') + '. ' + reddit_df.get('text','').fillna('')).astype(str)
    reddit_df['ai_summary'] = reddit_df['raw_text'].apply(lambda x: clean_text(x))

    # 2) News
    news_df = scrape_news(topic)
    if isinstance(news_df, pd.DataFrame):
        if 'title' in news_df.columns and 'description' in news_df.columns:
            news_df['raw_text'] = news_df['title'].fillna('') + '. ' + news_df['description'].fillna('')
        else:
            st.warning("⚠️ Missing columns in news data — skipping text merge.")
            news_df['raw_text'] = ''
    else:
        st.error("❌ News scraping failed — invalid or empty data.")
        news_df = pd.DataFrame(columns=['title', 'description', 'raw_text'])

    news_df['ai_summary'] = news_df['raw_text'].apply(lambda x: clean_text(x))

    # 3) Ollama paragraph
    ollama_text = generate_ollama_summary(topic)
    st.write("Generating AI summary... this may take a minute ⏳")
    print(">>> AI Summary Raw Output:", ollama_text[:500])
    ai_df = pd.DataFrame([{"query": topic, "ai_summary": ollama_text}])

    # 4) Load HF summarizer (optional, can be slow)
    summarizer = None
    try:
        summarizer = load_hf_summarizer(HF_SUMMARIZER_MODEL)
    except Exception as e:
        st.warning(f"HF summarizer unavailable: {e}")

    # 5) Summarize rows (if summarizer loaded)
    if summarizer is not None:
        reddit_df['ai_summary'] = summarize_texts(reddit_df['raw_text'].tolist(), summarizer)
        news_df['ai_summary'] = summarize_texts(news_df['raw_text'].tolist(), summarizer)

    # 6) Save intermediate CSVs
    reddit_df.to_csv('reddit_cleaned.csv', index=False, encoding='utf-8')
    news_df.to_csv('news_cleaned.csv', index=False, encoding='utf-8')
    ai_df.to_csv('ai_summary.csv', index=False, encoding='utf-8')

    return reddit_df, news_df, ai_df


def bias_analysis_and_plots(reddit_df, news_df, ai_df):
    # Add sentiment and bias score
    sentiment_pipe = None
    try:
        sentiment_pipe = pipeline('sentiment-analysis')
    except Exception as e:
        st.warning(f"Sentiment pipeline could not be loaded: {e}")

    reddit_df = add_sentiment(reddit_df, 'ai_summary', sentiment_pipe)
    news_df = add_sentiment(news_df, 'ai_summary', sentiment_pipe)
    ai_df = add_sentiment(ai_df, 'ai_summary', sentiment_pipe)

    # Sentiment counts
    datasets = {'Reddit': reddit_df, 'News': news_df, 'AI': ai_df}
    labels = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']
    counts = {k: [ (d['sentiment']==lab).sum() for lab in labels ] for k,d in datasets.items() }

    # Plot 1: grouped bar chart
    fig1, ax1 = plt.subplots(figsize=(8,4))
    x = range(len(datasets))
    width = 0.2
    ax1.bar([i-width for i in x], [counts[k][0] for k in datasets.keys()], width=width, label='Positive')
    ax1.bar(x, [counts[k][1] for k in datasets.keys()], width=width, label='Neutral')
    ax1.bar([i+width for i in x], [counts[k][2] for k in datasets.keys()], width=width, label='Negative')
    ax1.set_xticks(x)
    ax1.set_xticklabels(list(datasets.keys()))
    ax1.set_ylabel('Count')
    ax1.set_title('Sentiment comparison')
    ax1.legend()

    # Plot 2: top descriptive words per dataset
    top_words = {name: get_top_descriptive_words(df['ai_summary'].fillna('').tolist()) for name,df in datasets.items()}

    return fig1, top_words

# ----------------------
# Machine Learning Model Training (Sentiment Classification)
# ----------------------
def train_sentiment_model(combined_df):
    st.markdown("## 🧠 Machine Learning Model — Sentiment Classification")
    
    # Combine all text data and ensure it's clean
    combined_df = combined_df.dropna(subset=['ai_summary', 'sentiment'])
    combined_df = combined_df[combined_df['sentiment'].isin(['POSITIVE', 'NEGATIVE', 'NEUTRAL'])]

    if combined_df.empty:
        st.warning("No valid data available to train the model.")
        return None

    X = combined_df['ai_summary']
    y = combined_df['sentiment']

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_vec = vectorizer.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3, random_state=42)

    # Logistic Regression Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy and Confusion Matrix
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=['POSITIVE', 'NEUTRAL', 'NEGATIVE'])

    st.write(f"**Model Accuracy:** {acc*100:.2f}%")

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['POSITIVE', 'NEUTRAL', 'NEGATIVE'], yticklabels=['POSITIVE', 'NEUTRAL', 'NEGATIVE'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

    st.success("✅ Model trained successfully on the scraped and summarized data!")

    # Predict sentiment for AI summary
    if 'ai_summary' in combined_df.columns:
        st.markdown("### 🔮 Model Predictions on AI Summaries")
        X_ai = vectorizer.transform(combined_df['ai_summary'])
        preds = model.predict(X_ai)
        combined_df['model_predicted_sentiment'] = preds
        st.dataframe(combined_df[['ai_summary', 'sentiment', 'model_predicted_sentiment']].head(10))

    return model, vectorizer



# ----------------------
# App layout
# ----------------------

st.set_page_config(page_title="BiasScope", layout="wide", initial_sidebar_state="collapsed")

# Call render_background FIRST
render_background()

# Update title styling
st.markdown("""
    <style>
      .app-title {
        font-family: 'Courier New', monospace;
        color: #f0f0f0;
        text-align: center;
        font-size: 52px;
        margin-top: 60px;
        margin-bottom: 40px;
        text-shadow: 0 0 20px rgba(255,255,255,0.15);
      }
      .sub {
        color: #aaa;
        text-align: center;
      }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="app-title">BiasScope</div>', unsafe_allow_html=True)


with st.container():
    st.write("")
    c1, c2, c3 = st.columns([1,4,1])
    with c2:
        topic = st.text_input("Enter a topic to analyze", value="Your text here....")
        col_run = st.columns([1,1,1])
        run_btn = st.button("Run Analysis")
        st.caption("Note: Scraping and summarization may take time. Ollama and HF models must be available locally.")

if run_btn and topic.strip():
    with st.spinner("Running BiasScope pipeline — scraping, summarizing and analyzing..."):
        reddit_df, news_df, ai_df = run_biasscope_pipeline(topic.strip())

    # Show three columns with top summaries
    st.markdown("---")
    a, b, c = st.columns(3)

    # ---------- Reddit Column ----------
    with a:
        st.subheader("Reddit")
        if reddit_df.empty:
            st.info("No Reddit posts fetched")
        else:
            for i, row in reddit_df.head(5).iterrows():
                title = row.get('title', '')[:120]
                url = row.get('url', '#')
                summary = row.get('ai_summary','')[:300]
                score = row.get('score', 0)
                subreddit = row.get('subreddit', 'unknown')

                st.markdown(
                    f"""
                    <div style="background-color:#1e1e1e;padding:12px;border-radius:8px;margin-bottom:10px;">
                        <a href="{url}" target="_blank" style="font-size:16px;color:#61dafb;font-weight:bold;text-decoration:none;">
                            {title}
                        </a>
                        <p style="margin:4px 0;font-size:12px;color:#9a9a9a;">
                            Subreddit: r/{subreddit} | Score: {score}
                        </p>
                        <p style="margin-top:6px;color:#cfcfcf;font-size:14px;">{summary}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ---------- News Column ----------
    with b:
        st.subheader("News")
        if news_df.empty:
            st.info("No news fetched")
        else:
            for i, row in news_df.head(5).iterrows():
                title = row.get('title', '')[:120]
                url = row.get('url', '#')
                summary = row.get('ai_summary','')[:300]
                source = row.get('source','')
                date = row.get('publishedAt','')

                st.markdown(
                    f"""
                    <div style="background-color:#1e1e1e;padding:12px;border-radius:8px;margin-bottom:10px;">
                        <a href="{url}" target="_blank" style="font-size:16px;color:#61dafb;font-weight:bold;text-decoration:none;">
                            {title}
                        </a>
                        <p style="margin:4px 0;font-size:12px;color:#9a9a9a;">
                            Source: {source} | Published: {date}
                        </p>
                        <p style="margin-top:6px;color:#cfcfcf;font-size:14px;">{summary}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ---------- AI Summary Column ----------
    with c:
        st.subheader("AI Summary")
        if ai_df.empty:
            st.info("No AI summary generated")
        else:
            ai_text = ai_df.loc[0, 'ai_summary']
            st.markdown(
                f"""
                <div style="background-color:#1e1e1e;padding:12px;border-radius:8px;margin-bottom:10px;">
                    <p style="color:#cfcfcf;font-size:14px;">{ai_text}</p>
                </div>
                """,
                unsafe_allow_html=True
            )


    # Bias analysis + plots
    fig_sentiment, top_words = bias_analysis_and_plots(reddit_df, news_df, ai_df)
    st.markdown("## Bias & Sentiment")
    st.pyplot(fig_sentiment)

    st.markdown("### Top descriptive words (heuristic)")
    for name, words in top_words.items():
        st.write(f"**{name}**: " + ", ".join([f"{w}:{c}" for w,c in words[:10]]))

    # Provide CSV downloads
    st.markdown("---")
    st.download_button("Download Reddit CSV", data=reddit_df.to_csv(index=False), file_name='reddit_cleaned.csv')
    st.download_button("Download News CSV", data=news_df.to_csv(index=False), file_name='news_cleaned.csv')
    st.download_button("Download AI Summary CSV", data=ai_df.to_csv(index=False), file_name='ai_summary.csv')

    # Combine data and train ML model
    combined_df = pd.concat([reddit_df, news_df, ai_df], ignore_index=True)
    train_sentiment_model(combined_df)


else:
    st.info("Enter a topic above and click 'Run Analysis' to start.")