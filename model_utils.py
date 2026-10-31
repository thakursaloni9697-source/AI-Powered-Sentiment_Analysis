import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    """Load Hugging Face DistilBERT pipeline, tokenizer, and underlying model."""
    try:
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        classifier = pipeline("sentiment-analysis", model=model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return classifier, tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def generate_advanced_sample_dataset(domain="E-Commerce Reviews", size=30):
    """Generates rich synthetic domain datasets for batch testing."""
    if domain == "E-Commerce Reviews":
        pos_samples = [
            "The product exceeded all expectations! Premium packaging and exceptionally fast shipping.",
            "Absolutely phenomenal quality. Worth every single penny spent.",
            "Sleek design, robust build quality, and outstanding customer support.",
            "Exquisite craftsmanship and premium feel. Highly recommended to everyone!"
        ]
        neg_samples = [
            "Terrible build quality. Stopped working after two days of light usage.",
            "Appalling customer service. The item arrived damaged and nobody responded.",
            "Complete waste of money. Flimsy materials and sub-par performance.",
            "Very disappointed with the battery life and heating issues. Returning immediately."
        ]
    elif domain == "Financial Sentiment":
        pos_samples = [
            "Q3 earnings report shattered market estimates with robust revenue expansion.",
            "Strategic acquisition accelerates enterprise growth and margin accretion.",
            "Strong bullish momentum backed by solid institutional capital inflows."
        ]
        neg_samples = [
            "Regulatory headwinds and margin contraction lead to severe downgrades.",
            "Unanticipated supply chain disruptions threaten quarterly guidance targets.",
            "Bearish technical breakout signals prolonged downside consolidation."
        ]
    elif domain == "Tech Support Tickets":
        pos_samples = [
            "Resolved the latency spike instantly. Incredible support engineer assistance!",
            "Seamless API migration with zero downtime. Exceptional documentation.",
            "The authentication bug fix was deployed rapidly. Excellent turnaround time."
        ]
        neg_samples = [
            "Critical database deadlock caused prolonged service outage across clusters.",
            "OAuth token expiration bug is locking out active enterprise users.",
            "Unresponsive support desk while production environments remain stalled."
        ]
    else:
        pos_samples = [
            "An absolute joy to use! Intuitive interface and blazing fast performance.",
            "Brilliant experience overall. Will definitely continue using this service."
        ]
        neg_samples = [
            "Frustrating user experience with persistent bugs and crashes.",
            "Subpar performance and confusing navigation layout."
        ]

    np.random.seed(42)
    texts, sources, ratings = [], [], []
    
    for _ in range(size):
        is_pos = np.random.rand() > 0.45
        t = np.random.choice(pos_samples if is_pos else neg_samples)
        texts.append(t)
        sources.append(domain)
        ratings.append(np.random.randint(4, 6) if is_pos else np.random.randint(1, 3))

    return pd.DataFrame({
        "feedback_text": texts,
        "domain_category": sources,
        "satisfaction_rating": ratings
    })
