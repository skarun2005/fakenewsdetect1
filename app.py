import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from bs4 import BeautifulSoup
import requests
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
from urllib.parse import urlparse

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# 📦 Load Pre-trained Models and Data
# ================================
@st.cache_resource
def load_models_and_data():
    # Load tokenizer and BERT model
    MODEL_NAME = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    
    # Load and preprocess dataset (assuming a1_True.csv and a2_Fake.csv from shared folder)
    try:
        true_data = pd.read_csv('a1_True.csv')
        fake_data = pd.read_csv('a2_Fake.csv')
        true_data['label'] = 0  # Real
        fake_data['label'] = 1  # Fake
        data = pd.concat([true_data, fake_data], ignore_index=True)
        data = data[['Text', 'label', 'Site_url']].dropna()
        data = data[data['label'].isin([0, 1])]
        label_encoder = LabelEncoder()
        data['label'] = label_encoder.fit_transform(data['label'])  # 0=Fake, 1=Real
    except FileNotFoundError:
        st.error("Dataset files (a1_True.csv, a2_Fake.csv) not found. Please upload them.")
        return None, None, None, None

    # Prepare dataset for BERT
    dataset = Dataset.from_pandas(data)
    dataset = dataset.map(lambda x: tokenizer(x['Text'], padding="max_length", truncation=True, max_length=256), batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=['input_ids', 'attention_mask', 'labels'])

    # Prepare GNN data
    sources = list(data['Site_url'].unique())
    source_encoder = {src: i for i, src in enumerate(sources)}
    edge_index = []
    num_sources = len(sources)
    if num_sources > 1:
        for i in range(num_sources):
            for j in range(i + 1, num_sources):
                edge_index.append([i, j])
                edge_index.append([j, i])
        edge_index = torch.tensor(edge_index).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    x = torch.rand(num_sources, 8)  # Random initial features
    source_labels = data.groupby('Site_url')['label'].mean().reindex(sources).fillna(0.5)
    y = torch.tensor([int(label > 0.5) for label in source_labels.values])

    data_gnn = Data(x=x, edge_index=edge_index, y=y).to(device)

    # Define GNN model
    class GNN(nn.Module):
        def __init__(self, in_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, 16)
            self.conv2 = GCNConv(16, 1)
        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = F.relu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
            return torch.sigmoid(x)
    
    gnn_model = GNN(in_channels=x.shape[1]).to(device)
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    # Train GNN (simplified training loop)
    gnn_model.train()
    for _ in range(50):  # 50 epochs
        optimizer.zero_grad()
        out = gnn_model(data_gnn).view(-1)
        loss = loss_fn(out, data_gnn.y.float())
        loss.backward()
        optimizer.step()

    return tokenizer, model, gnn_model, data_gnn, label_encoder, source_encoder

# Load models and data
tokenizer, model, gnn_model, data_gnn, label_encoder, source_encoder = load_models_and_data()
if tokenizer is None:
    st.stop()

# ================================
# 🕸️ Prediction Functions
# ================================
def scrape_text_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        html = requests.get(url, timeout=10, headers=headers).text
        soup = BeautifulSoup(html, "html.parser")
        article_body = soup.find('article')
        if article_body:
            paragraphs = article_body.find_all('p')
        else:
            paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])
        return text.strip()
    except Exception as e:
        st.warning(f"Error scraping URL {url}: {e}")
        return ""

def extract_domain(url):
    if not isinstance(url, str) or not url.strip():
        return "unknown"
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else (netloc or "unknown")
    except:
        return "unknown"

def predict_news(text, site_url=""):
    model.eval()
    gnn_model.eval()
    with torch.no_grad():
        # BERT prediction
        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        output = model(**tokens)
        probabilities = torch.softmax(output.logits, dim=1).squeeze().cpu().numpy()
        pred = np.argmax(probabilities).item()
        label = label_encoder.inverse_transform([pred])[0]
        confidence = float(probabilities[pred])

        # GNN credibility score
        src_domain = extract_domain(site_url) if site_url else "unknown"
        if src_domain in source_encoder:
            src_idx = source_encoder[src_domain]
            credibility = gnn_model(data_gnn)[src_idx].item()
        else:
            credibility = 0.5  # Neutral for unknown sources

    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "source": src_domain,
        "credibility_score": round(float(credibility), 2)
    }

# ================================
# 🌐 Streamlit Interface
# ================================
st.title("Fake News Detector with Source Credibility")
st.write("Enter a news article or URL to check if it's fake or real, and get the source credibility score.")

# Input field
user_input = st.text_input("Enter news text or URL:", "")

if st.button("Predict"):
    if user_input.strip():
        # Check if input is a URL
        if user_input.lower().startswith(('http://', 'https://')):
            scraped_text = scrape_text_from_url(user_input)
            if scraped_text:
                result = predict_news(scraped_text, user_input)
                st.success("Prediction successful!")
                st.write(f"**Prediction**: {result['prediction']} | **Confidence**: {result['confidence']}")
                st.write(f"**Source**: {result['source']}")
                st.write(f"**Credibility Score**: {result['credibility_score']} (0 = Low, 1 = High)")
            else:
                st.error("Failed to scrape text from the URL.")
        else:
            result = predict_news(user_input)
            st.success("Prediction successful!")
            st.write(f"**Prediction**: {result['prediction']} | **Confidence**: {result['confidence']}")
            st.write(f"**Source**: {result['source']}")
            st.write(f"**Credibility Score**: {result['credibility_score']} (0 = Low, 1 = High)")
    else:
        st.warning("Please enter some text or a URL.")

# Optional: Display instructions
st.write("**Instructions**:")
st.write("- Enter a news article text or a URL (e.g., https://example.com/news).")
st.write("- Click 'Predict' to get the result.")
st.write("- The credibility score is based on the GNN analysis of source domains from the dataset.")

# Optional: Add a file uploader for dataset (if needed)
st.write("Upload dataset files if not already present:")
uploaded_files = st.file_uploader("Upload a1_True.csv and a2_Fake.csv", accept_multiple_files=True, type=["csv"])
if uploaded_files:
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success("Files uploaded successfully. Reload the page to apply changes.")
