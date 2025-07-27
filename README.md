# Google Carved – Personalized News Recommender

**Google Carved** is a machine learning-based news recommendation app inspired by Google Discover and Spotify Wrapped. It lets users interact with news cards, and based on their likes/dislikes, it recommends more personalized content. The app includes visual feedback through charts and lists of recently liked articles.

<img width="1920" height="884" alt="image" src="https://github.com/user-attachments/assets/c7bcd106-b0fe-4c5b-85b0-6b407297a8e9" />

---

## 🌐 Live Demo

- **App**: [https://carvednews.streamlit.app](https://carvednews.streamlit.app)

---

## 🧠 Features

### 🔍 Personalized News Discovery
- Browse categorized news cards
- Press heart to "like" a card and receive similar content or Cross to discard it
- Visualize your top liked categories using Pie & Bar charts

### 📊 Visual Feedback
- Bar chart and pie chart showing like frequency by category
- Summary text about your preferences
- Simple, clean UI using Streamlit

---

## 👩💻 Tech Stack

### 💻 Frontend
- **Streamlit**: Python-based web framework used to build the interactive user interface with minimal code.
- **Pandas**: For data loading and preprocessing.
- **Matplotlib & Seaborn**: For generating user-friendly visualizations of user preferences and content trends.

### 🧠 Machine Learning
- **Scikit-learn**: Used for implementing content-based recommendation logic (e.g., TF-IDF vectorization, cosine similarity).
- **Natural Language Processing (NLP)**: Applied for vectorizing and comparing article content.

### 🗂️ Data Handling
- **CSV File**: News dataset stored locally in `news_dataset.csv` for simplicity and fast prototyping.

### 🌐 Deployment
- **Streamlit Cloud**: Free and fast deployment platform for Streamlit apps.
- **GitHub**: Version control and project hosting.

---

## 🛠️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/google-carved.git
cd google-carved
