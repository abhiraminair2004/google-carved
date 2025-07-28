# Google Carved

**Google Carved** is a personalized news recommendation system inspired by **Google Discover** and **Spotify Wrapped**. It uses **content-based filtering (TF-IDF + cosine similarity)** to recommend news articles based on user preferences, and offers a rich visual dashboard to analyze those preferences.

Built using **Python**, **Streamlit**, and **Scikit-learn**, it’s designed to be fast, lightweight, and interactive.


<img width="1920" height="884" alt="image" src="https://github.com/user-attachments/assets/c7bcd106-b0fe-4c5b-85b0-6b407297a8e9" />

---
## 🌐 Live Demo

- **App**: [https://carvednews.streamlit.app](https://carvednews.streamlit.app)

---

## 🚀 Features

- 🔍 **Personalized Recommendations** using content similarity (TF-IDF on titles and categories)
- ❤️ **Like or Discard News** to tune future suggestions
- 📊 **Carved Dashboard** shows insights like:
  - Most liked categories
  - Like vs discard ratio
  - Topic word cloud (Spotify Wrapped-style)
- 📈 **Interactive Visualizations** using bar chart, pie chart, and word cloud
- 🧼 **Clean Data Pipeline** for both static and API-fetched news
- 📡 **Live News Fetching** from [NewsAPI](https://newsapi.org/)
- 💾 **User Log Tracking** to persist preferences

---

## 🧠 How It Works

1. News articles are vectorized using **TF-IDF** (title + category)
2. User likes/dislikes are recorded in `user_logs.csv`
3. Similar articles are recommended using **cosine similarity**
4. User insights are shown in the form of visual dashboards

---

## 📂 Project Structure

```
google-carved/
├── app.py                      # Streamlit UI & app logic
├── recommend.py                # Recommendation engine (TF-IDF + similarity)
├── data/
│   └── news_dataset.csv        # Base dataset of news articles
├── user_logs/
│   └── user_logs.csv           # Like/dislike logs per user
├── utils/
│   ├── cleaner.py              # Data cleaning functions
│   └── visualizer.py           # Plotting bar chart, pie chart, word cloud
├── requirements.txt            # Dependencies
├── .streamlit/
│   └── config.toml             # Streamlit theme config
└── README.md                   # Project overview (this file)
```

---

## 📊 Visuals & Insights

- **Bar Chart**: Most liked news categories  
- **Pie Chart**: Distribution of likes vs discards  
- **Word Cloud**: Most common words in liked headlines

These are updated live based on user interaction.

---

## 🛠️ Tech Stack

| Layer        | Tool / Library                  |
|--------------|----------------------------------|
| Web UI       | Streamlit                        |
| Data Handling| Pandas                           |
| ML Logic     | TF-IDF (Scikit-learn), Numpy     |
| Visualization| Matplotlib, WordCloud            |
| External API | NewsAPI.org                      |
| Storage      | CSV files for news and logs      |

---

## 🔧 Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/google-carved.git
   cd google-carved
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## ✅ Example Use Case

> A user launches the app, sees top headlines, likes a few sports and tech articles. The app learns their preferences and starts recommending more sports/tech. After a while, a dashboard shows charts and a word cloud summarizing their interests.

---

## 📌 To-Do (Optional Enhancements)

- [ ] Add user login for multi-user support
- [ ] Add collaborative filtering model
- [ ] Store logs in database instead of CSV
- [ ] Filter news by country or language
- [ ] Add sentiment analysis to liked news

---

