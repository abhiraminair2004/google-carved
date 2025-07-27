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

## 🧑‍💻 Tech Stack

| Layer        | Tech Used                   |
|--------------|-----------------------------|
| UI           | Streamlit (Python)          |
| ML Model     | Scikit-learn, Pandas, NumPy |
| Visualization| Seaborn, Matplotlib         |
| Deployment   | Streamlit Cloud             |
| Storage      | In-memory (session/local)   |

---

## 🛠️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/google-carved.git
cd google-carved
