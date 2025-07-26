import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os

USER_LOGS_PATH = os.path.join('user_logs', 'user_logs.csv')

# Helper: Load user logs
def load_user_logs(user_id):
    if os.path.exists(USER_LOGS_PATH):
        logs = pd.read_csv(USER_LOGS_PATH)
        return logs[logs['user_id'] == user_id]
    else:
        return pd.DataFrame(columns=['user_id', 'news_id', 'action', 'timestamp'])

# Helper: Save a new user interaction
def save_user_log(user_id, news_id, action, timestamp):
    logs = load_user_logs(user_id)
    new_log = pd.DataFrame([[user_id, news_id, action, timestamp]], columns=['user_id', 'news_id', 'action', 'timestamp'])
    all_logs = pd.concat([logs, new_log], ignore_index=True)
    all_logs.to_csv(USER_LOGS_PATH, index=False)

# Get user preferences from logs
def get_user_preferences(user_logs, news_df):
    liked = user_logs[user_logs['action'] == 'like']
    if liked.empty:
        return []
    liked_news = news_df.iloc[liked['news_id'].astype(int)]
    topics = liked_news['topic'].tolist()
    categories = liked_news['category'].tolist()
    return topics + categories

# Recommend news based on content similarity
def recommend_news(user_id, news_df, top_n=5):
    user_logs = load_user_logs(user_id)
    prefs = get_user_preferences(user_logs, news_df)
    # If no prefs, recommend most recent
    if not prefs:
        return news_df.sort_values('date', ascending=False).head(top_n)
    # Content-based filtering using TF-IDF on topic+category
    news_df = news_df.copy()
    news_df['content'] = news_df['topic'].astype(str) + ' ' + news_df['category'].astype(str)
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(news_df['content'])
    # Build a pseudo-query from user prefs
    user_profile = ' '.join(prefs)
    user_vec = tfidf.transform([user_profile])
    cosine_sim = linear_kernel(user_vec, tfidf_matrix).flatten()
    # Exclude already seen news
    seen_ids = user_logs['news_id'].astype(int).tolist()
    news_df['score'] = cosine_sim
    recs = news_df[~news_df.index.isin(seen_ids)].sort_values('score', ascending=False).head(top_n)
    if recs.empty:
        recs = news_df.sort_values('date', ascending=False).head(top_n)
    return recs 