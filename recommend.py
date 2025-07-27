import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import os

USER_LOGS_PATH = os.path.join('user_logs', 'user_logs.csv')

# Helper: Load user logs
def load_user_logs(user_id=None):
    if os.path.exists(USER_LOGS_PATH):
        logs = pd.read_csv(USER_LOGS_PATH)
        if user_id is not None:
            return logs[logs['user_id'] == user_id]
        return logs
    else:
        return pd.DataFrame(columns=['user_id', 'news_id', 'action', 'timestamp'])

# Helper: Save a new user interaction
def save_user_log(user_id, news_id, action, timestamp):
    logs = load_user_logs()
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

# --- Collaborative Filtering ---
def collaborative_recommend(user_id, news_df, logs_df, top_n=5):
    # Build user-item matrix (users x news)
    user_ids = logs_df['user_id'].unique()
    news_ids = news_df.index
    user_item = pd.DataFrame(0, index=user_ids, columns=news_ids)
    for _, row in logs_df.iterrows():
        if row['action'] == 'like':
            user_item.at[row['user_id'], int(row['news_id'])] = 1
        # Implicit feedback: simulate time spent (if available)
        # You can extend this to use a 'time_spent' column
    # Compute user-user similarity
    if user_id not in user_item.index:
        return pd.DataFrame()
    user_vec = user_item.loc[user_id].values.reshape(1, -1)
    all_vecs = user_item.values
    sim = linear_kernel(user_vec, all_vecs).flatten()
    # Get top similar users (excluding self)
    similar_users = user_item.index[np.argsort(sim)[::-1][1:3]]  # top 2 similar users
    # Get news liked by similar users but not seen by current user
    seen = set(user_item.loc[user_id][user_item.loc[user_id] > 0].index)
    recs = set()
    for u in similar_users:
        liked = set(user_item.loc[u][user_item.loc[u] > 0].index)
        recs.update(liked - seen)
    # Return top_n recommended news
    recs = list(recs)
    if not recs:
        return pd.DataFrame()
    return news_df.loc[recs].head(top_n)

# --- Content-Based Filtering ---
def content_based_recommend(user_id, news_df, user_logs, top_n=5):
    prefs = get_user_preferences(user_logs, news_df)
    
    # Ensure required columns exist
    news_df = news_df.copy()
    if 'date' not in news_df.columns:
        import datetime
        news_df['date'] = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # If no preferences yet, return a diverse set of recent articles from different categories
    if not prefs:
        # Get the most recent article from each category
        diverse_recs = pd.DataFrame()
        for category in news_df['category'].unique():
            category_news = news_df[news_df['category'] == category].sort_values('date', ascending=False).head(1)
            diverse_recs = pd.concat([diverse_recs, category_news])
        
        # If we still need more recommendations, add more recent articles
        if len(diverse_recs) < top_n:
            # Exclude categories we already have
            existing_categories = set(diverse_recs['category'])
            remaining = news_df[~news_df['category'].isin(existing_categories)].sort_values('date', ascending=False)
            diverse_recs = pd.concat([diverse_recs, remaining.head(top_n - len(diverse_recs))])
        
        # Ensure we don't exceed top_n
        diverse_recs = diverse_recs.head(top_n)
        return diverse_recs, ["Recent article from " + cat for cat in diverse_recs['category']]
    
    # Create content field for TF-IDF
    if 'topic' not in news_df.columns:
        news_df['topic'] = news_df['category']
    
    # Combine features for content-based filtering
    news_df['content_features'] = news_df['topic'].astype(str) + ' ' + news_df['category'].astype(str)
    
    # Apply TF-IDF
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(news_df['content_features'])
    user_profile = ' '.join(prefs)
    user_vec = tfidf.transform([user_profile])
    cosine_sim = linear_kernel(user_vec, tfidf_matrix).flatten()
    
    # Filter out seen articles
    seen_ids = user_logs['news_id'].astype(int).tolist()
    news_df['score'] = cosine_sim
    
    # Get recommendations with category diversity
    # First, get top scored articles for each category
    recs = pd.DataFrame()
    categories = news_df['category'].unique()
    
    # Prioritize categories based on user preferences
    if 'category' in prefs:
        # Move preferred categories to the front
        preferred_categories = [cat for cat in categories if cat in prefs]
        other_categories = [cat for cat in categories if cat not in prefs]
        sorted_categories = preferred_categories + other_categories
    else:
        # Shuffle categories for diversity
        import random
        sorted_categories = list(categories)
        random.shuffle(sorted_categories)
    
    # Select top article from each category until we reach top_n
    for category in sorted_categories:
        if len(recs) >= top_n:
            break
        
        # Get top scored unseen articles from this category
        category_recs = news_df[(news_df['category'] == category) & 
                               (~news_df.index.isin(seen_ids))].sort_values('score', ascending=False).head(1)
        
        if not category_recs.empty:
            recs = pd.concat([recs, category_recs])
    
    # If we still need more recommendations, add more top scored articles
    if len(recs) < top_n:
        remaining = news_df[~news_df.index.isin(seen_ids) & ~news_df.index.isin(recs.index)].sort_values('score', ascending=False)
        recs = pd.concat([recs, remaining.head(top_n - len(recs))])
    
    # Generate explanations
    explanations = [f"Matched topics/categories: {prefs}" for _ in range(len(recs))]
    
    # Fallback to recent articles if no recommendations
    if recs.empty:
        recs = news_df.sort_values('date', ascending=False).head(top_n)
        explanations = ["Most recent articles"] * top_n
        
    return recs, explanations

# --- Hybrid Recommendation ---
def hybrid_recommend(user_id, news_df, top_n=5):
    logs_df = load_user_logs()
    user_logs = logs_df[logs_df['user_id'] == user_id]
    
    # Check if user has any interactions yet
    user_has_interactions = not user_logs.empty
    
    # If user has no interactions, return diverse set of news from different categories
    if not user_has_interactions:
        # Get one article from each category
        diverse_recs = pd.DataFrame()
        for category in news_df['category'].unique():
            category_news = news_df[news_df['category'] == category].sample(1)
            diverse_recs = pd.concat([diverse_recs, category_news])
        
        # Ensure we don't exceed top_n
        diverse_recs = diverse_recs.sample(min(len(diverse_recs), top_n))
        explanations = [f"Featured article from {row.category}" for _, row in diverse_recs.iterrows()]
        return diverse_recs, explanations
    
    # Try collaborative filtering first
    collab_recs = collaborative_recommend(user_id, news_df, logs_df, top_n)
    explanations = []
    
    if not collab_recs.empty:
        explanations = ["Other users with similar likes also liked this"] * len(collab_recs)
        
        # Check if we have diverse categories in collaborative recommendations
        collab_categories = collab_recs['category'].unique()
        
        # If collaborative recs are all from the same category, mix in some content-based
        if len(collab_categories) == 1 and len(collab_recs) > 2:
            # Keep only 2 collaborative recommendations
            collab_recs = collab_recs.head(2)
            explanations = explanations[:2]
            
            # Fill the rest with content-based from different categories
            cb_recs, cb_expl = content_based_recommend(user_id, news_df, user_logs, top_n - len(collab_recs))
            recs = pd.concat([collab_recs, cb_recs]).head(top_n)
            explanations += cb_expl
            return recs, explanations
        
        # Fill up with content-based if not enough
        if len(collab_recs) < top_n:
            cb_recs, cb_expl = content_based_recommend(user_id, news_df, user_logs, top_n - len(collab_recs))
            recs = pd.concat([collab_recs, cb_recs]).head(top_n)
            explanations += cb_expl
            return recs, explanations
            
        return collab_recs, explanations
    
    # Fallback to content-based
    return content_based_recommend(user_id, news_df, user_logs, top_n)