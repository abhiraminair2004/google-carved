import streamlit as st
import pandas as pd
import os
from datetime import datetime
from recommend import hybrid_recommend, save_user_log, load_user_logs
from utils.cleaner import clean_news_data, clean_api_news
from utils.visualizer import plot_category_bar, plot_category_pie, plot_topic_wordcloud
import requests

# Paths
NEWS_PATH = os.path.join('data', 'news_dataset.csv')
USER_LOGS_PATH = os.path.join('user_logs', 'user_logs.csv')

# User ID (simulate single user for demo)
USER_ID = 'user_1'

# News API configuration
NEWSAPI_KEY = 'e6e5c39f7a09461f98750f7dc6ad45bd'
NEWSAPI_URL = 'https://newsapi.org/v2/top-headlines?country=us&pageSize=10&apiKey=' + NEWSAPI_KEY

# Load and clean data
def load_data():
    try:
        # Try with tab delimiter and Python engine to handle inconsistent field counts
        df = pd.read_csv(NEWS_PATH, delimiter='\t', engine='python', on_bad_lines='skip')
    except Exception as e:
        st.error(f"Error loading data with tab delimiter: {e}")
        try:
            # Fallback to C engine with flexible field count
            df = pd.read_csv(NEWS_PATH, delimiter='\t', engine='c', on_bad_lines='skip')
        except Exception as e2:
            st.error(f"Error loading data with fallback method: {e2}")
            # Last resort: try to infer delimiter
            try:
                df = pd.read_csv(NEWS_PATH, sep=None, engine='python', on_bad_lines='skip')
            except Exception as e3:
                st.error(f"Failed to load data: {e3}")
                # Return empty DataFrame with expected columns
                return pd.DataFrame(columns=['category', 'title', 'summary', 'topic', 'date'])
    
    df = clean_news_data(df)
    # Ensure all required columns exist
    required_columns = ['category', 'title', 'summary', 'topic', 'date']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
    df = df.reset_index(drop=True)
    return df

def fetch_latest_news():
    try:
        response = requests.get(NEWSAPI_URL)
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            if not articles:
                return None, 'No articles found.'
            df_api = pd.DataFrame(articles)
            df_api = clean_api_news(df_api)
            return df_api, None
        else:
            return None, f'API error: {response.status_code}'
    except Exception as e:
        return None, f'Error: {e}'

def load_logs():
    if os.path.exists(USER_LOGS_PATH):
        return pd.read_csv(USER_LOGS_PATH)
    else:
        return pd.DataFrame(columns=['user_id', 'news_id', 'action', 'timestamp'])

# --- Carved Dashboard (Spotify Wrapped style) ---
def carved_dashboard(news_df, logs_df):
    st.subheader('🪓 Carved Dashboard: Your News Year in Review')
    user_logs = logs_df[logs_df['user_id'] == USER_ID]
    total_seen = user_logs['news_id'].nunique()
    liked = user_logs[user_logs['action'] == 'like']
    most_liked_cat = None
    top_topics = []
    
    if not liked.empty:
        # Get indices of liked news, handling potential errors
        try:
            # Get the most recent liked news first
            liked = liked.sort_values('timestamp', ascending=False)
            liked_indices = liked['news_id'].astype(int).tolist()
            
            # Filter only valid indices
            valid_indices = [idx for idx in liked_indices if idx < len(news_df)]
            if valid_indices:
                liked_news = news_df.iloc[valid_indices]
                
                # Get most liked category if category column exists
                if 'category' in liked_news.columns and not liked_news['category'].empty:
                    # Use the most recent liked category (first row after sorting)
                    most_liked_cat = liked_news['category'].iloc[0]
                
                # Get top topics if topic column exists
                if 'topic' in liked_news.columns:
                    # Filter out numeric-only topics
                    topics = liked_news['topic'].astype(str)
                    non_numeric_topics = topics[~topics.str.match(r'^\d+$')]
                    
                    if not non_numeric_topics.empty:
                        top_topics = non_numeric_topics.value_counts().head(3).index.tolist()
                    elif 'category' in liked_news.columns:  # Use category as fallback
                        top_topics = liked_news['category'].value_counts().head(3).index.tolist()
                elif 'category' in liked_news.columns:  # Use category as fallback
                    top_topics = liked_news['category'].value_counts().head(3).index.tolist()
        except Exception as e:
            st.warning(f"Error processing liked news: {e}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric('Total News Seen', total_seen)
    col2.metric('Most Liked Category', most_liked_cat if most_liked_cat else '-')
    col3.metric('Top 3 Topics', ', '.join(top_topics) if top_topics else '-')
    st.markdown('---')
    st.write('**Your News Preferences Visualized:**')
    
    # Generate visualizations
    try:
        fig_dashboard = plot_category_bar(news_df)
        st.pyplot(fig_dashboard)
    except Exception as e:
        st.warning(f"Could not generate category bar chart: {e}")
    
    try:
        st.plotly_chart(plot_category_pie(news_df))
    except Exception as e:
        st.warning(f"Could not generate category pie chart: {e}")
    
    try:
        fig_wordcloud = plot_topic_wordcloud(news_df)
        st.pyplot(fig_wordcloud)
    except Exception as e:
        st.warning(f"Could not generate topic word cloud: {e}")

# --- Main App ---
def main():
    st.set_page_config(page_title='Google Carved', layout='wide')
    st.title('📰 Google Carved')
    st.caption('A personalized news recommendation system inspired by Google Discover & Spotify Wrapped.')

    # Load data
    news_df = load_data()
    logs_df = load_logs()

    # Sidebar analytics
    st.sidebar.header('📊 Your Analytics')
    user_logs = logs_df[logs_df['user_id'] == USER_ID]
    total_seen = user_logs['news_id'].nunique()
    liked = user_logs[user_logs['action'] == 'like']
    most_liked_cat = None
    top_topics = []
    
    if not liked.empty:
        # Get indices of liked news, handling potential errors
        try:
            # Get the most recent liked news first
            liked = liked.sort_values('timestamp', ascending=False)
            liked_indices = liked['news_id'].astype(int).tolist()
            
            # Filter only valid indices
            valid_indices = [idx for idx in liked_indices if idx < len(news_df)]
            if valid_indices:
                liked_news = news_df.iloc[valid_indices]
                
                # Get most liked category if category column exists
                if 'category' in liked_news.columns and not liked_news['category'].empty:
                    # Use the most recent liked category (first row after sorting)
                    most_liked_cat = liked_news['category'].iloc[0]
                
                # Get top topics if topic column exists
                if 'topic' in liked_news.columns:
                    # Filter out numeric-only topics
                    topics = liked_news['topic'].astype(str)
                    non_numeric_topics = topics[~topics.str.match(r'^\d+$')]
                    
                    if not non_numeric_topics.empty:
                        top_topics = non_numeric_topics.value_counts().head(3).index.tolist()
                    elif 'category' in liked_news.columns:  # Use category as fallback
                        top_topics = liked_news['category'].value_counts().head(3).index.tolist()
                elif 'category' in liked_news.columns:  # Use category as fallback
                    top_topics = liked_news['category'].value_counts().head(3).index.tolist()
        except Exception as e:
            st.sidebar.warning(f"Error processing liked news: {e}")
    
    st.sidebar.metric('Total News Seen', total_seen)
    st.sidebar.metric('Most Liked Category', most_liked_cat if most_liked_cat else '-')
    st.sidebar.metric('Top 3 Topics', ', '.join(top_topics) if top_topics else '-')
    st.sidebar.markdown('---')
    st.sidebar.write('**Visualizations:**')
    
    # Generate visualizations
    try:
        fig_sidebar = plot_category_bar(news_df)
        st.sidebar.pyplot(fig_sidebar)
    except Exception as e:
        st.sidebar.warning(f"Could not generate category bar chart: {e}")
    
    try:
        st.sidebar.plotly_chart(plot_category_pie(news_df))
    except Exception as e:
        st.sidebar.warning(f"Could not generate category pie chart: {e}")

    # Carved Dashboard
    carved_dashboard(news_df, logs_df)

    st.header('🧠 Your Personalized News Feed')
    recs, explanations = hybrid_recommend(USER_ID, news_df, top_n=5)
    for idx, (row, expl) in enumerate(zip(recs.itertuples(), explanations)):
        with st.container():
            st.subheader(row.title)
            st.write(f"**Category:** {row.category} | **Date:** {row.date}")
            st.write(row.summary)
            st.info(f"_Why recommended:_ {expl}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f'❤️ Like', key=f'like_{row.Index}'):
                    save_user_log(USER_ID, row.Index, 'like', datetime.now().isoformat())
                    st.success('Liked!')
            with col2:
                if st.button(f'❌ Discard', key=f'discard_{row.Index}'):
                    save_user_log(USER_ID, row.Index, 'discard', datetime.now().isoformat())
                    st.info('Discarded!')
    st.markdown('---')
    st.write('**Tip:** Like or discard news to improve your recommendations!')
    # --- Real-time news fetch button (API integration next) ---
    st.sidebar.markdown('---')
    if st.sidebar.button('📰 Fetch Latest News (API)', key='fetch_news'):
        df_api, err = fetch_latest_news()
        if err:
            st.sidebar.error(f'Failed to fetch news: {err}')
        else:
            # Merge new articles, deduplicate, and update CSV
            news_df = pd.concat([news_df, df_api], ignore_index=True)
            news_df = clean_news_data(news_df)
            news_df = news_df.drop_duplicates(subset=['title', 'summary'])
            news_df.to_csv(NEWS_PATH, index=False)
            st.sidebar.success(f'Fetched and merged {len(df_api)} new articles! Refresh the page to see updates.')

if __name__ == '__main__':
    main()