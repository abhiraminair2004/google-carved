import streamlit as st
import pandas as pd
import os
from datetime import datetime
from recommend import recommend_news, save_user_log, load_user_logs
from utils.cleaner import clean_news_data
from utils.visualizer import plot_category_bar, plot_category_pie, plot_topic_wordcloud

# Paths
NEWS_PATH = os.path.join('data', 'news_dataset.csv')
USER_LOGS_PATH = os.path.join('user_logs', 'user_logs.csv')

# User ID (simulate single user for demo)
USER_ID = 'user_1'

# Load and clean data
def load_data():
    df = pd.read_csv(NEWS_PATH)
    df = clean_news_data(df)
    df = df.reset_index(drop=True)
    return df

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
        liked_news = news_df.iloc[liked['news_id'].astype(int)]
        most_liked_cat = liked_news['category'].mode()[0] if not liked_news['category'].empty else None
        top_topics = liked_news['topic'].value_counts().head(3).index.tolist()
    col1, col2, col3 = st.columns(3)
    col1.metric('Total News Seen', total_seen)
    col2.metric('Most Liked Category', most_liked_cat if most_liked_cat else '-')
    col3.metric('Top 3 Topics', ', '.join(top_topics) if top_topics else '-')
    st.markdown('---')
    st.write('**Your News Preferences Visualized:**')
    fig_dashboard = plot_category_bar(news_df)
    st.pyplot(fig_dashboard)
    st.plotly_chart(plot_category_pie(news_df))
    fig_wordcloud = plot_topic_wordcloud(news_df)
    st.pyplot(fig_wordcloud)

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
        liked_news = news_df.iloc[liked['news_id'].astype(int)]
        most_liked_cat = liked_news['category'].mode()[0] if not liked_news['category'].empty else None
        top_topics = liked_news['topic'].value_counts().head(3).index.tolist()
    st.sidebar.metric('Total News Seen', total_seen)
    st.sidebar.metric('Most Liked Category', most_liked_cat if most_liked_cat else '-')
    st.sidebar.metric('Top 3 Topics', ', '.join(top_topics) if top_topics else '-')
    st.sidebar.markdown('---')
    st.sidebar.write('**Visualizations:**')
    fig_sidebar = plot_category_bar(news_df)
    st.sidebar.pyplot(fig_sidebar)
    st.sidebar.plotly_chart(plot_category_pie(news_df))

    # Carved Dashboard
    carved_dashboard(news_df, logs_df)

    st.header('🧠 Your Personalized News Feed')
    recs = recommend_news(USER_ID, news_df, top_n=5)
    for idx, row in recs.iterrows():
        with st.container():
            st.subheader(row['title'])
            st.write(f"**Category:** {row['category']} | **Date:** {row['date']}")
            st.write(row['summary'])
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f'❤️ Like', key=f'like_{idx}'):
                    save_user_log(USER_ID, idx, 'like', datetime.now().isoformat())
                    st.success('Liked!')
            with col2:
                if st.button(f'❌ Discard', key=f'discard_{idx}'):
                    save_user_log(USER_ID, idx, 'discard', datetime.now().isoformat())
                    st.info('Discarded!')
    st.markdown('---')
    st.write('**Tip:** Like or discard news to improve your recommendations!')

if __name__ == '__main__':
    main() 