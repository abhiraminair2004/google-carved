import pandas as pd

def clean_news_data(df):
    """
    Clean the news DataFrame by removing missing values, duplicates, and normalizing categories.
    """
    # Remove missing values
    df = df.dropna()
    # Remove duplicates
    df = df.drop_duplicates()
    # Normalize category names (strip and title case)
    if 'category' in df.columns:
        df['category'] = df['category'].str.strip().str.title()
    return df

def clean_api_news(df):
    """
    Clean and format news articles fetched from NewsAPI for merging with the main dataset.
    """
    # Standardize columns
    df = df.rename(columns={
        'title': 'title',
        'description': 'summary',
        'publishedAt': 'date',
        'category': 'category',
        'content': 'content',
        'source': 'source',
        'url': 'url',
        'author': 'author',
        'topic': 'topic',
    })
    # Fill missing columns
    for col in ['category', 'topic']:
        if col not in df.columns:
            df[col] = 'General'
    # Remove missing values and duplicates
    df = df.dropna(subset=['title', 'summary', 'date'])
    df = df.drop_duplicates(subset=['title', 'summary'])
    # Format date
    df['date'] = pd.to_datetime(df['date']).dt.date.astype(str)
    return df 