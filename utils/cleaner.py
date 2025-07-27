import pandas as pd

def clean_news_data(df):
    """
    Clean the news DataFrame by removing missing values, duplicates, and normalizing categories.
    Also adds missing columns needed by the app if they don't exist.
    """
    # Remove missing values
    df = df.dropna()
    # Remove duplicates
    df = df.drop_duplicates()
    # Normalize category names (strip and title case)
    if 'category' in df.columns:
        df['category'] = df['category'].str.strip().str.title()
    
    # Add missing columns needed by the app
    if 'summary' not in df.columns and 'content' in df.columns:
        # Create summary from content (first 100 characters)
        df['summary'] = df['content'].str[:100] + '...'
    
    if 'topic' not in df.columns:
        # Extract topic from filename or use category as fallback
        if 'filename' in df.columns:
            # Extract meaningful topic names instead of just numbers
            # First, extract the base filename without extension
            df['topic'] = df['filename'].str.split('.').str[0]
            
            # If topics are just numbers (like '001', '002'), use category instead
            if df['topic'].str.match(r'^\d+$').all() and 'category' in df.columns:
                df['topic'] = df['category']
        else:
            df['topic'] = df['category']
    
    if 'date' not in df.columns:
        # Add current date as fallback
        import datetime
        df['date'] = datetime.datetime.now().strftime('%Y-%m-%d')
    
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