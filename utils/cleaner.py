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