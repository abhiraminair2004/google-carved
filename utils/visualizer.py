import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import pandas as pd

# Bar chart using seaborn

def plot_category_bar(data):
    """Plot a bar chart of news counts by category."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots()
    sns.countplot(y='category', data=data, order=data['category'].value_counts().index, ax=ax)
    ax.set_title('News Count by Category')
    return fig

# Pie chart using plotly

def plot_category_pie(data):
    """Plot a pie chart of news by category."""
    fig = px.pie(data, names='category', title='News Distribution by Category')
    return fig

# Word cloud for topics

def plot_topic_wordcloud(data):
    """Generate a word cloud for news topics."""
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    
    # Check if topic column exists, use category as fallback
    text = ''
    if 'topic' in data.columns and not data['topic'].empty:
        # Filter out numeric-only topics
        topics = data['topic'].astype(str)
        # Only use non-numeric topics
        topics = topics[~topics.str.match(r'^\d+$')]
        if not topics.empty:
            text = ' '.join(topics)
    
    # If no valid topics, try using category
    if not text and 'category' in data.columns and not data['category'].empty:
        text = ' '.join(data['category'].astype(str))
    
    # If still no text, use default message
    if not text:
        text = 'No topics available'
    
    # Ensure there's at least one word to generate the cloud
    if len(text.split()) < 1:
        text = 'No topics available'
    
    fig = plt.figure(figsize=(10, 5))
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Topic Word Cloud')
    return fig