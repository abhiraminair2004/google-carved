# Google Carved

A personalized news recommendation system inspired by Google Discover and Spotify Wrapped, built with Python and Streamlit.

## Features
- Personalized news feed using content-based filtering (TF-IDF)
- Like/Discard news to improve recommendations
- User analytics: most liked category, top topics, total news seen
- Visualizations: bar chart, pie chart, word cloud
- "Carved Dashboard" (Spotify Wrapped style) at startup
- Responsive, interactive UI with Streamlit widgets
- Modular, beginner-friendly code

## File Structure
```
app.py                  # Main Streamlit app
recommend.py            # Recommendation logic
utils/cleaner.py        # Data cleaning functions
utils/visualizer.py     # Visualization utilities
requirements.txt        # Dependencies
README.md               # This file
data/news_dataset.csv   # Dummy news dataset
user_logs/user_logs.csv # User interaction logs
```

## Setup & Usage
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the app locally:**
   ```bash
   streamlit run app.py
   ```
3. **Interact:**
   - Like or discard news to improve your feed
   - View analytics and visualizations in the sidebar and dashboard

## Streamlit Cloud Deployment
- Upload all files to your Streamlit Cloud workspace
- Ensure `requirements.txt` is present
- Set `app.py` as the entry point
- App will run at your Streamlit Cloud URL

## Notes
- Uses dummy news data for demonstration
- All user logs are stored in `user_logs/user_logs.csv`
- Modular code for easy extension

## Dependencies
- streamlit
- pandas
- scikit-learn
- matplotlib
- seaborn
- plotly
- wordcloud

---
Enjoy your personalized news experience with Google Carved! 