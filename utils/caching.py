# utils/caching.py
import streamlit as st

@st.cache_data
def fetch_stock_data(_data_agent, ticker, start_date, end_date, sources=None):
    """
    Fetch stock data from the data agent and cache it.
    """
    return _data_agent.collect_stock_data(
        ticker=ticker,
        from_date=start_date.strftime("%Y-%m-%d"),
        to_date=end_date.strftime("%Y-%m-%d"),
        sources=sources
    )


def filter_articles_by_score(articles, score_threshold):
    """
    Filters articles based on a minimum sentiment score threshold.
    
    Args:
        articles (list): List of articles with sentiment information.
        score_threshold (float): Minimum absolute sentiment score required.

    Returns:
        list: Filtered list of articles meeting the score threshold.
    """
    filtered = []
    for article in articles:
        sentiment = article.get("sentiment", {})
        score = sentiment.get("score", 0.0)  # Default to 0.0 if score is missing
        if abs(score) >= score_threshold:
            filtered.append(article)
    return filtered



@st.cache_data
def fetch_company_overview(_data_agent, ticker):
    """
    Fetch and cache company overview data using Alpha Vantage.
    """
    try:
        overview, _ = _data_agent.fd.get_company_overview(ticker)
        return overview
    except Exception as e:
        return {"error": f"Failed to fetch company overview: {e}"}

