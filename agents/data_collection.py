from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.timeseries import TimeSeries
from newsapi import NewsApiClient
import os
import json
import datetime
from dotenv import load_dotenv
from agents.sentiment_analysis import SentimentAnalysisAgent
import logging


# Load environment variables
load_dotenv()

class DataCollectionAgent:
    def __init__(self, cache_dir="data/cache/"):
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.newsapi_key = os.getenv("NEWSAPI_KEY")
        self.fd = FundamentalData(self.alpha_vantage_key)
        self.ts = TimeSeries(self.alpha_vantage_key)
        self.news_client = NewsApiClient(api_key=self.newsapi_key)
        self.sentiment_agent = SentimentAnalysisAgent()
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def collect_stock_data(self, ticker, from_date=None, to_date=None, sources=None):

        """
        Fetch stock fundamentals, news data, and analyze sentiment.

        Parameters:
        - ticker (str): Stock ticker symbol.
        - from_date (str): Start date for news filtering (YYYY-MM-DD).
        - to_date (str): End date for news filtering (YYYY-MM-DD).
        - sources (str): Comma-separated list of publishers to filter news (optional).
        """
        # Check if cached data exists
        cache_file = os.path.join(self.cache_dir, f"{ticker}_data.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                return json.load(f)

        try:
            # Fetch company overview
            overview, _ = self.fd.get_company_overview(ticker)
            print("Company Overview", self.fd.get_company_overview(ticker)) 

            # Fetch historical trends (last 30 days)
            historical_data, _ = self.ts.get_daily_adjusted(symbol=ticker, outputsize="compact")

            # Fetch and analyze news data
            print("_fetch_news") 
            news_data = self._fetch_news(ticker, from_date, to_date, sources)

            for article in news_data:
                article["sentiment"] = self.sentiment_agent.analyze_sentiment(article["description"] or article["title"])

            # Combine results
            result = {
                "overview": overview,
                "historical_data": historical_data,
                "news_data": news_data,
            }

            # Save to cache
            with open(cache_file, "w") as f:
                json.dump(result, f)

            return result
        except Exception as e:
            return {"error": str(e)}

    def _fetch_news(self, ticker, from_date=None, to_date=None, sources=None):
        """
        Fetch relevant news articles for the stock ticker.

        Parameters:
        - ticker (str): Stock ticker symbol.
        - from_date (str): Start date for news filtering (YYYY-MM-DD).
        - to_date (str): End date for news filtering (YYYY-MM-DD).
        - sources (str): Comma-separated list of publishers to filter news (optional).

        Returns:
        - list: News articles with formatted dates.
        """
        try:
            params = {
                "q": ticker,
                "language": "en",
                "sort_by": "relevancy",
                "page_size": 20,
                "from_param": from_date,
                "to": to_date,
            }
            if sources:
                params["sources"] = sources

            response = self.news_client.get_everything(**params)
            articles = response.get("articles", [])

            # Format published dates
            formatted_articles = []
            for article in articles:
                published_date = article.get("publishedAt")
                if published_date:
                    try:
                        formatted_date = datetime.datetime.strptime(published_date, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")
                        article["published_date"] = formatted_date
                    except ValueError:
                        article["published_date"] = "Unknown"

                formatted_articles.append({
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "url": article.get("url"),
                    "published_date": article.get("published_date"),
                })

            return formatted_articles
        except Exception as e:
            return {"error": f"Failed to fetch news: {str(e)}"}
