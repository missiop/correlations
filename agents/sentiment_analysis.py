import torch
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OpenAISentimentAnalysisAgent:
    def __init__(self):
        # Load OpenAI API key
        self.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key

    def analyze_with_openai(self, content):
        """
        Use OpenAI's GPT models to analyze sentiment of the content.
        """
        try:
            response = openai.Completion.create(
                model="gpt-4o-mini",  # Use the appropriate GPT model
                prompt=f"Analyze the sentiment of the following text and explain your reasoning:\n\n{content}",
                max_tokens=150,
                temperature=0.7,
            )
            sentiment_analysis = response.choices[0].text.strip()
            return sentiment_analysis
        except Exception as e:
            print(f"Error with OpenAI API: {e}")
            return None


class SentimentAnalysisAgent:
    def __init__(self):

        """
        Initialize the sentiment analysis agent.
        """
        # Check if PyTorch is available
        print(f"PyTorch available: {torch.cuda.is_available()}")
        
        # Initialize Hugging Face sentiment analysis pipeline
        # distilbert-base-uncased
        # takala/financial_phrasebank
        # yiyanghkust/finbert
        # gpt4o
        # Access the Hugging Face token
        
        huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        self.sentiment_model = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        self.label_map = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}  # Adjust based on FinBERT's outputs
  

    @staticmethod
    def fetch_full_article(url):
        """
        Fetch the full article content from the provided URL.
        """
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                paragraphs = soup.find_all("p")  # Extract text from <p> tags
                full_text = " ".join([p.text for p in paragraphs])
                return full_text
            else:
                print(f"Failed to fetch article. HTTP status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching article: {e}")
            return None

    def analyze(self, news_data, max_length=512):
        """
        Analyze the sentiment of news articles, handling cases where content exceeds model token limits.
        """
        if not isinstance(news_data, list):
            raise ValueError("Expected news_data to be a list of dictionaries")

        sentiment_results = []

        for article in news_data:
            # Retrieve article fields
            title = article.get("title", "No Title")
            description = article.get("description", "No Description")
            url = article.get("url", "")
            source = article.get("source", {})  # Preserve source field
            published_at = article.get("publishedAt", "Unknown Date")  # Preserve publishedAt field
            content = f"{title}. {description}"

            # Fetch full article text if available
            full_text = ""
            if url:
                try:
                    full_text = self.fetch_full_article(url)
                except Exception as e:
                    st.error(f"Error fetching full text for URL '{url}': {e}")
            if full_text:
                content += f" {full_text}"

            # Split content into chunks
            content_chunks = [content[i:i + max_length] for i in range(0, len(content), max_length)]
            if not content_chunks:
                sentiment_results.append({
                    "title": title,
                    "description": description,
                    "url": url,
                    "source": source,
                    "publishedAt": published_at,
                    "sentiment": {"label": "Neutral", "score": 0.0},
                })
                st.warning(f"No analyzable content for article: {title}")
                continue

            # Analyze each chunk and aggregate sentiment
            chunk_sentiments = []
            for chunk in content_chunks:
                try:
                    sentiment = self.sentiment_model(chunk)[0]
                    chunk_sentiments.append(sentiment)
                except Exception as e:
                    chunk_sentiments.append({"label": "Neutral", "score": 0.0})  # Return dictionary format
                    st.warning(f"Error analyzing chunk for article '{title}': {e}")

            # Aggregate sentiment scores (simple average)
            avg_score = sum(chunk["score"] for chunk in chunk_sentiments) / len(chunk_sentiments)
            avg_label = max(chunk_sentiments, key=lambda x: x["score"])["label"]

            # Map label to readable format
            label = self.label_map.get(avg_label, "Neutral")

            # Append result with additional fields
            sentiment_results.append({
                "title": title,
                "description": description,
                "url": url,
                "source": source,  # Preserve original source field
                "publishedAt": published_at,  # Preserve original publishedAt field
                "sentiment": {"label": label, "score": avg_score},  # Return dictionary format
            })

        return sentiment_results



    def group_by_sentiment(self, sentiment_results):
        """
        Group news articles by sentiment category.

        Parameters:
        - sentiment_results (list): Results from the analyze() method.

        Returns:
        - Dictionary of sentiment categories and their associated article titles.
        """
        grouped = {"Positive": [], "Negative": [], "Neutral": []}

        for result in sentiment_results:
            sentiment = result.get("sentiment", {})
            if isinstance(sentiment, dict):  # Ensure sentiment is a dictionary
                label = sentiment.get("label", "Neutral")
                grouped[label].append(result)  # Append the entire article object
            else:
                grouped["Neutral"].append(result)  # Fallback if sentiment is not a dictionary

        return grouped




    def summarize_sentiment(self, sentiment_results):
        """
        Summarize sentiment analysis.

        Parameters:
        - sentiment_results (list): Results from the analyze() method.

        Returns:
        - Dictionary summarizing sentiment distribution and average score.
        """
        total = len(sentiment_results)
        if total == 0:
            return {"Positive": 0, "Negative": 0, "Neutral": 0, "Average Score": 0}

        summary = {"Positive": 0, "Negative": 0, "Neutral": 0, "Average Score": 0}
        for result in sentiment_results:
            summary[result["sentiment"]] += 1
            summary["Average Score"] += result["score"]

        summary["Average Score"] /= total
        summary = {k: (v / total * 100 if k != "Average Score" else v) for k, v in summary.items()}
        return summary

    def highlight_insights(self, sentiment_results, keyword="revenue"):
        """
        Highlight key insights based on a keyword and sentiment scores.

        Parameters:
        - sentiment_results (list): Results from the analyze() method.
        - keyword (str): Keyword to filter articles (default: "revenue").

        Returns:
        - List of key articles sorted by sentiment score.
        """
        key_articles = [res for res in sentiment_results if keyword.lower() in res["description"].lower()]
        return sorted(key_articles, key=lambda x: x["score"], reverse=True)


# Example Usage
if __name__ == "__main__":
    # Example news data from DataCollectionAgent
    example_news_data = [
        {"title": "Company reports record-breaking revenue growth", "description": "Profits soared by 25% in Q4."},
        {"title": "Rising steel prices impact construction companies", "description": "Steel prices increased by 30% this year."},
    ]

    sentiment_agent = SentimentAnalysisAgent()
    results = sentiment_agent.analyze(example_news_data)

    for result in results:
        print(result)
