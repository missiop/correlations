import streamlit as st
import json
from agents.data_collection import DataCollectionAgent
from agents.sentiment_analysis import SentimentAnalysisAgent
import datetime
import matplotlib.pyplot as plt
from openai import OpenAI 
import re

import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()
client = OpenAI()

# load data
with open("mock_stock_data.json", "r") as file:
    stock_data_mock = json.load(file)

# Initialize agents
data_agent = DataCollectionAgent()
sentiment_agent = SentimentAnalysisAgent()

st.title("Agentic AI: Enhanced Stock Research Assistant")

# Input fields
ticker = st.text_input("Enter Stock Ticker", placeholder="e.g., AAPL, TSLA")
market = st.selectbox("Select Market", options=["NASDAQ", "NYSE", "ASX", "LSE"])

# Add date range inputs
start_date = st.date_input("Start Date", value=datetime.date(2023, 1, 1))
end_date = st.date_input("End Date", value=datetime.date.today())

# Add publisher filter
publisher = st.text_input("Filter by Publisher (e.g., techcrunch)", placeholder="Leave blank for all publishers")

# Add the "Filter Key Insights by Keywords" input before processing
keywords = st.multiselect(
    "Filter Key Insights by Keywords",
    options=["valuation", "revenue", "profit", "growth", "market cap"],
    default=["valuation"]
)

# Add a search bar to filter articles by keywords
st.subheader("Search Articles by Keyword")
search_query = st.text_input("Enter a keyword to search in articles", placeholder="e.g., revenue, growth, costs")


# Add a slider to set sentiment score threshold
st.subheader("Filter Articles by Sentiment Score")
score_threshold = st.slider(
    "Select minimum sentiment score:",
    min_value=0.0,
    max_value=1.0,
    value=0.75,  # Default threshold
    step=0.01
)

# Fetch and display stock data when the Submit button is pressed
if st.button("Submit"):
    st.write(f"Analyzing {ticker} in {market} market...")


    # Call DataCollectionAgent with user inputs
    if "stock_data" not in st.session_state:
        st.session_state["stock_data"] = data_agent.collect_stock_data(
            ticker=ticker,
            from_date=start_date.strftime("%Y-%m-%d"),
            to_date=end_date.strftime("%Y-%m-%d"),
            sources=publisher if publisher else None
        ) 
    stock_data = st.session_state["stock_data"]

    # Use loaded mock data instead of fetching live data
    # Example: Display cost trends
    cost_data = stock_data_mock["overview"]["cost_data"]


    # Tabs for Navigation
    tab1, tab2, tab3 = st.tabs(["Overview", "Sentiment Analysis", "Predictions"])

    # Overview Tab
    with tab1:
        st.header("Overview")
        st.write("Stock details and recent news go here...")


        if "error" in stock_data:
            st.error(f"Error: {stock_data['error']}")
        else:
            st.success("Stock data fetched successfully!")

            # Display basic stock information
            st.subheader("Stock Overview")
            #st.json(stock_data.get("overview", {}))

            # Display recent news
            st.subheader("Recent News")
            news_data = stock_data.get("news_data", [])
            if news_data:
                for article in news_data:
                    st.markdown(f"**[{article['title']}]({article['url']})**")
                    st.write(article["description"])
                    # st.write(f"Published Date: {article['publishedAt']}")
                    
                    st.write(f"Source: {article['source']['name']} | Published Date: {article['publishedAt']}")
                    st.write(f"Author: {article['author']}")
            else:
                st.warning("No news data available for the selected criteria.")

            st.write("---")

            # Perform sentiment analysis
            st.subheader("Sentiment Analysis")
            sentiment_results = sentiment_agent.analyze(news_data)

            grouped_sentiment = sentiment_agent.group_by_sentiment(sentiment_results)
            sentiment_summary = sentiment_agent.summarize_sentiment(sentiment_results)

            for article in sentiment_results:
                st.write(f"Title: {article['title']}")
                st.write(f"Sentiment: {article['sentiment']} (Score: {article['score']:.2f})")
                st.markdown(f"[Read More]({article['url']})")
                st.write("---")


    # Sentiment Analysis Tab
    with tab2:
        st.header("Sentiment Analysis")
        with st.expander("Sentiment Filtering"):
            st.write("Add filtering options here...")
        st.write("Sentiment scores and distribution charts go here...")


        # Sentiment Summary
        st.markdown("### Sentiment Summary")
        st.write(sentiment_summary)

        # Positive Articles
        st.markdown("### Positive Articles")
        for article in grouped_sentiment["Positive"]:
            st.markdown(f"- [{article['title']}]({article['url']})")

        # Negative Articles
        st.markdown("### Negative Articles")
        for article in grouped_sentiment["Negative"]:
            st.markdown(f"- [{article['title']}]({article['url']})")

        # Neutral Articles
        st.markdown("### Neutral Articles")
        for article in grouped_sentiment["Neutral"]:
            st.markdown(f"- [{article['title']}]({article['url']})")

        # Generate AI Summary
        try:
            summary = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Summarize the following sentiment analysis results:\n\n" + str(sentiment_results)}
                ],
                max_tokens=300,
                temperature=0.7
            ).choices[0].message.content 
            st.subheader("AI-Generated Summary")
            st.write(summary)
        except Exception as e:
            st.error(f"Error generating AI summary: {str(e)}")

        # Key Insights
        for keyword in keywords:
            st.markdown(f"### Key Insights (Keyword: '{keyword}')")
            key_insights = sentiment_agent.highlight_insights(sentiment_results, keyword=keyword)
            for insight in key_insights:
                st.write(f"- {insight['title']} (Score: {insight['score']:.2f}) [Read more]({insight['url']})")


        # Filter articles based on the selected threshold
        filtered_sentiment_results = [
            article for article in sentiment_results
            if abs(article["score"]) >= score_threshold  # Use absolute value to include both positive and negative scores
        ]

        # Display filtered results
        st.subheader(f"Articles with Sentiment Scores ≥ {score_threshold:.2f}")
        if filtered_sentiment_results:
            for article in filtered_sentiment_results:
                st.markdown(f"**[{article.get('title', 'No Title')}]({article.get('url', '#')})**")
                st.write(f"Sentiment: {article.get('sentiment', 'Unknown')} (Score: {article.get('score', 0.0):.2f})")
                
                # Handle source and published date safely
                source_name = article.get("source", {}).get("name", "Unknown Source")
                published_date = article.get("publishedAt", "Unknown Date")
                st.write(f"Source: {source_name} | Published Date: {published_date}")
                st.write("---")
        else:
            st.warning("No articles meet the selected sentiment score threshold.")


        # Filter articles based on the search query
        search_results = [
            article for article in filtered_sentiment_results
            if search_query.lower() in article.get("title", "").lower() or search_query.lower() in article.get("description", "").lower()
        ]

        # Display search results
        st.subheader(f"Search Results for: '{search_query}'")
        if search_query:
            if search_results:
                for article in search_results:
                    st.markdown(f"**[{article.get('title', 'No Title')}]({article.get('url', '#')})**")
                    st.write(f"Sentiment: {article.get('sentiment', 'Unknown')} (Score: {article.get('score', 0.0):.2f})")
                    source_name = article.get("source", {}).get("name", "Unknown Source")
                    published_date = article.get("publishedAt", "Unknown Date")
                    st.write(f"Source: {source_name} | Published Date: {published_date}")
                    st.write("---")
            else:
                st.warning(f"No articles found matching '{search_query}'.")
        else:
            st.info("Enter a keyword above to search articles.")

        # Historical Trends
        st.subheader("Historical Trends")
        # st.json(stock_data.get("historical_data", {}))

        # Sentiment distribution chart
        st.subheader("Sentiment Distribution")
        labels = ["Positive", "Negative", "Neutral"]
        sizes = [len(grouped_sentiment["Positive"]), len(grouped_sentiment["Negative"]), len(grouped_sentiment["Neutral"])]

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

        st.pyplot(fig)

        # Historical Stock Performance
        st.subheader("Historical Stock Performance")
        historical_data = stock_data.get("historical_data", {})

        if historical_data:
            # Extract dates and closing prices
            dates = list(historical_data.keys())
            closing_prices = [float(data["5. adjusted close"]) for data in historical_data.values()]

            # Plot the historical stock prices
            plt.figure(figsize=(10, 5))
            plt.plot(dates, closing_prices, marker="o", linestyle="-", color="blue")
            plt.xlabel("Date")
            plt.ylabel("Adjusted Closing Price")
            plt.title(f"Stock Performance for {ticker}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(plt)
        else:
            st.warning("No historical stock data available.")


        # Sentiment Distribution
        st.subheader("Sentiment Distribution")
        labels = ["Positive", "Negative", "Neutral"]
        sizes = [len(grouped_sentiment["Positive"]), len(grouped_sentiment["Negative"]), len(grouped_sentiment["Neutral"])]

        # Bar Chart
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, sizes, color=["green", "red", "gray"])
        ax.set_title("Sentiment Distribution")
        ax.set_ylabel("Number of Articles")
        st.pyplot(fig)

        # Cost Trends
        st.subheader("Cost Trends")
        cost_data = stock_data_mock.get("overview", {}).get("cost_data", [])

        if cost_data:
            dates = [entry["date"] for entry in cost_data]
            costs = [entry["cost"] for entry in cost_data]

            # Plot the cost trends
            plt.figure(figsize=(10, 5))
            plt.plot(dates, costs, marker="o", linestyle="-", color="orange")
            plt.xlabel("Date")
            plt.ylabel("Cost")
            plt.title(f"Cost Trends for {ticker}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(plt)
        else:
            st.warning("No cost trend data available.")



        # Revenue Growth Trends
        st.subheader("Revenue Growth Trends")
        revenue_data = stock_data_mock.get("overview", {}).get("revenue_growth", [])

        if revenue_data:
            dates = [entry["date"] for entry in revenue_data]
            revenues = [entry["revenue"] for entry in revenue_data]

            # Plot the revenue growth
            plt.figure(figsize=(10, 5))
            plt.plot(dates, revenues, marker="o", linestyle="-", color="green")
            plt.xlabel("Date")
            plt.ylabel("Revenue")
            plt.title(f"Revenue Growth for {ticker}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(plt)
        else:
            st.warning("No revenue growth data available.")


        # Sentiment Diverging Bar Chart
        st.subheader("Sentiment Score Distribution")

        # Aggregate sentiment scores
        positive_score = sum(article["score"] for article in sentiment_results if article["sentiment"] == "Positive")
        negative_score = sum(article["score"] for article in sentiment_results if article["sentiment"] == "Negative")
        neutral_score = sum(article["score"] for article in sentiment_results if article["sentiment"] == "Neutral")

        sentiment_labels = ["Positive", "Negative", "Neutral"]
        sentiment_scores = [positive_score, -negative_score, neutral_score]  # Negative for negative scores

        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(sentiment_labels, sentiment_scores, color=["green", "red", "gray"])
        ax.axhline(0, color="black", linewidth=0.8)  # Add horizontal line for neutral
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Aggregated Score")
        ax.set_title("Diverging Sentiment Scores")
        st.pyplot(fig)


        # Scatter Plot of Sentiment Scores
        st.subheader("Sentiment Score Scatter Plot")

        # Extract data
        dates = [article["publishedAt"] for article in news_data]
        scores = [
            article["score"] if article["sentiment"] == "Positive" else
            -article["score"] if article["sentiment"] == "Negative" else 0
            for article in sentiment_results
        ]
        colors = ["green" if score > 0 else "red" if score < 0 else "gray" for score in scores]

        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(dates, scores, c=colors, alpha=0.6)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")  # Neutral line
        ax.set_xlabel("Date")
        ax.set_ylabel("Sentiment Score")
        ax.set_title("Sentiment Score Scatter Plot")
        plt.xticks(rotation=45)
        st.pyplot(fig)



        # Combined Sentiment and Frequency Chart
        st.subheader("Sentiment Analysis Overview")

        # Group by sentiment
        positive_scores = [article["score"] for article in sentiment_results if article["sentiment"] == "Positive"]
        negative_scores = [article["score"] for article in sentiment_results if article["sentiment"] == "Negative"]
        neutral_scores = [article["score"] for article in sentiment_results if article["sentiment"] == "Neutral"]

        labels = ["Positive", "Negative", "Neutral"]
        frequencies = [len(positive_scores), len(negative_scores), len(neutral_scores)]
        average_scores = [
            sum(positive_scores) / len(positive_scores) if positive_scores else 0,
            -sum(negative_scores) / len(negative_scores) if negative_scores else 0,  # Negative for alignment
            sum(neutral_scores) / len(neutral_scores) if neutral_scores else 0
        ]

        x = range(len(labels))

        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x, frequencies, width=0.4, label="Frequency", align="center", color="lightblue")
        ax.bar(x, average_scores, width=0.4, label="Avg Sentiment Score", align="edge", color=["green", "red", "gray"])
        ax.axhline(0, color="black", linewidth=0.8)  # Neutral line
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Value")
        ax.set_title("Sentiment Analysis Overview")
        ax.legend()
        st.pyplot(fig)




    # Predictions Tab
    with tab3:
        st.header("Predictions")
        model_choice = st.selectbox("Choose Forecasting Model", ["Prophet", "ARIMA"])
        st.write(f"Predictions using {model_choice} go here...")