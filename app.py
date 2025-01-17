import streamlit as st
from datetime import date
from utils.caching import fetch_stock_data, filter_articles_by_score, fetch_company_overview
from utils.visualization import plot_sentiment_distribution, plot_trends
from agents.data_collection import DataCollectionAgent
from agents.sentiment_analysis import SentimentAnalysisAgent
from utils.predictions import forecast_with_model
import matplotlib.pyplot as plt
from openai import OpenAI 
import re
import json

import os
from dotenv import load_dotenv

# load data
with open("mock_stock_data.json", "r") as file:
    stock_data_mock = json.load(file)

# Initialize agents
data_agent = DataCollectionAgent()
sentiment_agent = SentimentAnalysisAgent()

# Title
st.title("Agentic AI: Enhanced Stock Research Assistant")

# Create input controls
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    ticker = st.text_input("Enter Stock Ticker", placeholder="e.g., AAPL, TSLA")
    market = st.selectbox("Select Market", options=["NASDAQ", "NYSE", "ASX", "LSE"])

with col2:
    start_date = st.date_input("Start Date", value=date(2023, 1, 1))
    end_date = st.date_input("End Date", value=date.today())

with col3:
    publisher = st.text_input("Filter by Publisher", placeholder="e.g., techcrunch")
    keywords = st.multiselect(
        "Key Insights by Keywords",
        options=["valuation", "revenue", "profit", "growth", "market cap"],
        default=["valuation"]
    )

# Add additional inputs
search_query = st.text_input("Search Articles", placeholder="e.g., revenue, growth, costs")
score_threshold = st.slider(
    "Filter Articles by Sentiment Score",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.01
)

# Fetch stock data on submission
if st.button("Submit"):
    # Validate if a ticker has been entered
    if not ticker.strip():
        st.error("Please enter a stock ticker to proceed.")
    else:
        placeholder = st.empty()
        placeholder.write(f"Analyzing {ticker} in {market} market...")
        
        # Fetch and cache stock data
        try:
            stock_data = fetch_stock_data(data_agent, ticker, start_date, end_date, sources=publisher)
        except Exception as e:
            st.error(f"Failed to fetch stock data: {e}")
            stock_data = {}

        if "error" in stock_data:
            placeholder.error(f"Error: {stock_data['error']}")
        else:
            placeholder.success("Stock data fetched successfully!")

            # Clear the placeholder after a brief pause
            import time
            time.sleep(10)  # Optional: Add a small delay to show success message
            placeholder.empty()  # Clear the placeholder

            # Tabs for navigation
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Recent News", "Sentiment Analysis", "Financial Trends", "Predictions"])

            # Inside the "Overview" tab
            with tab1: 
                st.header("Company Overview")
                
                # Fetch and display company overview
                
                company_overview = fetch_company_overview(data_agent, ticker)
                
                if "error" in company_overview:
                    st.error(company_overview["error"])
                else:
                    # Streamlit code to display grouped overview data with each group in its own expander
                    

                    # Helper function to display metrics in columns with dynamic font sizes
                    def display_metrics_in_columns(data_dict, columns=3):
                        keys = list(data_dict.keys())
                        values = list(data_dict.values())

                        for i in range(0, len(keys), columns):
                            col_group = st.columns(columns)
                            for j, col in enumerate(col_group):
                                if i + j < len(keys):
                                    key = keys[i + j]
                                    value = values[i + j]
                                    
                                    # Adjust font size based on the length of the value
                                    font_size = "1.5rem"
                                    if len(str(value)) > 10:
                                        font_size = "1.2rem"
                                    if len(str(value)) > 15:
                                        font_size = "1rem"

                                    # Custom HTML for dynamic font size
                                    col.markdown(
                                        f"""
                                        <div style="text-align: center;">
                                            <p style="margin: 0; font-size: 1rem; font-weight: bold;">{key}</p>
                                            <p style="margin: 0; font-size: {font_size}; color: #2E7D32;">{value}</p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )


                    # General Information
                    with st.expander("General Information"):
                        general_info = {
                            "Symbol": company_overview.get("Symbol", "N/A"),
                            "Asset Type": company_overview.get("AssetType", "N/A"),
                            "Name": company_overview.get("Name", "N/A"),
                            "Exchange": company_overview.get("Exchange", "N/A"),
                            "Currency": company_overview.get("Currency", "N/A"),
                            "Country": company_overview.get("Country", "N/A"),
                            "Address": company_overview.get("Address", "N/A"),
                            "Official Website": company_overview.get("OfficialSite", "N/A"),
                        }
                        # Display the other metrics in columns
                        display_metrics_in_columns(general_info)

                        # Add the Description field as a full-width section
                        description = company_overview.get("Description", "N/A")
                        st.markdown(
                            f"""
                            <div style="padding: 10px; margin-top: 20px; border-top: 1px solid #ddd;">
                                <h5 style="margin: 0; font-weight: bold;">Description</h5>
                                <p style="margin: 0; font-size: 1rem; text-align: justify; color: #2E7D32;">{description}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Financial Performance
                    with st.expander("Financial Performance"):
                        financial_performance = {
                            "Market Capitalization": f"${int(company_overview.get('MarketCapitalization', 0)):,}",
                            "EBITDA": f"${int(company_overview.get('EBITDA', 0)):,}",
                            "Revenue (TTM)": f"${int(company_overview.get('RevenueTTM', 0)):,}",
                            "Gross Profit (TTM)": f"${int(company_overview.get('GrossProfitTTM', 0)):,}",
                            "EPS (Diluted, TTM)": company_overview.get("DilutedEPSTTM", "N/A"),
                            "Revenue per Share (TTM)": company_overview.get("RevenuePerShareTTM", "N/A"),
                            "Profit Margin": f"{float(company_overview.get('ProfitMargin', 0)):.2%}",
                            "Operating Margin (TTM)": f"{float(company_overview.get('OperatingMarginTTM', 0)):.2%}",
                            "Return on Assets (TTM)": f"{float(company_overview.get('ReturnOnAssetsTTM', 0)):.2%}",
                            "Return on Equity (TTM)": f"{float(company_overview.get('ReturnOnEquityTTM', 0)):.2%}",
                        }
                        display_metrics_in_columns(financial_performance)

                    # Valuation Metrics
                    with st.expander("Valuation Metrics"):
                        valuation_metrics = {
                            "P/E Ratio": company_overview.get("PERatio", "N/A"),
                            "PEG Ratio": company_overview.get("PEGRatio", "N/A"),
                            "Book Value": company_overview.get("BookValue", "N/A"),
                            "Price to Sales Ratio (TTM)": company_overview.get("PriceToSalesRatioTTM", "N/A"),
                            "Price to Book Ratio": company_overview.get("PriceToBookRatio", "N/A"),
                            "EV to Revenue": company_overview.get("EVToRevenue", "N/A"),
                            "EV to EBITDA": company_overview.get("EVToEBITDA", "N/A"),
                        }
                        display_metrics_in_columns(valuation_metrics)

                    # Dividends and Shares
                    with st.expander("Dividends and Shares"):
                        dividends_shares = {
                            "Dividend per Share": company_overview.get("DividendPerShare", "N/A"),
                            "Dividend Yield": f"{float(company_overview.get('DividendYield', 0)):.2%}",
                            "Dividend Date": company_overview.get("DividendDate", "N/A"),
                            "Ex-Dividend Date": company_overview.get("ExDividendDate", "N/A"),
                            "Shares Outstanding": f"{int(company_overview.get('SharesOutstanding', 0)):,}",
                        }
                        display_metrics_in_columns(dividends_shares)

                    # Stock Performance
                    with st.expander("Stock Performance"):
                        stock_performance = {
                            "Beta": company_overview.get("Beta", "N/A"),
                            "52-Week High": f"${float(company_overview.get('52WeekHigh', 0)):.2f}",
                            "52-Week Low": f"${float(company_overview.get('52WeekLow', 0)):.2f}",
                            "50-Day Moving Average": f"${float(company_overview.get('50DayMovingAverage', 0)):.2f}",
                            "200-Day Moving Average": f"${float(company_overview.get('200DayMovingAverage', 0)):.2f}",
                        }
                        display_metrics_in_columns(stock_performance)

                    # Analyst Ratings
                    with st.expander("Analyst Ratings"):
                        analyst_ratings = {
                            "Target Price": f"${float(company_overview.get('AnalystTargetPrice', 0)):.2f}",
                            "Strong Buy Ratings": company_overview.get("AnalystRatingStrongBuy", "N/A"),
                            "Buy Ratings": company_overview.get("AnalystRatingBuy", "N/A"),
                            "Hold Ratings": company_overview.get("AnalystRatingHold", "N/A"),
                            "Sell Ratings": company_overview.get("AnalystRatingSell", "N/A"),
                            "Strong Sell Ratings": company_overview.get("AnalystRatingStrongSell", "N/A"),
                        }
                        display_metrics_in_columns(analyst_ratings)



            with tab2:
                st.header("Recent News")

                # Process and display news articles with sentiment analysis
                news_data = stock_data.get("news_data", [])
                sentiment_results = []  # Initialize sentiment_results

                if news_data:
                    try:
                        # Perform sentiment analysis
                        sentiment_results = sentiment_agent.analyze(news_data)
                    except ValueError as e:
                        st.error(f"Sentiment analysis error: {e}")

                    for article in sentiment_results:
                        # Article Title
                        st.markdown(f"**[{article.get('title', 'No Title')}]({article.get('url', '#')})**")

                        # Article Description
                        st.write(article.get("description", "No description available."))

                        # Safely Access Source and Published Date
                        source = article.get("source", {})
                        source_name = source.get("name", "Unknown Source") if isinstance(source, dict) else "Unknown Source"
                        published_date = article.get("publishedAt", "Unknown Date")
                        st.write(f"Source: {source_name} | Published Date: {published_date}")

                        # Display Sentiment Information
                        sentiment = article.get("sentiment", {})
                        if isinstance(sentiment, dict):
                            label = sentiment.get("label", "Neutral")
                            score = sentiment.get("score", 0.0)
                            st.write(f"Sentiment: {label} (Score: {score:.2f})")
                        else:
                            st.write("Sentiment: Neutral (Score: 0.0)")  # Fallback if sentiment is not a dictionary

                        st.write("---")
                else:
                    st.warning("No news data available.")

                # Filter articles based on the selected threshold
                filtered_sentiment_results = [
                    article for article in sentiment_results
                    if abs(article.get("sentiment", {}).get("score", 0.0)) >= score_threshold
                ]

                # Display filtered and searched results
                st.subheader(f"Filtered Articles (Score ≥ {score_threshold:.2f})")
                # filtered_articles = filter_articles_by_score(news_data, score_threshold)
                for article in filtered_sentiment_results:
                    st.markdown(f"**[{article.get('title', 'No Title')}]({article.get('url', '#')})**")
                    st.write(article.get("description", "No description available."))
                    st.write(f"Sentiment: {article.get('sentiment', {}).get('label', 'Neutral')} "
                            f"(Score: {article.get('sentiment', {}).get('score', 0.0):.2f})")
                    st.write("---")

                # Filter articles based on the search query
                search_results = [
                    article for article in sentiment_results
                    if search_query.lower() in article.get("title", "").lower() or search_query.lower() in article.get("description", "").lower()
                ]

                # Display search results
                st.subheader(f"Search Results for: '{search_query}'")
                if search_query:
                    if search_results:
                        for article in search_results:
                            st.markdown(f"**[{article.get('title', 'No Title')}]({article.get('url', '#')})**")
                            st.write(f"Sentiment: {article.get('sentiment', {}).get('label', 'Unknown')} "
                                    f"(Score: {article.get('sentiment', {}).get('score', 0.0):.2f})")
                            source_name = article.get("source", {}).get("name", "Unknown Source")
                            published_date = article.get("publishedAt", "Unknown Date")
                            st.write(f"Source: {source_name} | Published Date: {published_date}")
                            st.write("---")
                    else:
                        st.warning(f"No articles found matching '{search_query}'.")
                else:
                    st.info("Enter a keyword above to search articles.")



            # Sentiment Analysis Tab
            with tab3:
                st.header("Sentiment Analysis")

                if news_data:
                    try:
                        # Perform sentiment analysis
                        sentiment_results = sentiment_agent.analyze(news_data)
                        grouped_sentiment = sentiment_agent.group_by_sentiment(sentiment_results)

                        # Sentiment Distribution Pie Chart
                        st.subheader("Sentiment Distribution")
                        labels = ["Positive", "Negative", "Neutral"]
                        sizes = [
                            len(grouped_sentiment.get("Positive", [])),
                            len(grouped_sentiment.get("Negative", [])),
                            len(grouped_sentiment.get("Neutral", []))
                        ]

                        fig, ax = plt.subplots()
                        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
                        ax.axis("equal")
                        st.pyplot(fig)

                        # Historical Stock Performance
                        st.subheader("Historical Stock Performance")
                        historical_data = stock_data.get("historical_data", {})
                        if historical_data:
                            dates = list(historical_data.keys())
                            closing_prices = [float(data.get("5. adjusted close", 0)) for data in historical_data.values()]

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

                        # Sentiment Distribution Bar Chart
                        st.subheader("Sentiment Distribution")
                        fig, ax = plt.subplots(figsize=(7, 4))
                        ax.bar(labels, sizes, color=["green", "red", "gray"])
                        ax.set_title("Sentiment Distribution")
                        ax.set_ylabel("Number of Articles")
                        st.pyplot(fig)

                        # Sentiment Diverging Bar Chart
                        st.subheader("Sentiment Score Distribution")
                        positive_score = sum(
                            article["sentiment"]["score"]
                            for article in sentiment_results if article["sentiment"]["label"] == "Positive"
                        )
                        negative_score = sum(
                            article["sentiment"]["score"]
                            for article in sentiment_results if article["sentiment"]["label"] == "Negative"
                        )
                        neutral_score = sum(
                            article["sentiment"]["score"]
                            for article in sentiment_results if article["sentiment"]["label"] == "Neutral"
                        )

                        sentiment_scores = [positive_score, -negative_score, neutral_score]
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.bar(labels, sentiment_scores, color=["green", "red", "gray"])
                        ax.axhline(0, color="black", linewidth=0.8)
                        ax.set_xlabel("Sentiment")
                        ax.set_ylabel("Aggregated Score")
                        ax.set_title("Diverging Sentiment Scores")
                        st.pyplot(fig)

                        # Scatter Plot of Sentiment Scores
                        st.subheader("Sentiment Score Scatter Plot")
                        dates = [article["publishedAt"] for article in sentiment_results]
                        scores = [
                            article["sentiment"]["score"] if article["sentiment"]["label"] == "Positive" else
                            -article["sentiment"]["score"] if article["sentiment"]["label"] == "Negative" else 0
                            for article in sentiment_results
                        ]
                        colors = ["green" if score > 0 else "red" if score < 0 else "gray" for score in scores]

                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.scatter(dates, scores, c=colors, alpha=0.6)
                        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Sentiment Score")
                        ax.set_title("Sentiment Score Scatter Plot")
                        plt.xticks(rotation=45)
                        st.pyplot(fig)

                        # Combined Sentiment and Frequency Chart
                        st.subheader("Sentiment Analysis Overview")
                        positive_scores = [
                            article["sentiment"]["score"] for article in sentiment_results if article["sentiment"]["label"] == "Positive"
                        ]
                        negative_scores = [
                            article["sentiment"]["score"] for article in sentiment_results if article["sentiment"]["label"] == "Negative"
                        ]
                        neutral_scores = [
                            article["sentiment"]["score"] for article in sentiment_results if article["sentiment"]["label"] == "Neutral"
                        ]
                        frequencies = [len(positive_scores), len(negative_scores), len(neutral_scores)]
                        average_scores = [
                            sum(positive_scores) / len(positive_scores) if positive_scores else 0,
                            -sum(negative_scores) / len(negative_scores) if negative_scores else 0,
                            sum(neutral_scores) / len(neutral_scores) if neutral_scores else 0
                        ]

                        x = range(len(labels))
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.bar(x, frequencies, width=0.4, label="Frequency", align="center", color="lightblue")
                        ax.bar(x, average_scores, width=0.4, label="Avg Sentiment Score", align="edge", color=["green", "red", "gray"])
                        ax.axhline(0, color="black", linewidth=0.8)
                        ax.set_xticks(x)
                        ax.set_xticklabels(labels)
                        ax.set_xlabel("Sentiment")
                        ax.set_ylabel("Value")
                        ax.set_title("Sentiment Analysis Overview")
                        ax.legend()
                        st.pyplot(fig)


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

                    except Exception as e:
                        st.error(f"Error during sentiment analysis or visualization: {e}")

                else:
                    st.warning("No news data available for sentiment analysis.")
                    

            # Financial Trends Tab
            with tab4:
                st.header("Financial Trends")

                # Example: Cost Trends
                st.subheader("Cost Trends")
                cost_data = stock_data.get("overview", {}).get("cost_data", {})
                plot_trends(
                    cost_data,
                    title="Cost Trends",
                    xlabel="Date",
                    ylabel="Cost"
                )

                # Example: Revenue Growth
                st.subheader("Revenue Growth Trends")
                revenue_data = stock_data.get("overview", {}).get("revenue_growth", {})
                plot_trends(
                    revenue_data,
                    title="Revenue Growth",
                    xlabel="Date",
                    ylabel="Revenue"
                )

            # Predictions Tab
            with tab5:
                st.header("Predictions")
                model_choice = st.selectbox("Choose Forecasting Model", ["Prophet", "ARIMA"])

                try:
                    historical_data = stock_data.get("historical_data", {})
                    forecast_data = forecast_with_model(historical_data, model_choice)
                    st.line_chart(forecast_data.set_index("ds")["yhat"])
                except Exception as e:
                    st.error(f"Error generating forecast: {e}")
