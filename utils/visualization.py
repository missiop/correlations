import matplotlib.pyplot as plt
import streamlit as st

def plot_sentiment_distribution(grouped_sentiment):
    """
    Creates a pie chart of sentiment distribution.
    """
    labels = grouped_sentiment.keys()
    sizes = [len(grouped_sentiment[label]) for label in labels]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

def plot_trends(data, title, xlabel, ylabel):
    """
    Plots trends for stock performance, revenue growth, or cost trends.

    Parameters:
    - data (dict): Contains 'dates' and 'values'.
    - title (str): Title of the chart.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    """
    dates = data.get("dates", [])
    values = data.get("values", [])

    if not dates or not values:
        st.warning(f"No data available for {title}.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, values, marker="o", linestyle="-")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
