class ReportGenerationAgent:
    def generate(self, ticker, market, fundamentals, sentiment, competition, risks):
        report = f"Report for {ticker} ({market})\n\n"
        report += "### Fundamental Analysis\n"
        report += f"Revenue Drivers: {fundamentals['revenue']}\n"
        report += f"Cost Drivers: {fundamentals['costs']}\n\n"
        
        report += "### Sentiment Analysis\n"
        for item in sentiment:
            report += f"- {item['title']} (Polarity: {item['polarity']}, Confidence: {item['confidence']:.2f})\n"
            report += f"  Context: {', '.join(item['context'])}\n"
        
        report += "\n### Competitive Analysis\n"
        report += f"Peers: {competition}\n\n"

        report += "### Risks\n"
        for risk in risks:
            report += f"- {risk}\n"

        return report
