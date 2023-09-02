import yfinance as yf
from datetime import datetime, timedelta

# Function to fetch stock information


def fetch_stock_info(stock_symbol):
    try:
        # Fetch historical data for the stock
        stock = yf.Ticker(stock_symbol)

        # Calculate the percentage return for 2 days
        end_date_2_days = datetime.now()
        start_date_2_days = end_date_2_days - timedelta(days=2)
        historical_data_2_days = stock.history(
            start=start_date_2_days, end=end_date_2_days)
        start_price_2_days = historical_data_2_days.iloc[0]['Close']
        end_price_2_days = historical_data_2_days.iloc[-1]['Close']
        percentage_return_2_days = (
            (end_price_2_days - start_price_2_days) / start_price_2_days) * 100

        # Calculate the percentage return for 1 week (7 days)
        end_date_1_week = datetime.now()
        start_date_1_week = end_date_1_week - timedelta(days=7)
        historical_data_1_week = stock.history(
            start=start_date_1_week, end=end_date_1_week)
        start_price_1_week = historical_data_1_week.iloc[0]['Close']
        end_price_1_week = historical_data_1_week.iloc[-1]['Close']
        percentage_return_1_week = (
            (end_price_1_week - start_price_1_week) / start_price_1_week) * 100

        # Calculate the percentage return for 10 days
        end_date_10_days = datetime.now()
        start_date_10_days = end_date_10_days - timedelta(days=10)
        historical_data_10_days = stock.history(
            start=start_date_10_days, end=end_date_10_days)
        start_price_10_days = historical_data_10_days.iloc[0]['Close']
        end_price_10_days = historical_data_10_days.iloc[-1]['Close']
        percentage_return_10_days = (
            (end_price_10_days - start_price_10_days) / start_price_10_days) * 100

        # Calculate the percentage return for 1 month (30 days)
        end_date_1_month = datetime.now()
        start_date_1_month = end_date_1_month - timedelta(days=30)
        historical_data_1_month = stock.history(
            start=start_date_1_month, end=end_date_1_month)
        start_price_1_month = historical_data_1_month.iloc[0]['Close']
        end_price_1_month = historical_data_1_month.iloc[-1]['Close']
        percentage_return_1_month = (
            (end_price_1_month - start_price_1_month) / start_price_1_month) * 100

        return {
            "Symbol": stock_symbol,
            "Percentage Return (2 Days)": percentage_return_2_days,
            "Percentage Return (1 Week)": percentage_return_1_week,
            "Percentage Return (10 Days)": percentage_return_10_days,
            "Percentage Return (1 Month)": percentage_return_1_month
        }

    except Exception as e:
        return {"Error": str(e)}
# TATAMOTORS.NS

stock_symbol = input("Enter the stock symbol : ")
stock_info = fetch_stock_info(stock_symbol)
if "Error" in stock_info:
    print(f"Error: {stock_info['Error']}")
else:
    print(f"Symbol: {stock_info['Symbol']}")
    print(
        f"Percentage Return (2 Days): {stock_info['Percentage Return (2 Days)']:.2f}%")
    print(
        f"Percentage Return (1 Week): {stock_info['Percentage Return (1 Week)']:.2f}%")
    print(
        f"Percentage Return (10 Days): {stock_info['Percentage Return (10 Days)']:.2f}%")
    print(
        f"Percentage Return (1 Month): {stock_info['Percentage Return (1 Month)']:.2f}%")
