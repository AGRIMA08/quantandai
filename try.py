import pandas as pd
import glob
import os
from datetime import datetime, timedelta
txt_files=glob.glob("*.txt")
dfs=[]
for file in txt_files:
    ticker=os.path.splitext(os.path.basename(file))[0]
    df=pd.read_csv(file, sep=',', parse_dates=['Date'])
    df['Ticker']=ticker
    dfs.append(df)
arr=pd.concat(dfs)
arr['Date']=pd.to_datetime(arr['Date'])
arr.set_index(['Ticker', 'Date'], inplace=True)
a=arr.isnull().groupby('Ticker').sum()
print("Missing values per ticker:\n", a)
arr=(arr.groupby(level=0).apply(lambda g: g.interpolate().ffill().bfill()))
arr.index=arr.index.droplevel(0)
arr.reset_index(inplace=True)
arr.set_index(['Ticker', 'Date'], inplace=True)
arr.sort_index(inplace=True)
ten_years_ago=pd.Timestamp.today() - pd.DateOffset(year=10)
arr=arr[arr.index.get_level_values('Date') >= ten_years_ago]
print(arr.groupby(level=0).head())
arr['Daily Return']=arr.groupby(level=0)['Close'].pct_change()
arr['7d MA']=arr.groupby(level=0)['Close'].transform(lambda x: x.rolling(window=7).mean())
arr['30d MA']=arr.groupby(level=0)['Close'].transform(lambda x: x.rolling(window=30).mean())
arr['30d Volatility']=arr.groupby(level=0)['Daily Return'].transform(lambda x: x.rolling(window=30).std())
print(arr.head(15))
average_returns=(arr.dropna(subset=['Daily Return']).groupby('Ticker')['Daily Return'].mean())
highest_avg_return_ticker,highest_avg_return_value=average_returns.idxmax(),average_returns.max()
print(f"Highest average return: {highest_avg_return_ticker} with {highest_avg_return_value:.4%} per day")
arr['Month']=arr.index.get_level_values('Date').to_period('M')
monthly_volatility=arr.groupby(['Ticker', 'Month'])['Daily Return'].std()
most_volatile_month,most_volatile_value= monthly_volatility.idxmax(),monthly_volatility.max()
print(f" Most volatile month: {most_volatile_month[0]} in {most_volatile_month[1]} with std dev {most_volatile_value:.4%}")