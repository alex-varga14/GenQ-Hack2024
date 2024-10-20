import json
from stock import Stock
import pandas as pd
import numpy as np



file_name = "Datasets/small_dataset.json"
with open(file_name, 'r') as f:
    stock_data = json.load(f)['data']
    
stocks = [Stock(stock) for stock in stock_data.values()]
# ann_returns = []
# for stock in stocks:
#     # print(stock)
#     ann_returns.append(stock.get_annualized_returns())
#     print(stock.Name,"\t", stock.get_ESR_score(),"\t", stock.get_annualized_returns())
#     # print(stock.get_annualized_returns())
#         # break
# print(np.average(ann_returns))

def ESG_scores(stocks):
    ESG = []
    for stock in stocks:
        ESG.append(stock.get_ESG_score())
    return ESG

def mean_returns(stocks):
    M =[]
    for stock in stocks:
        M.append(M.annualized_return())
    return M

def create_cov_matrix(stocks):
    # Create a DataFrame with the closing prices of the stocks
    df = pd.concat([stock.get_price_history()['Close'] for stock in stocks], axis=1)
    df.columns = [stock.Ticker for stock in stocks]
    
    # Calculate the percentage change
    returns = df.pct_change()
    
    # Calculate the covariance matrix
    cov_matrix = returns.cov()
    
    return cov_matrix