from util import get_data
import numpy as np
import pandas as pd

def normed_data(df):
    return df/df.iloc[0]

def compute_portvals(orders, start_val, commission, impact):
    start_date = orders.index[0]
    end_date = orders.index[-1]
    symbol = orders.columns[0]

    dates = pd.date_range(start_date, end_date)
    df_adj_close = get_data([symbol], dates).drop(["SPY"], axis=1)
    df_adj_close["Cash"] = 1

    df_trades = orders.copy()
    df_trades["Commission"] = np.where(df_trades[symbol] != 0, commission, 0)
    df_trades["Impact"] = np.where(df_trades[symbol] != 0, impact, 0)
    df_trades["Impact"] *= df_adj_close[symbol] * df_trades[symbol].abs()
    df_trades["Cash"] = df_adj_close[symbol] * df_trades[symbol] * -1 - df_trades["Commission"] - df_trades["Impact"]
    df_trades.drop(axis = 1, columns = ["Commission", "Impact"], inplace = True)

    df_holdings = df_trades.copy()
    df_holdings["Cash"].iloc[0] += start_val
    df_holdings = df_holdings.cumsum()

    df_value = df_adj_close * df_holdings
    df_porval = df_value.sum(axis=1)
    return df_porval

