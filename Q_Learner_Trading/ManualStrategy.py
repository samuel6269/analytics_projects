from util import get_data
import pandas as pd
import datetime as dt
import numpy as np
#import matplotlib.pyplot as plt
import indicators as ind

def get_BB_indicators(symbol,sd,ed):
    BB_Per = ind.BB_indicator(sd, ed, symbol)
    BB_Signal = BB_Per.copy()
    sell = BB_Signal.loc[BB_Signal > 1].index
    buy = BB_Signal.loc[BB_Signal < 0].index

    BB_Signal[sell] = -1
    BB_Signal[buy] = 1
    BB_Signal[(~BB_Signal.index.isin(sell)) & (~BB_Signal.index.isin(buy))] = 0
    return BB_Signal.astype(int)

def get_SMA_indicators(symbol,sd,ed, window):
    Price_SMA_Ratio = ind.SMA_indicator(sd, ed, symbol, window)
    Price_SMA_signal = Price_SMA_Ratio.copy()
    sell = Price_SMA_signal.loc[Price_SMA_signal > 1.05].index
    buy = Price_SMA_signal.loc[Price_SMA_signal < 0.95].index

    Price_SMA_signal[sell] = -1
    Price_SMA_signal[buy] = 1
    Price_SMA_signal[(~Price_SMA_signal.index.isin(sell)) & (~Price_SMA_signal.index.isin(buy))] = 0
    return Price_SMA_signal.astype(int)

def get_MACD_indicators(symbol,sd,ed, signal_window = 9):
    df = ind.MACD_indicator(sd, ed, symbol, signal_window)
    df.columns = ["MACD","Signal_Line"]
    df["Signal"] = np.where(df["MACD"] > df["Signal_Line"], 1, 0)
    df["Signal"] = df["Signal"].diff()
    df["Signal"] = df["Signal"].fillna(0).astype(int)
    df["MACD_Sign"] = np.where(df["MACD"] >= 0, -1, 1)
    df["Signal"] = np.where(df["Signal"] == df["MACD_Sign"], df["Signal"], 0)

    return df["Signal"]

def get_TSI_indicators(symbol,sd,ed):
    df = ind.TSI_indicator(sd, ed, symbol)
    df.columns = ["TSI", "Signal_Line"]
    df["Signal"] = np.where(df["TSI"] > df["Signal_Line"], 1, 0)
    df["Signal"] = df["Signal"].diff()

    df["Signal"] = df["Signal"].fillna(0)
    return df["Signal"].astype(int)

def testPolicy(symbol, sd, ed, sv):
    dates = pd.date_range(sd, ed)
    symbol_price_df = ind.get_data([symbol], dates)
    symbol_price_df = ind.normed_data(symbol_price_df)
    del symbol_price_df["SPY"]
    symbol_price_df["BB_signal"] = get_BB_indicators(symbol,sd,ed)
    #symbol_price_df["SMA_signal"] = get_SMA_indicators(symbol, sd, ed, window = 100)
    symbol_price_df["MACD_signal"] = get_MACD_indicators(symbol, sd, ed, signal_window =9)
    symbol_price_df["TSI_signal"] = get_TSI_indicators(symbol, sd, ed)

    shift_period = 4
    symbol_price_df["MACD_signal"] = symbol_price_df["MACD_signal"].shift(shift_period * -1).fillna(0).astype(int)
    symbol_price_df["TSI_signal"] = symbol_price_df["TSI_signal"].shift(shift_period * -1).fillna(0).astype(int)

    symbol_price_df["symbol"] = 0
    num_buy_signals = (symbol_price_df.iloc[:,1:-1] == 1).sum(axis = 1) >=1
    symbol_price_df["symbol"].loc[num_buy_signals] = 1
    num_sell_signal = (symbol_price_df.iloc[:, 1:-1] == -1).sum(axis=1) >=1
    symbol_price_df["symbol"].loc[num_sell_signal] = -1

    #create trade dataframe
    trades_df = pd.DataFrame(index = symbol_price_df.index, columns = [symbol])
    holdings = 0
    for date in symbol_price_df.index:
        if symbol_price_df["symbol"][date] == -1:
            if holdings == 0:
                trades_df.loc[date,symbol] = -1000
                holdings -= 1000
            elif holdings == 1000:
                trades_df.loc[date,symbol] = -2000
                holdings -= 2000
            else: trades_df.loc[date,symbol] = 0
        elif symbol_price_df["symbol"][date] == 1:
            if holdings == 0:
                trades_df.loc[date,symbol] = 1000
                holdings += 1000
            elif holdings == -1000:
                trades_df.loc[date,symbol] = 2000
                holdings += 2000
            else:trades_df.loc[date, symbol] = 0
        else: trades_df.loc[date,symbol] = 0

    return trades_df

def author():
    return 'syong7'

if __name__ == "__main__":
    testPolicy(symbol = "JPM", sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), sv = 100000)