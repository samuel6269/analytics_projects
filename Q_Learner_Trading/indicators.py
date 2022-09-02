from util import get_data
import pandas as pd
#import matplotlib.pyplot as plt

def normed_data(df):
    return df/df.iloc[0]

def get_rolling_mean(values, windows):
    """Return rolling mean of given values, using specified window size."""
    return values.rolling( window=windows).mean()

def get_rolling_std(values, windows):
    """Return rolling standard deviation of given values, using specified window size."""
    # Quiz: Compute and return rolling standard deviation
    return values.rolling(window=windows).std()

def get_exp_mov_avg(values,windows):
    df_ema = values.copy()
    df_ema.iloc[0]=df_ema.iloc[0:windows].mean()
    alpha = (2/(windows+1))
    return df_ema.ewm(alpha=alpha, adjust = False).mean()

def get_bollinger_bands(rm, rstd):
    """Return upper and lower Bollinger Bands."""
    # Quiz: Compute upper_band and lower_band
    upper_band = rm + rstd * 2
    lower_band = rm - rstd * 2
    return upper_band, lower_band


def BB_indicator(start_date, end_date, symbol):
    # Read data
    dates = pd.date_range(start_date, end_date)
    df = get_data([symbol], dates)
    df_normed = normed_data(df)

    # Compute Bollinger Bands
    # 1. Compute rolling mean
    rm_symbol = get_rolling_mean(df_normed[symbol], windows=40).fillna(method="bfill")

    # 2. Compute rolling standard deviation
    rstd_symbol = get_rolling_std(df_normed[symbol], windows=40).fillna(method="bfill")

    # 3. Compute upper and lower bands
    upper_band, lower_band = get_bollinger_bands(rm_symbol, rstd_symbol)

    bb_value = ((df_normed[symbol] - lower_band)/(upper_band - lower_band)).fillna(method="bfill")

    return bb_value

def SMA_indicator(start_date, end_date, symbol, window = 20):
    dates = pd.date_range(start_date, end_date)
    df = get_data([symbol], dates)
    df_normed = normed_data(df)

    # 1. Compute rolling mean
    rm_symbol = get_rolling_mean(df_normed[symbol], windows=window).fillna(method="bfill")
    Price_SMA_Ratio = df_normed[symbol]/rm_symbol

    return Price_SMA_Ratio

def TSI_indicator(start_date, end_date, symbol):
    dates = pd.date_range(start_date, end_date)
    df = get_data([symbol], dates)
    df_normed = normed_data(df)

    #Step 1: Compute Price Delta, Double smoothing
    df_PC = (df_normed[symbol] - df_normed[symbol].shift(1)).fillna(method="bfill")
    df_PC_smooth1 = get_exp_mov_avg(df_PC,25)
    df_PC_smooth2 = get_exp_mov_avg(df_PC_smooth1,13)

    #Same thing but with Absolute double smoothing
    df_PC_Abs = df_PC.abs()
    df_PC_Abs_smooth1 = get_exp_mov_avg(df_PC_Abs, 25)
    df_PC_Abs_smooth2 = get_exp_mov_avg(df_PC_Abs_smooth1, 13)

    TSI = (df_PC_smooth2/df_PC_Abs_smooth2) * 100
    TSI_smooth = get_exp_mov_avg(TSI, 10)


    return pd.concat([TSI,TSI_smooth], axis = 1)

def MACD_indicator(start_date, end_date, symbol, signal_window = 9):
    dates = pd.date_range(start_date, end_date)
    df = get_data([symbol], dates)
    df_normed = normed_data(df)

    MACD_Line = get_exp_mov_avg(df_normed[symbol],12) - get_exp_mov_avg(df_normed[symbol],26)
    Signal_Line = get_exp_mov_avg(MACD_Line,signal_window)
    MACD_Hist = MACD_Line - Signal_Line

    return pd.concat([MACD_Line, Signal_Line], axis = 1)

def Stochastic_indicator(start_date, end_date, symbol):
    dates = pd.date_range(start_date, end_date)
    df_Close = get_data([symbol], dates, colname ="Close").drop(["SPY"], axis=1)[symbol]
    df_High = get_data([symbol], dates, colname="High").drop(["SPY"], axis=1)[symbol]
    df_Low = get_data([symbol], dates, colname="Low").drop(["SPY"], axis=1)[symbol]
    df_Adj_Close = normed_data(get_data([symbol], dates))

    df_Highest_High = df_High.rolling(window=20).max().fillna(method="bfill")
    df_Lowest_Low = df_Low.rolling(window=20).min().fillna(method="bfill")
    Per_K = (df_Close - df_Lowest_Low) / (df_Highest_High - df_Lowest_Low) * 100
    Per_D = get_rolling_mean(Per_K, 5).fillna(method="bfill")

    fig, (axs1, axs2) = plt.subplots(2, sharex=False, figsize=(7.5,6))
    df_Adj_Close[symbol].plot(title="Stochastic Indicator", ax=axs1)
    axs1.set_ylabel("Normalized Price")
    axs1.legend(loc='best', fontsize='small')
    axs1.set_xticklabels([])

    Per_K.plot(label = "{}_%K".format(symbol), ax=axs2, color = "Orange")
    Per_D.plot(label = "{}_%D".format(symbol), ax=axs2, color = "Purple")
    axs2.axhline(y=20, color = "green")
    axs2.axhline(y=80, color = "green")
    axs2.legend(loc='lower right', fontsize='small')
    axs2.set_ylabel("Stochastic Indicator Value")
    plt.savefig("{}_Stochastic_Indicator.png".format(symbol))
    plt.close()

def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "syong7"  # Change this to your user ID