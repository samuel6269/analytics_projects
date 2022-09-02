import experiment1 as exp1
import experiment2 as exp2
import ManualStrategy as ms
import marketsimcode as mktsim
import datetime as dt
import indicators as ind
import matplotlib.pyplot as plt
import pandas as pd


def calc_cum_returns(df):
    return (df[-1] / df[0]) - 1

def compute_daily_returns(df):
    """Compute and return the daily return values."""
    daily_returns = (df / df.shift(1)) - 1 # much easier with Pandas!
    daily_returns.iloc[0] = 0 # Pandas leaves the 0th row full of Nans
    return daily_returns

def manual_strategy(sd, ed, symbol, sv, commission, impact, test_name):

    df_trades = ms.testPolicy(symbol, sd, ed, sv=sv)
    port_val = mktsim.compute_portvals(df_trades, sv, commission=commission, impact=impact)
    port_val_normed = ind.normed_data(port_val)
    sell_dates = df_trades[symbol].loc[(df_trades[symbol] < 0)].index
    buy_dates = df_trades[symbol].loc[(df_trades[symbol] > 0)].index

    # Benchmark
    df_bench_trades = pd.DataFrame(data=0, index=df_trades.index, columns=[symbol])
    df_bench_trades.iloc[0] = 1000
    bench_port_val = mktsim.compute_portvals(df_bench_trades, sv, commission=commission, impact=impact)
    bench_port_val_normed = ind.normed_data(bench_port_val)


    plt.figure(num=None, figsize=(10, 6))
    plt.plot(port_val_normed, label="Manual Strategy", color="red")
    plt.plot(bench_port_val_normed, label="Benchmark", color="green")
    for day in sell_dates:
        plt.axvline(x=day, color="black")
    for day in buy_dates:
        plt.axvline(x=day, color="blue")
    plt.legend(loc='best', fontsize="small")
    plt.title("Manual Strategy: " + symbol + " " + test_name + " Performance")
    plt.xlabel("Date")
    plt.xticks(rotation=25)
    plt.ylabel("Normalized Portfolio Value")
    plt.savefig("Manual_Strategy_" + test_name + ".png")
    plt.close()

    #Calc stats
    cum_return = calc_cum_returns(port_val_normed)
    bench_cum_return = calc_cum_returns(bench_port_val_normed)
    daily_return_std = compute_daily_returns(port_val_normed).std()
    bench_daily_return_std = compute_daily_returns(bench_port_val_normed).std()
    daily_return_mean = compute_daily_returns(port_val_normed).mean()
    bench_daily_return_mean = compute_daily_returns(bench_port_val_normed).mean()

    print("Manual Strategy Statistics: From " + str(sd) + " to " +str(ed))
    print("Manual Strategy Cumulative Return: " + str(cum_return))
    print("Manual Strategy Standard Deviation of Daily Returns: " + str(daily_return_std))
    print("Manual Strategy Mean of Daily Returns: " + str(daily_return_mean))
    print()
    print("Benchmark Cumulative Return: " + str(bench_cum_return))
    print("Benchmark Standard Deviation of Daily Returns: " + str(bench_daily_return_std))
    print("Benchmark Mean of Daily Returns: " + str(bench_daily_return_mean))
    print()

if __name__ == "__main__":

    #Set date range of interest. In Sample is data we are training on. Out Sample is data we are testing on in the future.
    in_samp_start_date = dt.datetime(2008, 1, 1)
    in_samp_end_date = dt.datetime(2009, 12, 31)
    out_samp_start_date = dt.datetime(2010, 1, 1)
    out_samp_end_date = dt.datetime(2011, 12, 31)

    symbol = "AAPL"
    impact = 0 #the market impact or affect after trade takes place
    commission = 0 #how much does it cost to trade
    sv = 100000 #starting cash value

    manual_strategy(in_samp_start_date, in_samp_end_date, symbol, sv, commission, impact, "In Sample")

    manual_strategy(out_samp_start_date, out_samp_end_date, symbol, sv, commission, impact, "Out Sample")
    exp1.test_exp1(in_samp_start_date, in_samp_end_date, symbol, impact, commission, sv)

    #commission = 0
    #exp2.test_exp2(in_samp_start_date, in_samp_end_date, symbol, commission, sv)
