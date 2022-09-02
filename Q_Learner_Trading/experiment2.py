import ManualStrategy as ms
import marketsimcode as mktsim
import indicators as ind
import matplotlib.pyplot as plt
import pandas as pd
import StrategyLearner as sl
import numpy as np

def test_exp2(sd, ed, symbol, commission, sv):
    start_date = sd
    end_date = ed
    symbol = symbol

    impact_range = np.arange(0.005, 0.05, step = 0.005)
    num_trades = []
    std_devs = []
    # In Sample Strategy Learner
    for i in impact_range:
        strat_learner = sl.StrategyLearner(verbose=False, impact= i, commission=commission)  # constructor
        strat_learner.add_evidence(symbol = "JPM", sd=start_date, ed=end_date, sv = sv) # training phase
        actual_frame = strat_learner.testPolicy(symbol = "JPM", sd=start_date, ed=end_date, sv = sv)
        num_trades.append((actual_frame != 0).sum()[0])
        strat_learner_df = mktsim.compute_portvals(actual_frame, sv, commission = commission, impact =i)
        strat_learner_df_normed = ind.normed_data(strat_learner_df)
        std_devs.append(strat_learner_df_normed.std())

    plt.figure(num=None, figsize=(10, 6))
    plt.plot(impact_range, num_trades)
    plt.xlabel("Impact Magnitude")
    plt.ylabel("Number of Trades")
    plt.title("Experiment 2: " + symbol + " In-Sample Strategy Learner Number of Trades")
    plt.savefig("Experiment_2_num_trades.png")
    plt.close()

    plt.figure(num=None, figsize=(10, 6))
    plt.plot(impact_range, std_devs)
    plt.xlabel("Impact Magnitude")
    plt.ylabel("Portfolio Standard Deviation")
    plt.title("Experiment 2: " + symbol + " In-Sample Strategy Learner Volatility")
    plt.savefig("Experiment_2_volatility.png")
    plt.close()

def author():
    return 'syong7'