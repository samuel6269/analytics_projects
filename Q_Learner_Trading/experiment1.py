import ManualStrategy as ms
import marketsimcode as mktsim
import indicators as ind
import matplotlib.pyplot as plt
import pandas as pd
import StrategyLearner as sl

def test_exp1(sd, ed, symbol, impact, commission, sv):
    start_date = sd
    end_date = ed
    symbol = symbol
    #In Sample Manual Strategy
    df_trades = ms.testPolicy(symbol, start_date, end_date, sv = sv)
    port_val = mktsim.compute_portvals(df_trades, sv, commission = commission, impact = impact)
    port_val_normed = ind.normed_data(port_val)


    #Benchmark
    df_bench_trades = pd.DataFrame(data = 0, index = df_trades.index, columns = [symbol])
    df_bench_trades.iloc[0] = 1000
    bench_port_val = mktsim.compute_portvals(df_bench_trades, sv, commission = commission, impact = impact)
    bench_port_val_normed = ind.normed_data(bench_port_val)

    #In Sample Strategy Learner
    strat_learner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)  # constructor
    strat_learner.add_evidence(symbol = symbol, sd=start_date, ed=end_date, sv = sv) # training phase
    actual_frame = strat_learner.testPolicy(symbol = symbol, sd=start_date, ed=end_date, sv = sv)
    strat_learner_df = mktsim.compute_portvals(actual_frame, sv, commission = commission, impact =impact)
    strat_learner_df_normed = ind.normed_data(strat_learner_df)

    plt.figure(num=None, figsize=(10, 6))
    plt.plot(port_val_normed, label="Manual Strategy Portfolio")
    plt.plot(strat_learner_df_normed, label="Strategy Learner Portfolio")
    plt.plot(bench_port_val_normed, label="Benchmark")
    plt.legend(loc='best', fontsize="small")
    plt.title("Experiment 1: " + symbol + " In-Sample Performance")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.xticks(rotation=25)
    plt.savefig(symbol + "_exp.png")
    plt.close()

    #print(strat_learner_df_normed[-1])