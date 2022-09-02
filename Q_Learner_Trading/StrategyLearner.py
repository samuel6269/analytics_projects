
import datetime as dt  		  	   		     		  		  		    	 		 		   		 		  
import random  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		     		  		  		    	 		 		   		 		  
import util as ut

import QLearner as ql
import indicators as ind
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
class StrategyLearner(object):  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		     		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		     		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		     		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		     		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		     		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		     		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    # constructor  		  	   		     		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False, impact=0.0, commission=0.0, transaction_amt = 1000):
        """  		  	   		     		  		  		    	 		 		   		 		  
        Constructor method  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		     		  		  		    	 		 		   		 		  
        self.impact = impact  		  	   		     		  		  		    	 		 		   		 		  
        self.commission = commission
        self.learner = 0
        self.disc_steps = 5
        self.transaction_amt = transaction_amt
  		  	   		     		  		  		    	 		 		   		 		  
    # this method should create a QLearner, and train it for trading  		  	   		     		  		  		    	 		 		   		 		  
    def add_evidence(  		  	   		     		  		  		    	 		 		   		 		  
        self,  		  	   		     		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		     		  		  		    	 		 		   		 		  
        sd=dt.datetime(2008, 1, 1),  		  	   		     		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 1, 1),  		  	   		     		  		  		    	 		 		   		 		  
        sv=10000,  		  	   		     		  		  		    	 		 		   		 		  
    ):  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol to train on  		  	   		     		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		     		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		     		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		     		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		     		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		     		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		     		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		     		  		  		    	 		 		   		 		  
        """
        dates = pd.date_range(sd, ed)
        symbol_price_data  = ind.get_data([symbol], dates)
        del symbol_price_data["SPY"]
        #symbol_price_data_diff = symbol_price_data.diff().fillna(0)

        #obtain discritized indicator data
        ind_data = obtain_ind_data(symbol, sd, ed, self.disc_steps)

        dates = ind_data.index
        num_states = self.disc_steps ** (len(ind_data.columns)+1)
        holdings = 0
        num_runs = 0
        min_runs = 20
        max_runs = 200
        converged = False
        min_runs_reached = False
        max_runs_reached = False
        previous_day = dates[0]

        #Create Qlearner
        self.learner = ql.QLearner(num_states=num_states,
                              num_actions=3,
                              alpha=0.1,
                              gamma=0.9,
                              rar=0.98,
                              radr=0.999,
                              dyna=0,
                              verbose=False)  # initialize the learner

        trades = pd.DataFrame(index = dates, columns = [symbol])
        while (converged == False or min_runs_reached == False) and max_runs_reached == False:
            state = calc_state(dates[0], ind_data, holdings, self.disc_steps)
            action = self.learner.querysetstate(state)
            previous_run_trades = trades.copy() #store initial trades
            for day in dates:
                reward = calc_reward(symbol_price_data.loc[previous_day][0], symbol_price_data.loc[day][0], action, holdings, self.impact, self.commission)
                action = self.learner.query(state, reward)
                holdings, action_amount = update_holdings(action, holdings, self.transaction_amt)
                trades.loc[day] = action_amount #record transaction
                state = calc_state(day, ind_data, holdings, self.disc_steps) #calculate new state
                previous_day = day #store the previous day
            converged = trades.equals(previous_run_trades)
            num_runs += 1
            if num_runs == min_runs: min_runs_reached = True
            if num_runs == max_runs: max_runs_reached = True

        #return trades

  		  	   		     		  		  		    	 		 		   		 		  
    # this method should use the existing policy and test it against new data  		  	   		     		  		  		    	 		 		   		 		  
    def testPolicy(
            self,
            symbol="IBM",
            sd=dt.datetime(2009, 1, 1),
            ed=dt.datetime(2010, 1, 1),
            sv=10000,
    ):
        """
        Tests your learner using data outside of the training data

        :param symbol: The stock symbol that you trained on on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to
            long so long as net holdings are constrained to -1000, 0, and 1000.
        :rtype: pandas.DataFrame
        """

        dates = pd.date_range(sd, ed)
        symbol_price_data = ind.get_data([symbol], dates)
        del symbol_price_data["SPY"]

        # obtain discritized indicator data
        ind_data = obtain_ind_data(symbol, sd, ed, self.disc_steps)

        dates = ind_data.index
        holdings = 0
        trades = pd.DataFrame(index=dates, columns=[symbol])

        state = calc_state(dates[0], ind_data, holdings, self.disc_steps)
        self.learner.querysetstate(state)
        for day in dates:
            action = self.learner.querysetstate(state)
            holdings, action_amount = update_holdings(action, holdings)
            trades.loc[day] = action_amount  # record transaction
            state = calc_state(day, ind_data, holdings, self.disc_steps)  # calculate new state

        return trades


def obtain_ind_data(symbol, sd, ed, num_disc_steps):

    dates = pd.date_range(sd, ed)
    symbol_price_df = ind.get_data([symbol], dates)
    symbol_price_df = ind.normed_data(symbol_price_df)
    del symbol_price_df["SPY"]

    symbol_price_df["BB_ind"] = ind.BB_indicator(sd, ed, symbol)
    #symbol_price_df["SMA_ind"] = ind.SMA_indicator(sd, ed, symbol, window = 100)
    MACD_data = ind.MACD_indicator(sd, ed, symbol, signal_window = 9)
    TSI_data = ind.TSI_indicator(sd, ed, symbol)
    symbol_price_df["MACD_ind"] = MACD_data.iloc[:,0] - MACD_data.iloc[:,1]
    symbol_price_df["TSI_ind"] = TSI_data.iloc[:, 0] - TSI_data.iloc[:, 1]

    shift_period = 4
    symbol_price_df["MACD_ind"] = symbol_price_df["MACD_ind"].shift(shift_period * -1).fillna(method = "ffill")
    symbol_price_df["TSI_ind"] = symbol_price_df["TSI_ind"].shift(shift_period * -1).fillna(method = "ffill")

    #discritize the data
    disc_ind_df = pd.DataFrame()
    disc_ind_df["BB"] = pd.qcut(symbol_price_df["BB_ind"] , num_disc_steps, False)
    disc_ind_df["MACD"] = pd.qcut(symbol_price_df["MACD_ind"], num_disc_steps, False)
    disc_ind_df["TSI"] = pd.qcut(symbol_price_df["TSI_ind"], num_disc_steps, False)
    #disc_ind_df["SMA"] = pd.qcut(symbol_price_df["SMA_ind"], num_disc_steps, False)

    return disc_ind_df

def calc_state(date, ind_data, holdings, num_disc_steps):
    day_ind_data = ind_data.loc[date]
    num_ind = len(day_ind_data)
    state = 0
    for i in range(num_ind):
        state += num_disc_steps ** (num_ind-(i+1))*day_ind_data[i]
    if holdings == 0:
        state += num_disc_steps ** (num_ind) * int((num_disc_steps-1)/2)
    if holdings == 1000: state += num_disc_steps ** (num_ind) * (num_disc_steps - 1)
    return int(state)

def calc_reward(prev_sym_price, sym_price, action, holdings, impact, commission):
    if action == 0:
        price_diff = sym_price - prev_sym_price * (1 + impact)
        return price_diff * holdings - commission  # reward is the difference in price and holdings
    elif action == 1: #sell
        price_diff = sym_price - prev_sym_price * (1- impact)
        return price_diff * holdings - commission  # reward is the difference in price and holdings
    else:
        price_diff = sym_price - prev_sym_price
        return price_diff * holdings  # reward is the difference in price and holdings


def update_holdings(action, current_holdings, transaction_amount = 1000):
    holdings = current_holdings
    if action == 0:  # buy
        if holdings == 0:
            holdings += transaction_amount
            action_amount = transaction_amount
        #no shorting
        #elif holdings == -transaction_amount:
            #holdings += transaction_amount*2
            #action_amount = transaction_amount*2
        else:
            action_amount = 0
    elif action == 1:  # sell
        #no shorting
        #if holdings == 0:
            #holdings -= transaction_amount
            #action_amount = -transaction_amount
        if holdings == transaction_amount:
            holdings -= transaction_amount
            action_amount = -transaction_amount
        else:
            action_amount = 0
    else:
        action_amount = 0
    return holdings, action_amount


