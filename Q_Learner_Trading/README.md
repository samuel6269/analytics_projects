<h1 align="center">Q-Learning Stock Trader</h1>

This project directory contains the project files for an automated Q-learning stock trader. It currently assumes the portfolio can be in 3 different positions: 1000 shares long, 1000 shares short, or 0 shares. This model takes into account market impact of trades, as well as commission prices. Here are the steps to run the program.

-testproject.py: This is the main file which runs the overall project. In the main function, you can choose choose a date range to train on, and another date range in the future to test the performance on. Here is where a user would specify the stock ticker of interest, as well as the impact and commission amount, and starting portfolio value. Two manual strategies will be performed, followed by the Q-learner, and their performances are compared and evaluated.

-StrategyLearner.py: This contains the StrategyLearner class that creates a Q-learner object as well as provide useful helper functions

-Qlearner.py: This is the Q-learner object class which contains the specific routines of training and testing given financial data

-marketsimcode.py: This module calculates the portfolio value over the specified date range given a specific company and trade requests. This is used to evaluate performance of the models

-ManualStrategy.py: This module uses the manual strategy by applying technical indicators on the provided data and showing when to buy, sell, or hold within the given date range. Given multiple technical indicators, this module looks to see the most common action for any given date, and performs that most popular one. For example, if on January 1st, 2022, 2 technical indicators shows this date as a stock buy, and 1 technical indicator shows the date as a sell, then the strategy is to label this date as a stock buy.

-indicators.py: This module contains all the calculations behind the 3 technical indicators used

-experiment<>.py: These files run certain experiments. Experiment 1 compares the performance of the Q-learner vs. Manual Strategy. Experiment 2 evaluates the impact and commission on performance.

-Before running this program, make sure the company of interest is in the data folder. This folder currently has csv files for AAPL and SPY (SP 500 index fund). To test any other company and/or date range, the appropriate csv file needs to be imported into this directory.
