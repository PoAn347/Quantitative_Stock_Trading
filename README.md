 In this project, we focus on utilizing financial data analysis techniques to assess stock performance and simulate future outcomes. Here's a breakdown of what this project entails:

Calculating Return Rate and Volatility: Our first step involves utilizing the 'stock_return&volatility' file to compute the return rate and volatility of the stock under consideration.

Monte Carlo Simulation for Price Prediction: Next, we employ Monte Carlo Simulation techniques to predict future stock prices. By adjusting parameters such as return rate and volatility, we simulate the stock price and determine its range, with the bottom and top bounds representing the 5th and 95th percentiles respectively.

Gridline Optimization and Performance Evaluation: We then fine-tune the simulation by deciding the optimal number of gridlines between the top and bottom bounds. Running the simulation allows us to evaluate its performance effectively.

For this project, we utilize NVIDIA's stock data from the last three months of 2023 and test it against data from 2024, beginning from January 1st. Notably, the observed return rate stands at 28%. Despite this figure being lower than that of a buy-and-hold strategy, the grid trading strategy is deemed less risky, resulting in relatively higher Sharpe ratios.

By exploring these methodologies, we aim to provide insights into potential trading strategies and risk management approaches in the dynamic world of financial markets.

Feel free to explore the code and contribute to further enhancements!
