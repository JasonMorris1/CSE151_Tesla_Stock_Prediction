# CSE151_Tesla_Stock_Prediction
## Abstract

The proposed dataset contains time series information about Tesla share prices, along with technical stock analysis features such as the relative strength index (RSI), simple and exponential moving averages, bollinger bands, and more. We hope to use this information to train models that will predict the close price of the stock the next day (continuous values) and also provide a more ternary prediction of whether the stock price will go up or down, or stay in the middle (1, 0, -1). This will allow us to explore using different types of models in the class. In case the proposed dataset is too small, we will use a larger dataset that has information about car crashes along with factors such as the weather, visibility, etc. This is a very large dataset and should prove sufficient for the purposes of this project. 



## Data Exploration


You can access our Jupiter Notebook Here that contains our Data Exploration [![model.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JasonMorris1/CSE151_TESLA_STOCK_PREDICTION/blob/main/eda.ipynb)

![Tesla stock price graph](/plots/stock_price_img.png)
#### Figure 1. Tesla Stock Price


![Pairplot graph](/plots/pair_plot.png)
#### Figure 2. PairPlot with histogram on diagonals

![Heatmap graph](/plots/heat_map.png)
#### Figure 3. correlation coefficient heatmap

## Data Preprocessing

## Data Download

You can download our data from Kaggle Data here [[Tesla Stock Price](https://www.kaggle.com/datasets/aspillai/tesla-stock-price-with-indicators-10-years)]
