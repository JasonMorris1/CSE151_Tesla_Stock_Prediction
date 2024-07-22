# CSE151_Tesla_Stock_Prediction

**Authors**

- [Aarush Mehrotra](https://github.com/iAarush),

- [Anuj Jain](https://github.com/Anujjain2579),

- [Jason Morris](https://github.com/JasonMorris1),

- [Rishika Kejriwal](https://github.com/rkejriw),

- [Vandita Jain](https://github.com/vanditajain10)


**Date**
07-14-2024


## Abstract
The proposed dataset contains time series information about Tesla share prices, along with technical stock analysis features such as the relative strength index (RSI), simple and exponential moving averages, bollinger bands, and more. We hope to use this information to train models that will predict the close price of the stock the next day (continuous values) and also provide a more ternary prediction of whether the stock price will go up or down, or stay in the middle (1, 0, -1). This will allow us to explore using different types of models in the class. In case the proposed dataset is too small, we will use a larger dataset that has information about car crashes along with factors such as the weather, visibility, etc. This is a very large dataset and should prove sufficient for the purposes of this project. 


## Data Exploration
You can access our Jupiter Notebook here [![model.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JasonMorris1/CSE151_TESLA_STOCK_PREDICTION/blob/main/eda.ipynb)

[Kaggle Dataset](https://www.kaggle.com/datasets/aspillai/tesla-stock-price-with-indicators-10-years/data
)


![Tesla stock price graph](/plots/stock_price_img.png)
#### Figure 1. Tesla Stock Price

![Pairplot graph](/plots/pair_plot.png)
#### Figure 2. PairPlot with histogram on diagonals

![Heatmap graph](/plots/heat_map.png)
#### Figure 3. correlation coefficient heatmap


## Data Preprocessing
Our preprocessing began with an exploration of the dataset, using pandas methods such as `head()` and `describe()`. Additionally, we used histograms, pairplots, and correlation matrix visualizations to explore different features of our data and how various columns related to each other. Lastly, we did basic validation to check for NaNs and conflicting data types. 

As a result of our preprocessing, we removed multiple columns to avoid endogenous variable situations, where the model loses accuracy because features are influenced by each other. This would have led to a case where the coefficients from the future model we build would not have been accurate. Specifically, we removed the open, high, and low prices and kept the close prices. This simplifies the dataset, our model, and the task at hand. On the feature side, when we have technical indicators with different time frames we kept only one, such as a Relative Strength Index (RSI) for only 14 day periods instead of both 14 and 7 day periods. A similar change was made for Average True Range (ATR) values, where we also kept the 14 day measurement. For Commodity Channel Index (CCI) values, we kept the 7 day period and dropped the 14 day period measurements. 

For simple and exponential moving averages (SMA/EMA), we realized the values were very similar to each other. As a result, we only kept the 100-day EMA value. This will simplify the model and hopefully avoid both overfitting and endogenous variables in the model. 

## Data Download
You can download our data from Kaggle here [[Tesla Stock Price](https://www.kaggle.com/datasets/aspillai/tesla-stock-price-with-indicators-10-years)]


## 3 Evaluation
The first model that we decided to train was logistic regression. We used the classification report to do the analysis. It performed with about 50% accuracy & precision due to some randomness in its predication algorithm. The model has high recall, because of bias towards increasing price which results in minimizing false negatives. Further on the Test VS Train analysis, the results of all (accuracy, recall, F1 Score) were quite close for both. This reveals a key concept that the there is no overfitting in the model.

## 4 Fitting
We can clearly see from the fitting graph (confusion matrix) that model predicts increase in Stock price almost 98% of times which alligns with the given period of Tesla's growth observed in our dataset.
We aim to minimize this bias using models which can also predict the potential price decrease. This can be done by penalizing the false negatives and encouraging the true negatives.

## 6 Conclusion and Potential Improvement
Overall, logistic regression model which was our first attempt is not great for continuing in future as summarised above. 
Since the logistic regression model could not capture the complexity involved, we made an attempt towards creating a **Neural Network** which may or may not improve in accuracy but can study the vast variations in the stock price movements. 
We have also tried [Linear Regression](https://github.com/JasonMorris1/CSE151_Tesla_Stock_Prediction/blob/main/eda_linear_regression2.ipynb) in a separate file to see potential improvemnet in binary prediction after a regression on stock price values.

