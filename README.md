# Introduction 
The stock market bussines is not unknown to common man. The earliest known stock market was founded in Amsterdam in 1611 [(source)](https://www.sofi.com/learn/content/history-of-the-stock-market/#:~:text=Who%20Created%20the%20Stock%20Market,Amsterdam%20stock%20exchange%20was%20created), and financial markets have progessed significantly since then. In the US, stock markets began entering the public consciousness around 1978, when the 401(k) retirement plan was created ( "History of 401(k) Plans: An Update" (PDF). Employee Benefit Research Institute. 2018-11-05). Consequently, a lot of people now choose to invest in stock markets. Stock prices are dynamic and depend on a multitude of factors. The growing popularity of companies such as Tesla make predicting its stock price a very lucrative task if done right. It will not only benefit firms that make market hypothesis but also help induviduals make financial decisions. (Please note that this project is in no way financial advice). 

We chose this project because we wanted to use Machine Learning concepts taught in class to create an impact in the real world. Our aim with this project was to build a binary prediction Machine Learning model that could predict if the Tesla stock price would go up or down in the future. This model is important as it would allow an individual to study the Tesla stock price which would help them analyze the stock market better. Having a good predictive model would benefit individuals in financial markets in creating effective hedging statergies and reducing potential losses. It might also reduce investment risk by allowing individuals to better model risk to rewards. This intersection between finance and technology not only helps in understanding the market better but also creates space for new innovative techniques to emerge which can help the economy grow. 

![Tesla stock price graph](/plots/stock_price_img.png)
#### Figure 1. Tesla Stock Price Chart

# Abstract
The proposed dataset contains time series information about Tesla share prices, along with technical stock analysis features such as the relative strength index (RSI), simple and exponential moving averages, bollinger bands, and more. In our project, we use this information to train models that will predict the close price of the stock the next day (continuous values) and also provide a binary prediction of whether the stock price will go up or down, or stay in the middle (1, -1). This will allow us to explore using different types of models in the class. 

Dataset used - [[Tesla Stock Price](https://www.kaggle.com/datasets/aspillai/tesla-stock-price-with-indicators-10-years)]

# Methods 
***This section will include the exploration results, preprocessing steps, models chosen in the order they were executed. You should also describe the parameters chosen. Please make sub-sections for every step. i.e Data Exploration, Preprocessing, Model 1, Model 2, additional models are optional. Please note that models can be the same i.e. DNN but different versions of it if they are distinct enough. Changes can not be incremental. You can put links here to notebooks and/or code blocks using three ` in markup for displaying code. so it would look like this: ``` MY CODE BLOCK ```
 Note: A methods section does not include any why. the reason why will be in the discussion section. This is just a summary of your methods***

## Data exploration 
Our preprocessing began with an exploration of the dataset, using pandas methods such as `head()` and `describe()`. Additionally, we used histograms, pairplots, and correlation matrix visualizations to explore different features of our data and how various columns related to each other. Lastly, we did basic validation to check for NaNs and conflicting data types. 

![Pairplot graph](/plots/pair_plot.png)
#### Figure 2. PairPlot with histogram on diagonals

As a result of our preprocessing, we removed multiple columns to avoid endogenous variable situations, where the model loses accuracy because features are influenced by each other. This would have led to a case where the coefficients from the future model we build would not have been accurate. Specifically, we removed the open, high, and low prices and kept the close prices. This simplifies the dataset, our model, and the task at hand. 

![Heatmap graph](/plots/heat_map.png)
#### Figure 3. Correlation coefficient heatmap

On the feature side, when we have technical indicators with different time frames we kept only one, such as a Relative Strength Index (RSI) for only 14 day periods instead of both 14 and 7 day periods. A similar change was made for Average True Range (ATR) values, where we also kept the 14 day measurement. For Commodity Channel Index (CCI) values, we kept the 7 day period and dropped the 14 day period measurements. At first, we had added 2 new columns to get the upper bollinger band and median of the bollinger band from the dataset's provided lower bollinger band column. However, after considering the correlation between these columns, we decided to only use 1 of the 3 and dropped the synthetic upper and median columns from our dataset. 

![RSI](/plots/rsi.png)
#### Figure 4. TSLA RSI chart 

![bollinger_bands](/plots/bollinger_bands.png)
#### Figure 5. TSLA Bollinger Bands against closing price 

For simple and exponential moving averages (SMA/EMA), we realized the values were very similar to each other. As a result, we only kept the 100-day EMA value. This both simplified our models and helped avoid endogenous variables in the dataset. 
![SMA PLOT](/plots/sma.png)
#### Figure 6. Chart of various TSLA SMA/EMA indicators

## Preprocessing 

## Model 1
The first model that we decided to train was Logistic regression. This was a simple, straightforward logistic regression model that directly used features from our dataset (after preprocessing, of course). 
```python
logreg = LogisticRegression(max_iter=10000)
logreg.fit(X_train, y_train)
yhat_test = logreg.predict(X_test)
yhat_train = logreg.predict(X_train)
``` 

## Model 2 Linear Regression

This was the second model we tried. For our training set we used the data from 2014-2022 and the testing data was from 2022-2023.The features we used were volume, rsi_7, rsi_14, cci_7, sma_50, ema_50, sma_100, ema_100, macd, bollinger, TrueRange, atr_7 and atr_14

```python
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
```

## Model 3 - Ensemble XGBoost and RandomForest
The ensemble model combines XGBoost and Random Forest to enhance predictive performance. XGBoost is initialized with 'logloss' as the evaluation metric and trained on the training data, followed by predictions on the test set. Similarly, Random Forest is initialized with 100 trees and trained on the same data. Their predictions are then combined using a weighted average, with Random Forest predictions given twice the weight, creating a robust ensemble that reduces variance and bias.

```python
# XGBoost model
xgb_model = XGBClassifier(eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Combine the predictions using a simple average
y_pred_ensemble = (y_pred_xgb + 2*y_pred_rf) / 3
```
## Model 4 - LSTM

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; For our LSTM model, we utilized 4 LSTM layers, each followed by a dropout layer, and concluded with a final dense layer containing a single unit. We ran this model using both technical analysis features and price as the sole feature. The LSTM model uses the data from the previous 40 days to predict the stock price on each day. For each LSTM layer, we initially used a small number of units, ranging from 50 to 100. We then increased the number of units to between 80 and 120. For each layer we used the relu activation function.

```python
regressor = Sequential()

regressor.add(LSTM(units = 80, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 15)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 90, activation = 'relu', return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 100, activation = 'relu', return_sequences = True))
regressor.add(Dropout(0.4))

regressor.add(LSTM(units = 120, activation = 'relu'))
regressor.add(Dropout(0.5))

regressor.add(Dense(units = 1))
```

# Results 

This will include the results from the methods listed above (C). You will have figures here about your results as well. No exploration of results is done here. This is mainly just a summary of your results. The sub-sections will be the same as the sections in your methods section.

## Model 1: Logistic Regression
This model was chosen to fulfill the classification part of our goal, where we wanted to predict whether the stock price would go up or down the next day. Although a relatively simple model, we thought it would be helpful as a baseline and we were hopeful that a simple model would still have an acceptable level of accuracy. Unfortunately, this did not work and the model was highly inaccurate, with a test accuracy of 49.8%, which is more or less random. The model has a high recall, because of bias towards increasing price which results in minimizing the False negatives. Further on the Test VS Train analysis, the results of all (accuracy, recall, F1 Score) were quite close for both. This reveals a key concept that the there is no overfitting in the model. We used the `classification_report` method to ascertain these results. 

Although a failure, this helped us learn that we needed to be more mindful about how we were processing data before sending it into the model, and what kinds of models we wanted to explore next. As a result, it was a very good learning experience. 

## Model 2: Linear Regression
The linear regression model had a mean squarred error of 158.58 and a mae of 9.86. 
The model accruay was 51%. The recall for the stock price will increase class was 0.50 and the precision was 0.52. The recall and precision for the stock will decrease class was 0.49 and 0.51.

![Linear Regression Plot](/plots/linear_regression.png)

![Linear Regression Plot](/plots/linear_regression_pie.png)

## Model 4: Ensemble
The ensemble model demonstrated a slight improvement over individual models. Across different splits, the ensemble model achieved an accuracy of around 48%-57%. Precision and recall metrics varied between splits, indicating a more balanced performance in some cases. 
![Ensemble Pie Chart](/plots/Ensemble_classification.png)

![Ensemble Classification](/plots/Ensemble_pie.png)

## Model 4: LSTM

The LSTM model had a mean squarred error of 997 and a mean absolute error of 24.41. Coverting the price prediction into classification gives an accuracy of 52%. For the prediction the stock price will increase we had a prevision of 0.43 and recall of 0.54. The prediction the stock will decrease had a precision of 0.62 and recall of 0.51.
![LSTM Plot](/plots/lstm_fig.png)



# Discussion 
***This is where you will discuss the why, and your interpretation and your though process from beginning to end. This will mimic the sections you have created in your methods section as well as new sections you feel you need to create. You can also discuss how believable your results are at each step. You can discuss any short comings. It's ok to criticize as this shows your intellectual merit, as to how you are thinking about things scientifically and how you are able to correctly scrutinize things and find short comings. In science we never really find the perfect solution, especially since we know something will probably come up int he future (i.e. donkeys) and mess everything up. If you do it's probably a unicorn or the data and model you chose are just perfect for each other!***



In our logistic regression model, we aimed to predict whether the stock price would increase or decrease. We converted our numerical data into binary classifications by subtracting the closing price from the next day's closing price and then classifying the result as "1" if the price increased and "0" if it decreased.
One issue we noticed is that small increases in price, such as a few cents, are classified the same way as larger increases, such as 20%. For example, if the stock price ends slightly higher or significantly higher, both scenarios are classified as a "1," which could lead to inaccuracies in our model.
This approach might cause the model to miss important patterns. For instance, if a certain pattern exists where a spike in price follows an increase in trading volume, but this pattern occurs infrequently, the model may struggle to detect it. If, out of ten instances, nine show low trading volume with a minor price increase and one shows high trading volume with a significant price increase, the model might not learn the significance of the high trading volume scenario because the frequent low volume cases dominate the training data. As a result, the model might not accurately capture the relationship between trading volume and substantial price increases.


In our logistic regression model, we aimed to predict whether the stock price would increase or decrease. One issue we noticed is that small increases in price, such as a few cents, are classified the same way as larger increases, such as 20%. For example, if the stock price ends slightly higher or significantly higher, both scenarios are classified as a "1," which could lead to inaccuracies in our model. This approach might cause the model to miss important patterns. For instance, if a certain pattern exists where a spike in price follows an increase in trading volume, but this pattern occurs infrequently, the model may struggle to detect it. If, out of ten instances, nine show low trading volume with a minor price increase and one shows high trading volume with a significant price increase, the model might not learn the significance of the high trading volume scenario because the frequent low volume cases dominate the training data. As a result, the model might not accurately capture the relationship between trading volume and substantial price increases.

Given the TA feedback, we have tried using logistic regression with SMOTE() in order to reduce bias and imporve upon logistic regression. This can be found in [Models_resampled_without_technical_indicators](https://github.com/JasonMorris1/CSE151_Tesla_Stock_Prediction/blob/main/Models_resampled_without_technical_indicators.ipynb)


We experimented with other models such as linear regression. When we plotted the actual stock prices against the predicted ones, the graph showed a close resemblance. We chose to use linear regression as a baseline for our other models. If our other models cannot outperform linear regression, we can conclude that they are not effective at predicting Tesla’s stock price. Looking at our linear regression model we had a MSE of 86.87 and a mean absolute error of 6.93. When we compare our predicted price to the actual closing price that day, we convert our prediction into a binary format: 1 if the stock is predicted to close higher, and 0 if it is predicted to close lower. Our model has a 51% accuracy. This model has very poor performance, equivalent to randomly predicting whether the stock price will increase or decrease for that day. Looking at the linear regression coefficients, the one with the most weight was the closing price, with a value of 0.98. This isn’t very surprising, as the model essentially uses the closing price to predict the next day's closing price, which is often quite similar to the previous day's closing price. If we drop the closing price from the feature list, the highest coefficient becomes 3.2 for the MACD. MACD closely follows stock prices because it is derived from the stock’s moving averages, which is inherently based on past price data. Dropping close feature linear regression achieves an accuracy of 50%. In essence, this model leverages past price history and a weighted combination of various technical indicators, which are primarily based on historical price data, to predict the next day's closing price. With an accuracy near or at 50%, this model performs as poorly as possible, effectively no better than random guessing. An accuracy below 50% would mean our predictions are consistently wrong, allowing us to reverse the predictions and achieve an accuracy of 1 minus the reported accuracy which would be above 50%. Our model seems to be non-linear, so we decided to try other models that are better suited for predicting non-linear data.

We decided to try a LSTM model because LSTMs (Long Short-Term memory Models) are designed to handle and predict time series data. LSTM are a type of recurrent neural network that are good at capturing long-term trends and patterns in data. This makes them well suited for time series forecasting, where you are trying to predict future values based on past observations. Our LSTM model achieved poor accuracy at 51%. We decided to go with 4 LSTM layers based on both common practice and empirical results we observed. In many LSTM-based models, 3-4 layers are typically used because this depth is often sufficient to capture the complexity in the data without overfitting. Adding more layers doesn't necessarily lead to better performance; in fact, it can sometimes degrade the model's accuracy. For each LSTM layer, we initially used a small number of units, ranging from 50 to 100. We then increased the number of units to between 80 and 120 to improve model accuracy but observed little to no effect. For each layer we used the relu activation function because it doesn’t have the issues of vanishing gradient descent. After each layer we used dropout which randomly sets a fraction of the input units to zero at each update of the training phase. Dropout can reduce overfitting and improve generalization. Despite our attempts to increase the accuracy of our model, we couldn’t achieve any meaningful improvement. We believe this is due to the nature of the data itself. Past price history and technical indicators, which are mathematical functions of past price history, do not reliably indicate the future price of the stock. Without considering real-world factors like company financials and earnings reports, our model, which attempts to predict the day-to-day price of a stock, appears to be failing to identify any patterns or capture any complexity by just looking at past price history. With the exception of the volume feature, we are relying on the variable we are trying to predict to forecast its own future values, which has proven to be ineffective for achieving meaningful or accurate predictions.

To address the limitations observed with logistic regression, neural networks, and LSTM models, we experimented with ensemble learning methods using XGBoost (Extreme Gradient Boosting) and Random Forest. Both models are well-suited for capturing complex, non-linear relationships in the data, making them a good fit for the unpredictable nature of stock price movements.

XGBoost is a highly efficient implementation of gradient boosting, optimizing a differentiable loss function through an iterative process to correct errors made by existing models. This results in a strong predictive model capable of handling intricate patterns within the data. Random Forest, on the other hand, constructs multiple decision trees during training and averages their predictions to reduce overfitting, providing a robust and stable model.

Given the strengths of both models, we combined their predictions in an ensemble approach. XGBoost was initialized with 'logloss' as the evaluation metric and trained on the data, followed by predictions on the test set. Random Forest was initialized with 100 trees and a fixed random state for reproducibility, and trained similarly. The ensemble prediction is a weighted average, where Random Forest predictions are given twice the weight of XGBoost predictions, calculated as ```y_pred_ensemble = (y_pred_xgb + 2 * y_pred_rf)/{3}```. This combination aimed to enhance predictive performance by capturing a broader range of patterns and trends in the stock price data.

The ensemble model demonstrated a slight improvement over individual models. Across different splits, the ensemble model achieved an accuracy of around 48%-57%. Precision and recall metrics varied between splits, indicating a more balanced performance in some cases. For example, in certain splits, precision and recall for class 1 (price increase) were closer to those for class 0 (price decrease), suggesting better generalization.

Stock prices can be highly volatile and tesla stock price is no exception to this fact. Stock prices are influenced by numerous unpredictable factors such as economic indicators, political events, market sentiment, and company-specific news. Our model does not take any of of these factors into account. Our model sole used technical indicators which are derived directly from previous price history. Using technical indicators has several implications. By only using technical indicators the model ignores fundamental factors such as earning reports and other financial information about the underlying company. 

# Conclusion 
***This is where you do a mind dump on your opinions and possible future directions. Basically what you wish you could have done differently. Here you close with final thoughts.***





# Statement of collaboration 

This project has been completed by all the members as a team. We hosted regular meetings to discuss the project and make project. Everyone shared their opinions and made their contributions as mentioned below.

Aarush Mehrotra:
I helped form the group and schedule many of our discussions, especially in the first few weeks. I contributed to discussions about picking a dataset; helped with the logistic regression, linear regression and XGBoost models; and usually submitted milestones to Gradescope and added team members to the submission. 

Anuj Jain:
I worked on initial data processing and collaborated on features selection. I eventually created Neural Nets, Random Forest and Ensemble models. I took feedback from TA on possible ways to improve model and incorporated them into improved logistic regression file without technical indicators. Finally, I created generic functions such as Pie Charts and Confusion Matrix to help with results comparison across models.

Jason Morris:
I developed both the linear regression and LSTM models and then worked on the write-up, specifically focusing on describing these models in the Methods, Results, and Discussion sections. I also worked on creating the figures and helped with writing the conclusion in the report.


Rishika Kejriwal: I helped with coming up with the idea for the project and finding the datasets we could potentially use. I contributed to discussion by suggesting changes and ideas alongwith feedback for data exploration, preprocessing, logistic regression and linear regression. Additionally, I worked on the report alongside by members and worked on formatting the README.md. I was active and present in all group meetings as well as discussed and got inputs from the course staff for our models and project.

Vandita Jain: I helped with selecting the dataset. Alongside, I attended all of the meetings and provided my insights on the work we together did as a team on zoonm. I also helped in writing the Introduction and Conclusion. Apart from this, I also regularly interacted with the teaching staff and professors to get their valuable opinions.
