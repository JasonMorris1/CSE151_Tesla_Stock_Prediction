# Introduction 


# Abstract
The proposed dataset contains time series information about Tesla share prices, along with technical stock analysis features such as the relative strength index (RSI), simple and exponential moving averages, bollinger bands, and more. In our project, we use this information to train models that will predict the close price of the stock the next day (continuous values) and also provide a binary prediction of whether the stock price will go up or down, or stay in the middle (1, -1). This will allow us to explore using different types of models in the class. 

Dataset used - [[Tesla Stock Price](https://www.kaggle.com/datasets/aspillai/tesla-stock-price-with-indicators-10-years)]

Why was it chosen? 
Why is it cool? Discuss the general/broader impact of having a good predictive mode. i.e. why is this important?

# Figures
Your report should include relevant figures of your choosing to help with the narration of your story, including legends (similar to a scientific paper). For reference you search machine learning and your model in google scholar for reference examples.

# Methods 
This section will include the exploration results, preprocessing steps, models chosen in the order they were executed. You should also describe the parameters chosen. Please make sub-sections for every step. i.e Data Exploration, Preprocessing, Model 1, Model 2, additional models are optional. Please note that models can be the same i.e. DNN but different versions of it if they are distinct enough. Changes can not be incremental. You can put links here to notebooks and/or code blocks using three ` in markup for displaying code. so it would look like this: ``` MY CODE BLOCK ```
 Note: A methods section does not include any why. the reason why will be in the discussion section. This is just a summary of your methods

## Data exploration 
Our preprocessing began with an exploration of the dataset, using pandas methods such as `head()` and `describe()`. Additionally, we used histograms, pairplots, and correlation matrix visualizations to explore different features of our data and how various columns related to each other. Lastly, we did basic validation to check for NaNs and conflicting data types. 

As a result of our preprocessing, we removed multiple columns to avoid endogenous variable situations, where the model loses accuracy because features are influenced by each other. This would have led to a case where the coefficients from the future model we build would not have been accurate. Specifically, we removed the open, high, and low prices and kept the close prices. This simplifies the dataset, our model, and the task at hand. On the feature side, when we have technical indicators with different time frames we kept only one, such as a Relative Strength Index (RSI) for only 14 day periods instead of both 14 and 7 day periods. A similar change was made for Average True Range (ATR) values, where we also kept the 14 day measurement. For Commodity Channel Index (CCI) values, we kept the 7 day period and dropped the 14 day period measurements. 

For simple and exponential moving averages (SMA/EMA), we realized the values were very similar to each other. As a result, we only kept the 100-day EMA value. This will simplify the model and hopefully avoid both overfitting and endogenous variables in the model. 
## Preprocessing 

## Model 1 @Aarush
The first model that we decided to train was Logistic regression. We used the classification report to do the analysis. It performed with about 50% accuracy & precision due to some randomness in its predication algorithm. The model has a high recall, because of bias towards increasing price which results in minimizing the False negatives. Further on the Test VS Train analysis, the results of all (accuracy, recall, F1 Score) were quite close for both. This reveals a key concept that the there is no overfitting in the model.

## Model 2 - XGB @Anuj

## Model 3 - LSTM @Jason

## Model 4  - Linear Regression ? 

# Results 

This will include the results from the methods listed above (C). You will have figures here about your results as well. No exploration of results is done here. This is mainly just a summary of your results. The sub-sections will be the same as the sections in your methods section.


We experimented with other models such as linear regression. When we plotted the actual stock prices against the predicted ones, the graph showed a close resemblance. We chose to use linear regression as a baseline for our other models. If our other models cannot outperform linear regression, we can conclude that they are not effective at predicting Tesla’s stock price. Looking at our linear regression model we had a MSE of 86.87 and a mean absolute error of 6.93. When we compare our predicted price to the actual closing price that day, we convert our prediction into a binary format: 1 if the stock is predicted to close higher, and 0 if it is predicted to close lower. Our model has a 51% accuracy. This model has very poor performance, equivalent to randomly predicting whether the stock price will increase or decrease for that day. Looking at the linear regression coefficients, the one with the most weight was the closing price, with a value of 0.98. This isn’t very surprising, as the model essentially uses the closing price to predict the next day's closing price, which is often quite similar to the previous day's closing price. If we drop the closing price from the feature list, the highest coefficient becomes 3.2 for the MACD. MACD closely follows stock prices because it is derived from the stock’s moving averages, which is inherently based on past price data. Dropping close feature linear regression achieves an accuracy of 50%. In essence, this model leverages past price history and a weighted combination of various technical indicators, which are primarily based on historical price data, to predict the next day's closing price. With an accuracy near or at 50%, this model performs as poorly as possible, effectively no better than random guessing. An accuracy below 50% would mean our predictions are consistently wrong, allowing us to reverse the predictions and achieve an accuracy of 1 minus the reported accuracy which would be above 50%. Our model seems to be non-linear, so we decided to try other models that are better suited for predicting non-linear data.

![Linear Regression Plot](/plots/linear_regression.png)

# Discussion 
This is where you will discuss the why, and your interpretation and your though process from beginning to end. This will mimic the sections you have created in your methods section as well as new sections you feel you need to create. You can also discuss how believable your results are at each step. You can discuss any short comings. It's ok to criticize as this shows your intellectual merit, as to how you are thinking about things scientifically and how you are able to correctly scrutinize things and find short comings. In science we never really find the perfect solution, especially since we know something will probably come up int he future (i.e. donkeys) and mess everything up. If you do it's probably a unicorn or the data and model you chose are just perfect for each other!

## Model 1 @Aarush
Overall, the Logistic regression model which was our first attempt is not great for continuing in the future as summarised above. 
Since the Logistic regression model could not capture the complexity involved, we made an attempt towards creating a **Neural Network** which may or may not improve in accuracy but can study the vast variations in the stock price movements. 
We have also tried [Linear Regression](https://github.com/JasonMorris1/CSE151_Tesla_Stock_Prediction/blob/main/eda_linear_regression2.ipynb) in a separate file to see potential improvement in the binary prediction after a regression on stock price values.

# Conclusion 
## This is where you do a mind dump on your opinions and possible future directions. Basically what you wish you could have done differently. Here you close with final thoughts.

# Statement of collaboration 
This is a statement of contribution by each member. This will be taken into consideration when making the final grade for each member in the group. Did you work as a team? was there a team leader? project manager? coding? writer? etc. Please be truthful about this as this will determine individual grades in participation. There is no job that is better than the other. If you did no code but did the entire write up and gave feedback during the steps and collaborated then you would still get full credit. If you only coded but gave feedback on the write up and other things, then you still get full credit. If you managed everyone and the deadlines and setup meetings and communicated with teaching staff only then you get full credit. Every role is important as long as you collaborated and were integral to the completion of the project. If the person did nothing. they risk getting a big fat 0. Just like in any job, if you did nothing, you have the risk of getting fired. Teamwork is one of the most important qualities in industry and academia!
Format: Start with Name: Title: Contribution. If the person contributed nothing then just put in writing: Did not participate in the project.

Aarush Mehrotra:

Anuj Jain:

Jason Morris:

Rishika Kejriwal:

Vandita Jain: 


