# StockLearner
Deep learning for China A share stock data

Purpose:
User machine learning to perdict stock price

Requirements:
tensorflow >= 1.8.0
pandas >= 0.23.2
numpy >= 0.23.2

Usage:
1. Modify app.config to set the training data path, eval data path and model config path
2. Create or modify your own model config ini file and put it into the model config path
3. Run main.py

Models:
We currently support MLP and LSTM only

Data:
Please use CSV file with columns as below:
"DATE", "OPEN", "HIGH", "CLOSE", "LOW", "VOLUME", "PRICE_CHANGE", "P_CHANGE", "TURNOVER", "LABEL"
*Date will be removed during training
or you can modify /feed/csv_data.py to support your own columns

Known issue:
Multiprocessing has issues, so only place one model config file and train one time. This will be fixed in future

TODO:
Will add batch normalization in MLP soon
