## Coding Principle
No any fallbacks, compatability, legacy code, tries
No any useless helper functions, normalizers, type checks
No any useless args， we prefer micros
Keep minimal code style, delete all useless or unused variables, functions, classes etc. 
Always write comments(chinese)
Use uv run, keep Modular, Data-Centric, Declarative config, Readability

Structure:
    Features + Models + Postprocess.
    Models only provide standard structure/train/predict interface. For ML methods, prefer sklearn style,for DL methods, prefer torch style.
    Do model experiments in JupyterNotebook when there are actual experiments.

## Core Task：Choose a portfolio(5 stocks and weights) for the next week to get best return.

Core: ranking
Data: day level k line data for each stock
Sample: (n weeks input window, 1 week for prediction and label)

Defination:
    stock_week: a natual stock week with 5 stock trading day
    stock_data: a sample that includes past history weeks as input and a future week for label
    window: the number of week that a stock_data history has

## Current Method
Feature:
    baseline 39+158
Model:
    xgboost + lambdarankic/pairwise loss
Portfolio:
    top10 from models and top5 from MeanVariance

Improvement direction:
feature: 
    use different factors/features for different windows 
    DL representation for those features
    cross-section feature between stocks
architecture:
    Mixture of features for one model?
    Mixture of models?

Current config:
    model_params bind input_window with feature_type, (input_window,feature_type)
    windows.py resolves each pair to feature_num
    utils/validation.py provides holdout and rolling_kfold validation modes
    code/models/xgboost provides standard train/predict interface
    model experiment code belongs in notebooks, not production modules
    notebook experiments use rolling_kfold, all history before each validation fold, and at least 100 rounds
