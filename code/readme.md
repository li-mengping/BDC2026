## Coding Principle
- No any fallbacks, compatability, legacy code, tries
- Keep minimal code style, no any useless helper functions, normalizers, type checks, delete all useless or unused variables, functions, classes etc. 
- Always write comments(chinese)
- Keep Modular, Data-Centric, Declarative config, Readability

## Experiment Principle
- Always use local uv venv, always use GPU/cuda(pick less used).
- Always do experiments in JupyterNotebook, use time rolling kfold at validation.py to test new features or models.
- Utilize the stored variable in JupyterNotebook to avoid reloading data and redoing preprocessing, use subagent to monitor experiments to avoid frequent checks or just handover the button to me.

## Structure:
    Features + Models + Postprocess.
    Models only provide standard structure/train/predict interface. For ML methods, prefer sklearn style,for DL methods, prefer torch style.

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
