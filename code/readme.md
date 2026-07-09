## Coding Principle
- No any fallbacks, compatability, legacy code, tries
- Keep minimal code style, no any useless helper functions, normalizers, type checks, delete all useless or unused variables, functions, classes etc. 
- Always write comments(chinese)
- Keep Modular, Data-Centric, Declarative config, Readability

## Experiment Principle
- Always use local uv venv, always use GPU/cuda(pick less used).
- Always do experiments with time rolling kfold at validation.py to test new features or models.
- Use subagent to monitor experiments to avoid frequent checks or just handover the button to me.
- Write the params/settings of model that can possibly be used later under code/models/configs/*.yaml, when experimenting, keep that config file to SOTA.

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
    baseline 39
Model:
    xgboost + pairwise loss
Portfolio:
    top10 from models and top5 from MeanVariance

current experiments shows input window 12 with 30 features wins a good result, and feature length for such model should not be very long. the quality of factor matters more than numbers.

next move:
    try different ranking model and loss.
    try use DL model to get better features
