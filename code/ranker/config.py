config = {
    # 数据路径固定，训练和预测使用同一份全量行情。
    'data_path': './data',
    'full_data_file': 'stock_data.csv',

    # 输出目录只服务当前 LambdaRankIC-XGBoost 方法。
    'output_dir': './model/lambdarankic_xgb',

    # 所有窗口单位都是交易周，test_date 是目标周起点，目标周行情未知。
    'start_date': '2023-01-02',
    'test_date': '2026-06-29',
    'feature_num': '158+39+window_multi_cross+xsec',
    'input_window': 12,
    'num_validation_weeks': 8, # 用于调参/选迭代。
    'num_test_weeks': 4, # 只用于最终 holdout 表现报告。
    'min_group_size': 30, # only train/evaluate a target week if at least 30 stocks have valid features and labels for that week.
    'seed': 42,

    'num_boost_round': 300,
    'rolling_cv_folds': 4,
    'rolling_cv_validation_weeks': 4,
    'rolling_cv_gap_weeks': 1,
    'rolling_cv_min_train_weeks': 120,
    'rolling_cv_num_boost_round': 150,
    'model_names': [
        'lambdarankic',
        'xgb_rank_pairwise',
    ],
    'xgb_params': {
        'lambdarankic': {
            'device': 'cuda',
            'booster': 'gbtree',
            'tree_method': 'hist',
            "max_depth": 5,
            "eta": 0.025,
            "subsample": 0.8,
            "colsample_bytree": 0.75,
            "min_child_weight": 30.0,
            "lambda": 18.0,
            "alpha": 0.2,
            "max_delta_step": 1.0,
            'seed': 42,
            'verbosity': 1,
        },
        'xgb_rank_pairwise': {
            'device': 'cuda',
            'booster': 'gbtree',
            'objective': 'rank:pairwise',
            'tree_method': 'hist',
            "max_depth": 5,
            "eta": 0.025,
            "subsample": 0.8,
            "colsample_bytree": 0.75,
            "min_child_weight": 30.0,
            "lambda": 18.0,
            "alpha": 0.2,
            "max_delta_step": 1.0,
            'seed': 42,
            'verbosity': 1,
        },
    },
}
