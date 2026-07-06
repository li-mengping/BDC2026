"""全局声明式配置；训练、预测和实验统一从 `code.config import config` 读取。"""

config = {
    # 数据路径固定，训练和预测使用同一份全量行情。
    'data_path': './data',
    'full_data_file': 'stock_data.csv',

    # 输出目录保存按 模型_特征_迭代 命名的排序模型。
    'output_dir': './model/xgboost',

    # 所有窗口单位都是交易周，test_date 是目标周起点，目标周行情未知。
    'start_date': '2023-01-02',
    'test_date': '2026-06-29',
    'min_group_size': 30,  # 只训练/评估至少 30 只股票特征和标签有效的目标周。
    'seed': 42,

    'num_boost_round': 200,
    'validation': {
        # holdout 是正式训练模式；rolling_kfold 供 notebook 实验使用，不预留 test weeks。
        'mode': 'rolling_kfold',
        'num_validation_weeks': 8,
        'num_test_weeks': 4,
        'rolling_folds': 4,
        'rolling_validation_weeks': 4,
        'rolling_gap_weeks': 0,
        'rolling_min_train_weeks': 120,
    },
    'model_names': [
        # 'lambdarankic',
        'xgb_rank_pairwise',
    ],
    'model_params': {
        'lambdarankic': {
            'input_window': 12,
            'feature_type': '158+39+window_multi_cross+xsec',
            'device': 'cuda',
            'booster': 'gbtree',
            'tree_method': 'hist',
            'max_depth': 5,
            'eta': 0.025,
            'subsample': 0.8,
            'colsample_bytree': 0.75,
            'min_child_weight': 30.0,
            'lambda': 18.0,
            'alpha': 0.2,
            'max_delta_step': 1.0,
            'seed': 42,
            'verbosity': 1,
        },
        'xgb_rank_pairwise': {
            'input_window': 12,
            'feature_type': '158+39+window_multi_cross+xsec',
            'device': 'cuda',
            'booster': 'gbtree',
            'objective': 'rank:pairwise',
            'tree_method': 'hist',
            'max_depth': 5,
            'eta': 0.025,
            'subsample': 0.8,
            'colsample_bytree': 0.75,
            'min_child_weight': 30.0,
            'lambda': 18.0,
            'alpha': 0.2,
            'max_delta_step': 1.0,
            'seed': 42,
            'verbosity': 1,
        },
    },
}
