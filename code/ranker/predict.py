import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import config
from ..portfolio.postprocess import DEFAULT_PORTFOLIO_CONFIG, select_portfolio
from ..features.baseline import preprocess_stock_history_samples
from ..utils.runtime_split import load_market_data, split_runtime_data


def main() -> None:
    """加载排序模型，输出各自 topK 和候选并集。"""
    output_dir = Path(config['output_dir'])
    metadata_path = output_dir / 'metadata.json'
    output_path = Path('./output/result.csv')
    top10_path = Path('./output/top10_by_model.csv')
    candidates_path = Path('./output/candidates.csv')
    portfolio_path = Path('./output/portfolio_selection.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        raise FileNotFoundError(f'metadata file not found: {metadata_path}')

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    stock_ids = metadata['stock_ids']
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
    features = metadata['features']
    configured_names = set(config.get('model_names') or [model['name'] for model in metadata['models']])
    models = [model for model in metadata['models'] if model['name'] in configured_names]
    if not models:
        raise ValueError('no configured models found in metadata')

    raw_df = load_market_data(config)
    runtime = split_runtime_data(raw_df, config)
    latest, _ = preprocess_stock_history_samples(
        runtime.get_prediction_samples(),
        config['feature_num'],
        stockid2idx,
    )
    latest_date = runtime.test_date
    latest = latest[latest['股票代码'].isin(stockid2idx)].sort_values('股票代码').reset_index(drop=True)

    if len(latest) < 5:
        raise ValueError(f'not enough stocks on latest date {latest_date.date()}: {len(latest)}')

    x = latest[features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    dtest = xgb.DMatrix(x, missing=np.nan)

    for model in models:
        name = model['name']
        model_path = output_dir / model['path']
        if not model_path.exists():
            raise FileNotFoundError(f'model file not found: {model_path}')

        booster = xgb.Booster()
        booster.load_model(model_path)
        # 预测阶段按模型自己的参数恢复设备。
        model_params = config['xgb_params'].get(name, model.get('xgb_params', {}))
        booster.set_param({'device': model_params.get('device', 'cpu')})
        iteration_range = (0, int(model['best_iteration']) + 1)
        score_col = f'{name}_score'
        latest[score_col] = booster.predict(dtest, iteration_range=iteration_range)

    model_names = [model['name'] for model in models]
    # 各模型取验证集选择的 topK，合并候选池后由组合优化器选出最终 top5。
    top10, candidates, selected = select_portfolio(
        latest,
        raw_df,
        latest_date,
        runtime.get_train_returns(),
        model_names,
        DEFAULT_PORTFOLIO_CONFIG,
    )

    top10.to_csv(top10_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    selected.to_csv(portfolio_path, index=False)

    output_df = pd.DataFrame({
        'stock_id': selected['stock_id'].tolist(),
        'weight': selected['final_weight'].round(10).tolist(),
    })
    output_df.to_csv(output_path, index=False)

    overlap_count = int(top10['stock_id'].duplicated().sum())
    if DEFAULT_PORTFOLIO_CONFIG.union_top_k_by_model:
        topk_description = ','.join(
            f'{name}:{top_k}' for name, top_k in DEFAULT_PORTFOLIO_CONFIG.union_top_k_by_model
        )
    else:
        topk_description = f'top_k_per_model={DEFAULT_PORTFOLIO_CONFIG.top_k_per_model}'
    print(f"Prediction: target={latest_date.date()} | universe={len(latest)}")
    print(
        f"Candidate pool: model_topk_rows={len(top10)} | unique={len(candidates)} | "
        f"overlap={overlap_count} | {topk_description}"
    )
    print("Model TopK:")
    print(top10[['model', 'model_rank', 'stock_id', 'score', 'model_rank_norm']].to_string(index=False))
    print("Final Portfolio:")
    print(selected[['final_rank', 'stock_id', 'source_models', 'final_weight', 'consensus_score', 'mu']].to_string(index=False))
    print(f'result written to: {output_path}')
    print(f'top10 written to: {top10_path}')
    print(f'candidates written to: {candidates_path}')
    print(f'portfolio selection written to: {portfolio_path}')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
