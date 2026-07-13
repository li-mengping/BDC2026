"""比较单模型不同候选池大小的历史组合表现。"""

import argparse
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from code.config import config
from code.features.baseline import date_group_sizes
from code.models.xgboost.model import XGBoostRankModel
from code.portfolio.postprocess import PortfolioConfig, select_portfolio
from code.utils.runtime_split import load_market_data
from code.utils.validation import build_validation_plan


def evaluate(frame, predictions, raw_df, train_returns, model_name: str, candidate_top_k: int) -> dict:
    grouped, _ = date_group_sizes(frame, int(config['min_group_size']))
    grouped = grouped.copy()
    grouped['prediction'] = np.asarray(predictions, dtype=np.float64)
    returns = []
    excess = []
    portfolio_config = PortfolioConfig(
        final_k=5,
        weight_strategy='equal',
        union_top_k_by_model=((model_name, candidate_top_k),),
    )
    for target_date, week in grouped.groupby('日期', sort=True):
        scored = week[['股票代码', '日期']].copy()
        scored[f'{model_name}_score'] = week['prediction'].to_numpy()
        _, _, selected = select_portfolio(
            scored,
            raw_df,
            target_date,
            train_returns,
            [model_name],
            portfolio_config,
        )
        labels = week.set_index('股票代码')['label']
        portfolio_return = float(sum(labels.loc[row.stock_id] * row.final_weight for row in selected.itertuples()))
        benchmark = float(week['label'].mean())
        returns.append(portfolio_return)
        excess.append(portfolio_return - benchmark)
    values = np.asarray(excess, dtype=np.float64)
    return {
        'candidate_top_k': candidate_top_k,
        'weeks': len(values),
        'return_mean': float(np.mean(returns)),
        'excess_mean': float(values.mean()),
        'excess_worst': float(values.min()),
        'excess_std': float(values.std(ddof=0)),
        'positive_rate': float(np.mean(values > 0.0)),
    }


def select_by_validation(rows: list[dict], baseline_top_k: int = 10) -> dict:
    """只用滚动 validation 选择候选，holdout 仅在选择后报告。"""
    baseline = next(row for row in rows if row['validation']['candidate_top_k'] == baseline_top_k)
    eligible = [
        row for row in rows
        if row['validation']['excess_mean'] >= baseline['validation']['excess_mean']
        and row['validation']['excess_worst'] >= baseline['validation']['excess_worst']
    ]
    if not eligible:
        return baseline
    return max(
        eligible,
        key=lambda row: (
            row['validation']['excess_mean'],
            row['validation']['excess_worst'],
            -row['validation']['excess_std'],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('.agents/loops/loop-001/evidence/portfolio_scan.json'))
    args = parser.parse_args()
    model = XGBoostRankModel()
    metadata = json.loads((model.output_dir / 'metadata.json').read_text(encoding='utf-8'))
    artifact = next(item for item in metadata['models'] if item['name'] == 'xgb_rank_pairwise')
    raw_df = load_market_data(config)
    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
    plan = build_validation_plan(raw_df, config)
    fold = plan.holdout_split()
    bindings = model.model_bindings()
    datasets = model.build_train_datasets(plan, fold, bindings, stockid2idx)
    input_window = int(artifact['input_window'])
    feature_num = artifact['feature_num']
    _, validation, holdout, features = datasets[(input_window, feature_num)]
    booster = xgb.Booster()
    booster.load_model(model.output_dir / artifact['path'])
    booster.set_param({'device': config['model_params']['xgb_rank_pairwise']['device']})

    def predict(frame):
        grouped, _ = date_group_sizes(frame, int(config['min_group_size']))
        matrix = xgb.DMatrix(
            grouped[features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32),
            missing=np.nan,
        )
        return booster.predict(matrix, iteration_range=(0, int(artifact['best_iteration']) + 1))

    validation_pred = predict(validation)
    holdout_pred = predict(holdout)
    train_returns = plan.get_train_returns(fold, input_window)
    rows = []
    for candidate_top_k in (5, 10, 15, 20):
        rows.append({
            'validation': evaluate(validation, validation_pred, raw_df, train_returns, artifact['name'], candidate_top_k),
            'holdout': evaluate(holdout, holdout_pred, raw_df, train_returns, artifact['name'], candidate_top_k),
        })
    baseline = next(row for row in rows if row['validation']['candidate_top_k'] == 10)
    selected = select_by_validation(rows)
    payload = {
        'experiment': 'xgb_portfolio_candidate_pool_scan',
        'selection_policy': 'validation_only; holdout is reported after selection and cannot select a candidate',
        'baseline': baseline,
        'rows': rows,
        'selected': selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
