import json
import multiprocessing as mp
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from ..features.baseline import preprocess_stock_data_samples
from ..ranker.config import config
from ..utils.runtime_split import load_market_data, split_runtime_data
from .postprocess import DEFAULT_PORTFOLIO_CONFIG, PortfolioConfig, select_portfolio


class PortfolioValidator:
    """验证后处理参数。"""

    def __init__(self) -> None:
        """加载元数据，只定义运行范围。"""
        self.output_dir = Path(config['output_dir'])
        metadata_path = self.output_dir / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f'metadata file not found: {metadata_path}')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        self.raw_df = load_market_data(config)
        self.runtime = split_runtime_data(self.raw_df, config)
        self.stockid2idx = {sid: idx for idx, sid in enumerate(self.metadata['stock_ids'])}
        self.loaded_models = self.load_models()

    def portfolio_configs(self) -> list[tuple[str, PortfolioConfig]]:
        """固定验证 pairwise top10 候选池，再后处理选 Top5。"""
        return [(
            'union_xgb_rank_pairwise10_mvselect_equal',
            replace(
                DEFAULT_PORTFOLIO_CONFIG,
                weight_strategy='equal',
                model_subset=None,
                union_top_k_by_model=(
                    ('xgb_rank_pairwise', 10),
                ),
                recall_model=None,
                recall_top_k=None,
                rerank_model=None,
                rerank_top_k=None,
                risk_aversion=12.0,
                selection_max_weight=0.20,
                covariance_lookback_days=40,
                covariance_shrinkage=0.1,
            ),
        )]

    def load_models(self) -> list[tuple[dict, xgb.Booster]]:
        """加载训练好的排序模型。"""
        loaded = []
        configured_names = set(config.get('model_names') or [model['name'] for model in self.metadata['models']])
        for model in self.metadata['models']:
            if model['name'] not in configured_names:
                continue
            booster = xgb.Booster()
            booster.load_model(self.output_dir / model['path'])
            model_params = config['xgb_params'].get(model['name'], model.get('xgb_params', {}))
            booster.set_param({'device': model_params.get('device', 'cpu')})
            loaded.append((model, booster))
        if not loaded:
            raise ValueError('no configured models found in metadata')
        return loaded

    def score_samples(self, samples) -> pd.DataFrame:
        """按给定 StockData 打模型分。"""
        sample_df, features = preprocess_stock_data_samples(
            samples,
            config['feature_num'],
            self.stockid2idx,
        )
        sample_df = sample_df.dropna(subset=['label']).sort_values(['日期', '股票代码']).reset_index(drop=True)
        x = sample_df[features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
        dmatrix = xgb.DMatrix(x, missing=np.nan)
        scored = sample_df.copy()
        for model, booster in self.loaded_models:
            iteration_range = (0, int(model['best_iteration']) + 1)
            scored[f"{model['name']}_score"] = booster.predict(dmatrix, iteration_range=iteration_range)
        return scored

    def evaluate_config(
        self,
        config_name: str,
        scored: pd.DataFrame,
        train_returns: np.ndarray,
        model_names: list[str],
        portfolio_config: PortfolioConfig,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """评估一组后处理参数。"""
        detail_frames = []
        summaries = []
        for target_date, group in scored.groupby('日期', sort=True):
            group = group.copy()
            group['股票代码'] = group['股票代码'].astype(str).str.zfill(6)
            top10, candidates, selected = select_portfolio(
                group,
                self.raw_df,
                target_date,
                train_returns,
                model_names,
                portfolio_config,
            )
            label_map = group.set_index('股票代码')['label']
            selected = selected.copy()
            selected['config'] = config_name
            selected['target_date'] = target_date
            selected['label'] = selected['stock_id'].map(label_map)
            selected['weighted_return'] = selected['label'] * selected['final_weight']
            detail_frames.append(selected)

            consensus_top5 = candidates.sort_values(
                ['consensus_score', 'source_count', 'best_model_rank', 'stock_id'],
                ascending=[False, False, True, True],
            ).head(portfolio_config.final_k).copy()
            consensus_top5['label'] = consensus_top5['stock_id'].map(label_map)

            model_returns = {}
            for name in model_names:
                score_col = f'{name}_score'
                model_top = group.sort_values([score_col, '股票代码'], ascending=[False, True])
                top5_ids = model_top.head(portfolio_config.final_k)['股票代码'].astype(str).str.zfill(6)
                model_returns[f'{name}_top5_return'] = float(label_map.reindex(top5_ids).mean())

            summaries.append({
                'config': config_name,
                'target_date': str(pd.Timestamp(target_date).date()),
                'candidate_count': int(len(candidates)),
                'portfolio_weighted_return': float(selected['weighted_return'].sum()),
                'portfolio_equal_return': float(selected['label'].mean()),
                'consensus_top5_equal_return': float(consensus_top5['label'].mean()),
                'selected_stock_ids': ','.join(selected['stock_id']),
                'selected_weights': ','.join(selected['final_weight'].astype(str)),
                **model_returns,
            })

        return pd.concat(detail_frames, ignore_index=True), pd.DataFrame(summaries)

    def run(self) -> list[dict]:
        """用 validation 选后处理参数，再只报告 holdout test。"""
        validation_scored = self.score_samples(self.runtime.get_validation_samples())
        train_returns = self.runtime.get_train_returns()
        model_names = [model['name'] for model, _ in self.loaded_models]
        configs = self.portfolio_configs()

        detail_frames = []
        summary_frames = []
        for config_name, portfolio_config in configs:
            detail_df, summary_df = self.evaluate_config(
                config_name,
                validation_scored,
                train_returns,
                model_names,
                portfolio_config,
            )
            detail_frames.append(detail_df)
            summary_frames.append(summary_df)

        validation_dir = self.output_dir / 'portfolio_validation'
        validation_dir.mkdir(parents=True, exist_ok=True)
        detail_df = pd.concat(detail_frames, ignore_index=True)
        summary_df = pd.concat(summary_frames, ignore_index=True)
        metric_cols = [col for col in summary_df.columns if col.endswith('_return')]
        averages = summary_df.groupby('config')[metric_cols].mean().add_prefix('avg_').reset_index()
        best = averages.sort_values('avg_portfolio_weighted_return', ascending=False).iloc[0].to_dict()
        best_config = dict(configs)[best['config']]

        holdout_scored = self.score_samples(self.runtime.get_test_samples())
        holdout_detail_df, holdout_summary_df = self.evaluate_config(
            best['config'],
            holdout_scored,
            train_returns,
            model_names,
            best_config,
        )
        holdout_averages = (
            holdout_summary_df
            .groupby('config')[metric_cols]
            .mean()
            .add_prefix('avg_')
            .reset_index()
        )
        holdout = holdout_averages.iloc[0].to_dict()

        detail_df.to_csv(validation_dir / 'portfolio_selections.csv', index=False)
        summary_df.to_csv(validation_dir / 'summary.csv', index=False)
        averages.to_csv(validation_dir / 'averages.csv', index=False)
        holdout_detail_df.to_csv(validation_dir / 'holdout_test_portfolio_selections.csv', index=False)
        holdout_summary_df.to_csv(validation_dir / 'holdout_test_summary.csv', index=False)
        holdout_averages.to_csv(validation_dir / 'holdout_test_averages.csv', index=False)
        with open(validation_dir / 'summary.json', 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'selection_basis': 'validation avg_portfolio_weighted_return',
                    'best_validation': best,
                    'validation_averages': averages.to_dict('records'),
                    'holdout_test_for_best_validation_config': holdout,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(averages.sort_values('avg_portfolio_weighted_return', ascending=False).to_string(index=False))
        print(f"best validation config: {best['config']} ({best['avg_portfolio_weighted_return']:.6f})")
        print(f"holdout test weighted return: {holdout['avg_portfolio_weighted_return']:.6f}")
        print(f"validation details written to: {validation_dir / 'portfolio_selections.csv'}")
        print(f"validation summary written to: {validation_dir / 'summary.csv'}")
        return summary_df.to_dict('records')


def main() -> list[dict]:
    """入口。"""
    return PortfolioValidator().run()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
