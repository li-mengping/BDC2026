"""LightGBM LambdaRank 模型接口；使用 `LightGBMLambdaRankModel().train()` / `.predict()`。"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from ...config import config
from ..ranking_common import (
    evaluate_predictions,
    make_rank_data,
    predict_family,
    select_best_iteration,
    train_family,
)


class LightGBMLambdaRankModel:
    """LightGBM LambdaRank 排序模型。"""

    def __init__(self) -> None:
        self.family_config = config['lightgbm']
        self.output_dir = Path(self.family_config['output_dir'])

    def train_one(self, name, input_window, feature_type, feature_num, train_df, val_df, holdout_df, features) -> dict:
        """训练单个 LightGBM LambdaRank 模型并保存模型文件。"""
        model_config = dict(self.family_config['model_params'][name])
        params_config = dict(model_config)
        max_relevance = int(params_config.pop('max_relevance'))
        params = {key: value for key, value in params_config.items() if key not in {'input_window', 'feature_type', 'num_boost_round'}}
        params['label_gain'] = list(range(max_relevance + 1))

        train_data = make_rank_data(train_df, features, max_relevance)
        val_data = make_rank_data(val_df, features, max_relevance)
        holdout_data = make_rank_data(holdout_df, features, max_relevance)
        print(f"{name} Train Samples: {len(train_data.groups)} | rows={len(train_data.raw_label)}")
        print(f"{name} Validation Samples: {len(val_data.groups)} | rows={len(val_data.raw_label)}")
        print(f"{name} Holdout Test Samples: {len(holdout_data.groups)} | rows={len(holdout_data.raw_label)}")

        lgb_train = lgb.Dataset(train_data.x, label=train_data.relevance, group=train_data.groups, free_raw_data=False)
        lgb_val = lgb.Dataset(val_data.x, label=val_data.relevance, group=val_data.groups, reference=lgb_train, free_raw_data=False)
        num_boost_round = int(model_config.get('num_boost_round', self.family_config.get('num_boost_round', config['num_boost_round'])))
        booster = lgb.train(
            params=params,
            train_set=lgb_train,
            num_boost_round=num_boost_round,
            valid_sets=[lgb_train, lgb_val],
            valid_names=['train', 'validation'],
            callbacks=[lgb.log_evaluation(period=20)],
        )

        selection, validation_top5_excess_values = select_best_iteration(
            lambda iteration: booster.predict(val_data.x, num_iteration=iteration + 1),
            val_data.raw_label,
            val_data.groups,
            num_boost_round,
            f'select {name}',
        )
        best_iteration = selection['best_iteration']
        val_pred = booster.predict(val_data.x, num_iteration=best_iteration + 1)
        holdout_pred = booster.predict(holdout_data.x, num_iteration=best_iteration + 1)
        val_rank_ic, val_top5, val_top10 = evaluate_predictions(val_pred, val_data.raw_label, val_data.groups)
        holdout_rank_ic, holdout_top5, holdout_top10 = evaluate_predictions(holdout_pred, holdout_data.raw_label, holdout_data.groups)

        model_dir = self.output_dir / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        artifact = f'{name}_w{input_window}_{feature_num}_{best_iteration}'
        artifact_ext = '.txt'
        model_path = model_dir / f'{artifact}{artifact_ext}'
        booster.save_model(str(model_path))
        with open(model_dir / f'{artifact}_evals.json', 'w', encoding='utf-8') as f:
            json.dump({'validation_top5_excess_return': validation_top5_excess_values}, f, ensure_ascii=False, indent=2)

        print(f'saved {name}: {model_path}')
        print(f'{name} best iteration by validation top5 excess: {best_iteration}')
        print(f"{name} best validation top5 excess return: {selection['score']:.6f}")
        print(f'{name} validation RankIC: {val_rank_ic:.6f}')
        print(f"{name} validation pred top5 return avg: {val_top5['pred_top5_return_avg']:.6f}")
        print(f"{name} validation pred top5 excess return avg: {val_top5['pred_top5_excess_return_avg']:.6f}")
        print(f"{name} validation pred top5 precision avg: {val_top5['pred_top5_precision_avg']:.6f}")
        print(f"{name} holdout test RankIC: {holdout_rank_ic:.6f}")
        print(f"{name} holdout test pred top5 return avg: {holdout_top5['pred_top5_return_avg']:.6f}")
        print(f"{name} holdout test pred top5 excess return avg: {holdout_top5['pred_top5_excess_return_avg']:.6f}")
        print(f"{name} holdout test pred top5 precision avg: {holdout_top5['pred_top5_precision_avg']:.6f}")

        return {
            'name': name,
            'artifact_name': artifact,
            'artifact_ext': artifact_ext,
            'input_window': input_window,
            'feature_type': feature_type,
            'feature_num': feature_num,
            'features': features,
            'path': str(model_path.relative_to(self.output_dir)),
            'model_params': model_config,
            'best_iteration': best_iteration,
            'best_selection_metric': selection['metric'],
            'best_selection_score': selection['score'],
            'best_selection': selection,
            'validation_rank_ic': val_rank_ic,
            'validation_score': val_top5['pred_top5_excess_return_avg'],
            'validation_top10': val_top10,
            'validation_top5': val_top5,
            'holdout_test_rank_ic': holdout_rank_ic,
            'holdout_test_top10': holdout_top10,
            'holdout_test_top5': holdout_top5,
        }

    def train(self) -> float:
        """正式 holdout 训练入口。"""
        return train_family('lightgbm_lambdarank_models', self.family_config, self.train_one, self.output_dir)

    def predict(self) -> None:
        """加载训练 artifact，生成 output/result.csv。"""
        def score_model(model: dict, x: np.ndarray) -> np.ndarray:
            model_path = self.output_dir / model['path']
            if not model_path.exists():
                raise FileNotFoundError(f'model file not found: {model_path}')
            booster = lgb.Booster(model_file=str(model_path))
            return booster.predict(x, num_iteration=int(model['best_iteration']) + 1)

        predict_family(self.family_config, self.output_dir, score_model)
