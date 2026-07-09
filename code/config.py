"""全局声明式配置；模型配置从 `code/models/configs/*.yaml` 加载。"""

from pathlib import Path

import yaml


MODEL_CONFIG_DIR = Path(__file__).resolve().parent / 'models' / 'configs'
MODEL_FAMILY_DEFAULTS = {
    'xgboost': {
        'output_dir': './model/xgboost',
        'num_boost_round': 200,
    },
    'lightgbm': {
        'output_dir': './model/lightgbm',
        'num_boost_round': 200,
    },
    'catboost': {
        'output_dir': './model/catboost',
        'num_boost_round': 200,
    },
}


def _read_model_yaml(path: Path) -> dict:
    """读取单个模型 YAML。"""
    with path.open('r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f'model config must be a mapping: {path}')
    return data


def _merge_family_value(family_config: dict, key: str, value, path: Path) -> None:
    """合并同一模型族的共享配置，冲突时显式报错。"""
    if value is None:
        return
    if key in family_config and family_config[key] != value:
        raise ValueError(
            f'conflicting {key} for model family {family_config["family"]}: '
            f'{family_config[key]!r} vs {value!r} in {path}'
        )
    family_config[key] = value


def load_model_configs(config_dir: Path = MODEL_CONFIG_DIR) -> tuple[dict, dict[str, str]]:
    """加载所有启用的模型 YAML，并按模型族整理成旧版配置结构。"""
    if not config_dir.exists():
        raise FileNotFoundError(f'model config directory not found: {config_dir}')

    families = {}
    config_files = {}
    for path in sorted(config_dir.glob('*.yaml')):
        data = _read_model_yaml(path)
        if not data.get('enabled', True):
            continue

        name = str(data.get('name') or path.stem)
        family = str(data.get('family') or '').strip()
        if family not in MODEL_FAMILY_DEFAULTS:
            raise ValueError(f'unsupported model family for {name}: {family!r}')

        features = data.get('features') or {}
        params = data.get('params') or {}
        if not isinstance(features, dict) or not isinstance(params, dict):
            raise ValueError(f'features/params must be mappings in {path}')
        for required in ('input_window', 'feature_type'):
            if required not in features:
                raise ValueError(f'missing features.{required} in {path}')

        family_config = families.setdefault(
            family,
            {
                'family': family,
                'model_names': [],
                'model_params': {},
            },
        )
        _merge_family_value(family_config, 'output_dir', data.get('output_dir'), path)

        if name in family_config['model_params']:
            raise ValueError(f'duplicate model config name in {config_dir}: {name}')
        family_config['model_names'].append(name)
        model_config = {**features, **params}
        if 'num_boost_round' in data:
            model_config['num_boost_round'] = data['num_boost_round']
        family_config['model_params'][name] = model_config
        config_files[name] = str(path.relative_to(Path(__file__).resolve().parent.parent))

    if not families:
        raise ValueError(f'no enabled model YAML files found in {config_dir}')

    for family, family_config in families.items():
        family_config.setdefault('output_dir', MODEL_FAMILY_DEFAULTS[family]['output_dir'])
        family_config.setdefault('num_boost_round', MODEL_FAMILY_DEFAULTS[family]['num_boost_round'])
        family_config.pop('family', None)
    return families, config_files


_model_families, _model_config_files = load_model_configs()
_xgboost_config = _model_families.get('xgboost', {
    'output_dir': MODEL_FAMILY_DEFAULTS['xgboost']['output_dir'],
    'num_boost_round': MODEL_FAMILY_DEFAULTS['xgboost']['num_boost_round'],
    'model_names': [],
    'model_params': {},
})

config = {
    # 数据路径固定，训练和预测使用同一份全量行情。
    'data_path': './data',
    'full_data_file': 'stock_data.csv',

    # 模型配置来源。
    'model_config_dir': str(MODEL_CONFIG_DIR),
    'model_config_files': _model_config_files,

    # XGBoost 仍保留旧版顶层键，兼容现有训练/预测/实验代码。
    'output_dir': _xgboost_config['output_dir'],
    'num_boost_round': _xgboost_config['num_boost_round'],
    'model_names': _xgboost_config['model_names'],
    'model_params': _xgboost_config['model_params'],

    # 所有窗口单位都是交易周，test_date 是目标周起点，目标周行情未知。
    'start_date': '2023-01-02',
    'test_date': '2026-06-29',
    'min_group_size': 30,  # 只训练/评估至少 30 只股票特征和标签有效的目标周。
    'seed': 42,

    'validation': {
        # holdout 是正式训练模式；rolling_kfold 供 notebook 实验使用，不预留 test weeks。
        'mode': 'holdout',
        'num_validation_weeks': 8,
        'num_test_weeks': 4,
        'rolling_folds': 4,
        'rolling_validation_weeks': 4,
        'rolling_gap_weeks': 0,
        'rolling_min_train_weeks': 120,
    },
    'xgboost': _xgboost_config,
    'lightgbm': _model_families.get('lightgbm', {
        'output_dir': MODEL_FAMILY_DEFAULTS['lightgbm']['output_dir'],
        'num_boost_round': MODEL_FAMILY_DEFAULTS['lightgbm']['num_boost_round'],
        'model_names': [],
        'model_params': {},
    }),
    'catboost': _model_families.get('catboost', {
        'output_dir': MODEL_FAMILY_DEFAULTS['catboost']['output_dir'],
        'num_boost_round': MODEL_FAMILY_DEFAULTS['catboost']['num_boost_round'],
        'model_names': [],
        'model_params': {},
    }),
}
