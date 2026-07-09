"""CatBoost 预测入口。"""

from .model import CatBoostRankModel


if __name__ == '__main__':
    CatBoostRankModel().predict()
