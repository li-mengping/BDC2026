"""CatBoost 训练入口。"""

from .model import CatBoostRankModel


if __name__ == '__main__':
    CatBoostRankModel().train()
