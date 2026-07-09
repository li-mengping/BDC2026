"""LightGBM 训练入口。"""

from .model import LightGBMLambdaRankModel


if __name__ == '__main__':
    LightGBMLambdaRankModel().train()
