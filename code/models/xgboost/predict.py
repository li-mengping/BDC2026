"""XGBoost 正式预测入口。"""

from .model import XGBoostRankModel


def main() -> None:
    XGBoostRankModel().predict()


if __name__ == '__main__':
    main()
