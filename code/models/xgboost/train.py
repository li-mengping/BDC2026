"""XGBoost 正式训练入口。"""

from .model import XGBoostRankModel


def main() -> None:
    XGBoostRankModel().train()


if __name__ == '__main__':
    main()
