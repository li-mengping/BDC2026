from code.models.xgboost.predict import main as predict_main
from code.models.xgboost.train import main as train_main


def test_xgboost_entrypoints_are_callable():
    assert callable(train_main)
    assert callable(predict_main)
