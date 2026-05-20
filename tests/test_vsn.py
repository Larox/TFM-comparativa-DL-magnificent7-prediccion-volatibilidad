import pytest

pf = pytest.importorskip("pytorch_forecasting")

from tfm_volatility.interpret.vsn import rank_features_from_vsn_dict


def test_rank_features_from_vsn_dict_returns_normalized_table():
    vsn_dict = {
        "AAPL": {"VIX": 0.5, "log_rv": 1.5, "FED_FUNDS": 0.2},
        "MSFT": {"VIX": 0.7, "log_rv": 1.3, "FED_FUNDS": 0.1},
    }
    ranking = rank_features_from_vsn_dict(vsn_dict)
    assert set(ranking.columns) >= {"ticker", "feature", "weight", "rank"}
    aapl = ranking[ranking["ticker"] == "AAPL"].sort_values("rank")
    assert aapl.iloc[0]["feature"] == "log_rv"
