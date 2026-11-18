import pytest
from unittest.mock import Mock

from lenpro.statistics.statistics import (
    Statistic,
    CountStatistic,
    EstimatorStatistic,
)

def test_statistic_is_abstract():
    with pytest.raises(TypeError):
        Statistic() # cannot instantiate abstract class

def test_count_statistic_counts_non_none():
    s = CountStatistic()
    data = ["a", None, "b", 0, "", None]
    assert s.evaluate(data) == 4.0 # everything except None assert isinstance(s.evaluate(data), float)

def test_count_statistic_on_empty_data():
    s = CountStatistic()
    assert s.evaluate([]) == 0.0

def test_count_statistic_to_dict():
    s = CountStatistic()
    assert s.to_dict() == {"name": "Count"}

def test_estimator_statistic_calls_underlying_estimator():
    estimator = Mock()
    estimator.estimate.return_value = 7
    estimator.to_dict.return_value = {"name": "MockEstimator"}
    s = EstimatorStatistic(estimator) 
    data = ["x", "y"] 
    result = s.evaluate(data) 
    assert result == 7.0 
    estimator.estimate.assert_called_once_with(data)

def test_estimator_statistic_to_dict_includes_estimator_dict():
    estimator = Mock()
    estimator.to_dict.return_value = {"name": "MyEstimator", "param": 3}
    s = EstimatorStatistic(estimator) 
    assert s.to_dict() == { 
        "name": "Estimator", 
        "estimator": {
            "name": "MyEstimator", 
            "param": 3
            }, 
        }