import h2o
from h2o.estimators import H2OGeneralizedLinearEstimator

from tests import pyunit_utils


def pubdev_5265():
    # The MeanImputation strategy actually counts 'mode' in case of GLM.
    # There are more Bs, so the first 'nan' value should be replaced by '2'
    data_missing_A = {
        'first': ['A', 'A', 'A', 'A', 'A',
                  'B', 'B', 'B', 'B', 'B', 'B'],
        'second': ['nan', 1, 1, 1, 1,
                   2, 2, 2, 2, 2, 2]
    }

    data_missing_A = h2o.H2OFrame(data_missing_A)
    data_missing_A['second'] = data_missing_A['second'].asfactor()

    a_estimator = H2OGeneralizedLinearEstimator(family="binomial", missing_values_handling="MeanImputation", seed=1234)
    a_estimator.train(x=["second"], y="first", training_frame=data_missing_A)
    prediction = a_estimator.predict(test_data=data_missing_A)
    grouped = prediction.group_by((0)).count(na="all").get_frame().as_data_frame(use_pandas=False)
    # 11 rows in total, 4 with A, 7 with B (6 in original data)
    assert grouped == [['predict', 'nrow'], ['A', '4'], ['B', '7']]

    # There are more As, so the first 'nan' value should be replaced by '1'
    data_missing_B = {
        'first': ['A', 'A', 'A', 'A', 'A',
                  'B', 'B', 'B'],
        'second': ['nan', 1, 1, 1, 1,
                   2, 2, 2]
    }

    data_missing_B = h2o.H2OFrame(data_missing_B)
    data_missing_B['second'] = data_missing_B['second'].asfactor()

    a_estimator = H2OGeneralizedLinearEstimator(family="binomial", missing_values_handling="MeanImputation", seed=1234)
    a_estimator.train(x=["second"], y="first", training_frame=data_missing_B)
    prediction = a_estimator.predict(test_data=data_missing_B)
    grouped = prediction.group_by((0)).count(na="all").get_frame().as_data_frame(use_pandas=False)
    # 8 rows in total, 5 with A (4 in original data), 3 rows with B
    assert grouped == [['predict', 'nrow'], ['A', '5'], ['B', '3']]

if __name__ == "__main__":
    pyunit_utils.standalone_test(pubdev_5265)
else:
    pubdev_5265()
