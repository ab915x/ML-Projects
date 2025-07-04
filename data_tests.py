from typing import Dict, List, Union
from evidently.metrics import *
from evidently.metric_preset import (
    DataQualityPreset,
    DataDriftPreset,
    RegressionPreset
)
from evidently.test_suite import TestSuite
from evidently.tests import *
from evidently import ColumnMapping

def test_and_report_inference_data(current):
    reference = pd.read_csv("reference_data.csv")

    report = Report(metrics=[
        DataQualityPreset(),
        DataDriftPreset(),
        RegressionPreset()
    ])
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=ColumnMapping(
            numerical_features=[
                'length', 'num_uppercase', 'num_lowercase',
                'num_digits', 'num_special', 'unique_chars', 'entropy'
            ],
            target='target'
        )
    )
    report.save_html("data_report.html")
    report.save_json("data_report.json")

    tests = TestSuite(tests=[
        TestValueRMSE(),  
        TestShareOfMissingValues(),  
        TestNumberOfConstantColumns(),  
        TestShareOfDriftedColumns(),  
        TestNumberOfOutRangeValues()  
    ])
    
    tests.run(
        reference_data=reference,
        current_data=current,
        column_mapping=ColumnMapping(
            numerical_features=[
                'length', 'num_uppercase', 'num_lowercase',
                'num_digits', 'num_special', 'unique_chars', 'entropy'
            ]
        )
    )
    
    tests.save_html("tests.html")
    tests.save_json("tests.json")
    if tests.as_dict()["summary"]["failed_tests"] == 0:    
        print("Data test passed successfully, retraining will be executed")
        return True   
    print("Invalid data detected, no retraining will be executed")
    return False