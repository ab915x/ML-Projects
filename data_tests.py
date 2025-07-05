from typing import Dict, List, Union
from evidently.metrics import *
from evidently.presets import (
    DataSummaryPreset,
    DataDriftPreset,
    ClassificationPreset
)
from evidently.tests import *
import pandas as pd
from evidently import Report

def test_and_report_inference_data(current):
    reference = pd.read_csv("reference_data.csv")

    report = Report(metrics=[
        DataSummaryPreset(),
        DataDriftPreset(method="psi"),
        ],
        include_tests=True
        )
    report_snapshot = report.run(
        reference_data=reference,
        current_data=current,
    )
    report_snapshot.save_html("reports/data_report.html")
    report_snapshot.save_json("reports/data_report.json")
    return True
    tests = TestSuite(tests=[
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