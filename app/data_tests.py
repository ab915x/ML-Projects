from evidently.presets import DataSummaryPreset, DataDriftPreset
import pandas as pd
from evidently import Report
import json


def test_and_report_inference_data(current):
    reference = pd.read_csv("app/reference_data.csv")

    report = Report(
        metrics=[
            DataSummaryPreset(),
            DataDriftPreset(method="psi"),
        ],
        include_tests=True,
    )
    report_snapshot = report.run(
        reference_data=reference,
        current_data=current,
    )
    report_snapshot.save_html("app/reports/data_report.html")
    with open("app/reports/data_report.json", "w") as f:
        json.dump(report_snapshot.json(), f)

    report_dict = json.loads(report_snapshot.json())
    status = all([test["status"] == "SUCCESS" for test in report_dict["tests"]])
    if status:
        print("Data test passed successfully, retraining will be executed")
        return True
    print("Invalid data detected, no retraining will be executed")
    return False
