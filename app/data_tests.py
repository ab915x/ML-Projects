from evidently.presets import DataSummaryPreset, DataDriftPreset
import pandas as pd
from evidently import Report
import json
import os


def test_and_report_inference_data(current):
    reference_path = "data/reference_data.csv"
    report_dir = "reports"

    os.makedirs(report_dir, exist_ok=True)

    reference = pd.read_csv(reference_path)

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

    report_snapshot.save_html(f"{report_dir}/data_report.html")
    with open(f"{report_dir}/data_report.json", "w") as f:
        json.dump(report_snapshot.json(), f)

    report_dict = json.loads(report_snapshot.json())
    status = all([test["status"] == "SUCCESS" for test in report_dict["tests"]])
    if status:
        print("Data test passed successfully, retraining will be executed")
        return True
    print("Invalid data detected, no retraining will be executed")
    return False
