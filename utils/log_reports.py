import json, os
import torch
from tabulate import tabulate
from pathlib import Path
from datetime import date

failed = []

group_info = []

torch_version = torch.__version__

total_num_failed = 0
for log in Path().glob("*.log"):
    section_num_failed = 0
    with open(log, "r") as f:
        for line in f:
            line = json.loads(line)
            if line.get("nodeid", "") != "":
                test = line["nodeid"]
                if line.get("duration", None) is not None:
                    duration = f'{line["duration"]:.4f}'
                    if line.get("outcome", "") == "failed":
                        section_num_failed += 1
                        failed.append([test, duration, log.name.split('_')[0]])
                        total_num_failed += 1
    group_info.append([str(log), torch_version, section_num_failed, failed])
    failed = []
    log.unlink()
message = ""
if total_num_failed > 0:
    for name, num_failed, failed_tests in group_info:
        if num_failed > 0:
            if len(num_failed) == 1:
                message += f"*{name}: {num_failed} failed test*\n"
            else:
                message += f"*{name}: {num_failed} failed tests*\n"
            failed_table = []
            for test in failed_tests:
                failed_table += test[0].split("::")
            
            failed_table = tabulate(failed_table, headers=["Test Location", "Test Case", "Test Name"], showindex="always", tablefmt="grid", maxcolwidths=[12,12,12])
            message += failed_table
    print(f'### {message}')
else:
    message = "No failed tests! 🤗"
    print(f'## {message}')

if os.environ.get("TEST_TYPE", "") != "":
    from slack_sdk import WebClient
    message = f'*Nightly {os.environ.get("TEST_TYPE")} test results for {date.today()}:*\n{message}'

    client = WebClient(token=os.environ['SLACK_API_TOKEN'])
    client.chat_postMessage(channel='#accelerate-ci-daily', text=message)