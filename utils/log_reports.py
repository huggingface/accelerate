import json
from pathlib import Path 
import subprocess

failed = []
passed = []

group_info = []
result_table = '| Test Section | All Passed? | Number of Failures |\n|---|---|---|\n'

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
                        failed.append([test, duration])
                    else:
                        passed.append([test, duration])

    group_info.append([str(log), section_num_failed])

if len(failed) > 0:
    result = "## Failed Tests:\n"
    failed_table = '| Test Location | Test Class | Test Name |\n|---|---|---|\n| '
    for test in failed:
        failed_table += ' | '.join(test[0].split("::"))
    failed_table += " |"
    result += failed_table
    print(result)