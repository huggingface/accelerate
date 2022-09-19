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
    if section_num_failed > 0:
        success = ":no_entry:"
    else:
        success = ":heavy_checkmark:"
    result_table += f'| {str(log).replace(".log","").title()} | {success} | {section_num_failed} |\n'
    total_num_failed += section_num_failed
    group_info.append([str(log), section_num_failed])

result = "## Overall Results:\n" + result_table 
result += "\n## Failed Tests:\n"

failed_table = '| Test Location | Test Class | Test Name |\n|---|---|---|\n| '
for test in failed:
    failed_table += ' | '.join(test[0].split("::"))
failed_table += " |"
result += failed_table

subprocess.run(["echo", result, ">>", "$GITHUB_STEP_SUMMARY"])