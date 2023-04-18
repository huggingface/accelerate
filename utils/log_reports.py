import json, os
from pathlib import Path
from datetime import date

failed = []
passed = []

group_info = []

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
                    else:
                        passed.append([test, duration, log.name.split('_')[0]])
    group_info.append([str(log), section_num_failed, failed])
    log.unlink()
    failed = []
message = ""
if total_num_failed > 0:
    for name, num_failed, failed_tests in group_info:
        if num_failed > 0:
            if num_failed == 1:
                message += f"*{name}: {num_failed} failed test*\n"
            else:
                message += f"*{name}: {num_failed} failed tests*\n"
            max_location = max(failed_tests, key=lambda x: len(x[0].split("::")[1])) + 2
            max_class = max(failed_tests, key=lambda x: len(x[0].split("::")[0])) + 2
            max_name = max(failed_tests, key=lambda x: len(x[0].split("::")[2]))
            failed_table = '```\n'
            failed_table += f'Test Location |'.center(max_location)
            failed_table += f'Test Class |'.center(max_class)
            failed_table += f'Test Name\n'.center(max_name)
            failed_table += f'|:{"-"*max_location}:|:{"-"*max_class}:|:{"-"*max_name}:|\n'
            for test in failed_tests:
                for i, part in test[0].split("::"):
                    if i == 0:
                        failed_table += part.center(max_location) + " | "
                    elif i == 1:
                        failed_table += part.center(max_class) + " | "
                    else:
                        failed_table += part.center(max_name) + "\n"
            # failed_table += f" | {test[2]} |"
            message += failed_table
            message += "\n```\n"
    print(f'### {message}')
else:
    message = "No failed tests! 🤗"
    print(f'## {message}')

if os.environ.get("TEST_TYPE", "") != "":
    from slack_sdk import WebClient
    message = f'*Nightly {os.environ.get("TEST_TYPE")} test results for {date.today()}:*\n{message}'

    client = WebClient(token=os.environ['SLACK_API_TOKEN'])
    client.chat_postMessage(channel='#accelerate-ci-daily', text=message)