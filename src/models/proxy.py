import litellm
import os
import json

logs_dir = os.getenv("LOGS_DIR", "/workspaces/Agent-2/logs")
os.makedirs(logs_dir, exist_ok=True)

def log_success(kwargs, completion_obj, start_time, end_time):
    with open(f"{logs_dir}/success-logs.jsonl", "a") as dest:
        dest.write(
            json.dumps({
                "kwargs": kwargs,
                "completion_obj": completion_obj,
                "start_time": start_time,
                "end_time": end_time,
            }, ensure_ascii=False, default=str) + "\n"
        )

def log_failure(kwargs, completion_obj, start_time, end_time):
    with open(f"{logs_dir}/failure-logs.jsonl", "a") as dest:
        dest.write(
            json.dumps({
                "kwargs": kwargs,
                "completion_obj": completion_obj,
                "start_time": start_time,
                "end_time": end_time,
            }, ensure_ascii=False, default=str) + "\n"
        )

litellm.success_callback = [log_success]
litellm.failure_callback = [log_failure]
