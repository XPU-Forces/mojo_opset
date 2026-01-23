import os
import shutil

from typing import Dict

from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


def normalize_parameters(param_str: str) -> str:
    """
    Normalize parameter string to ensure consistent keys for deduplication.
    1. Replace newlines with spaces.
    2. Remove extra spaces inside string.
    3. Strip leading/trailing whitespaces.
    Example:
        "  gate_out: Tensor...\n  up_out: Tensor...  "
        -> "gate_out: Tensor... up_out: Tensor..."
    """

    no_newline = param_str.replace("\n", " ").replace("\r", " ")
    normalized = " ".join(no_newline.split())

    return normalized


def reorganize_benchmark_md():
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.md")

    if not os.path.exists(log_path):
        return

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 2:
        return

    header = lines[:2]
    content_lines = [line.strip() for line in lines[2:] if line.strip()]
    unique_entries: Dict[tuple, dict] = {}

    for line in content_lines:
        parts = line.split("|")

        if len(parts) >= 6:
            timestamp = parts[1].strip()
            op_name = parts[2].strip()
            raw_parameters = parts[3].strip()

            norm_params = normalize_parameters(raw_parameters)
            key = (op_name, norm_params)
            unique_entries[key] = {"line": line, "timestamp": timestamp, "op_name": op_name}

    parsed_lines = list(unique_entries.values())

    parsed_lines.sort(key=lambda x: (x["op_name"], x["timestamp"]))

    temp_path = log_path + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.writelines(header)
            for item in parsed_lines:
                f.write(item["line"] + "\n")

        shutil.move(temp_path, log_path)
        logger.info(f"Reorganized benchmark log: kept {len(parsed_lines)} unique entries.")
    except Exception as e:
        logger.error(f"Failed to reorganize benchmark log: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def pytest_sessionfinish(session, exitstatus):
    if not hasattr(session.config, "workerinput"):
        reorganize_benchmark_md()
