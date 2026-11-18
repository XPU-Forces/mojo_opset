"""Generate the development test report from actual JUnit and benchmark output."""

import argparse
import json
import os
import xml.etree.ElementTree as ET

from pathlib import Path


def generate(junit_files, benchmark_file, output):
    def relative_path(path):
        return os.path.relpath(Path(path).resolve(), Path(output).resolve().parent)

    benchmarks = json.loads(Path(benchmark_file).read_text())
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for path in junit_files:
        tree = ET.parse(path)
        for case in tree.iter("testcase"):
            counts["tests"] += 1
            for key, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped")):
                counts[key] += case.find(tag) is not None
    lines = [
        f"# vision_pos_embed_interp {benchmarks['environment']['device']}开发验证报告",
        "",
        "本报告由实际JUnit和profiler输出生成；不等同于缺失的官方随附测试套件验收。",
        "小图官方门槛为E2E不劣于native；下表设备耗时相对线属于当前开发脚本的额外检查。",
        "",
        f"性能采集时间：{benchmarks['started_at']} 至 {benchmarks.get('finished_at', '采集中')}。",
        "",
        f"测试：{counts['tests']}项，失败{counts['failures']}，错误{counts['errors']}，跳过{counts['skipped']}。",
        "",
        "环境：`" + json.dumps(benchmarks["environment"], ensure_ascii=False) + "`。",
        "",
        "测量：5次预热、30次有效调用，全部NPU kernel Duration(us)之和/30；E2E为另行30次同步墙钟平均。",
        "",
        f"现场fill_带宽：{benchmarks['calibration']['fill_GBs']:.3f} GB/s；"
        f"copy_带宽：{benchmarks['calibration']['copy_GBs']:.3f} GB/s。MBU分母采用fill_。",
        "",
        "|配置|native设备us|融合设备us|native/融合|MBU|门槛|单launch|结果|",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in benchmarks["cases"]:
        required = f"{row['required_MBU']:.0%}" if row["required_MBU"] is not None else "设备及E2E不劣于native"
        lines.append(
            f"|{row['name']}|{row['native']['total_device_us']:.3f}|{row['triton']['total_device_us']:.3f}|"
            f"{row['speedup']:.3f}|{row['MBU']:.2%}|{required}|{row['single_launch']}|"
            f"{'通过' if row['passed'] else '未达标'}|"
        )
    lines += [
        "",
        f"性能覆盖完整：{benchmarks.get('complete', False)}；全部性能判定通过：{benchmarks.get('passed', False)}。",
        "",
        "未达标项：",
        "",
    ]
    failures = [r for r in benchmarks["cases"] if not r["passed"]]
    if not failures:
        lines.append("本次已采集配置无未达标项。")
    for row in failures:
        lines.append(
            f"- {row['name']}：本次MBU {row['MBU']:.2%}；需结合对应原始kernel_details.csv分析，"
            "不能以native加速比替代MBU达标。"
        )
    lines += [
        "",
        "原始数据入口：",
        "",
        *[f"- JUnit：`{relative_path(p)}`" for p in junit_files],
        f"- 性能与逐kernel路径：`{relative_path(benchmark_file)}`；其目录包含calibration和逐配置原始profiler结果。",
        "",
        "本报告只覆盖上述设备与本次运行；其他平台、官方缺失套件、最终commit复现及上传状态以任务进度记录为准。",
    ]
    Path(output).write_text("\n".join(lines) + "\n")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", action="append", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(generate(args.junit, args.benchmark, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
