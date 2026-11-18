"""Summarize measured pipeline counters without replacing the task's MBU test."""

import argparse
import csv
import json
import statistics

from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.benchmark.read_text())
    lines = [
        f"# {data['environment']['device']} 性能剖析",
        "",
        f"数据时间：{data['started_at']}。下表从该轮kernel_details.csv生成，未重新测量。",
        "各流水线可重叠，比例不能相加；均值不能等同于整卡硬件极限。",
        "",
        "|配置|实际设备us|达标所需us上限|MBU|AIV向量比例|AIV标量比例|MTE2比例|MTE3比例|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data["cases"]:
        files = list((args.benchmark.parent / row["name"] / "triton").rglob("kernel_details.csv"))
        if len(files) != 1:
            raise RuntimeError(f"expected one kernel CSV for {row['name']}: {files}")
        with files[0].open() as stream:
            records = list(csv.DictReader(stream))
        if len(records) != data["measurement"]["repeats"]:
            raise RuntimeError(f"incomplete single-launch records for {row['name']}")
        counters = []
        for field in ("aiv_vec_ratio", "aiv_scalar_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio"):
            values = [float(r[field]) for r in records if r.get(field) not in (None, "", "N/A")]
            counters.append(f"{statistics.mean(values):.1%}" if values else "未提供")
        required_us = (
            row["read_write_bytes"] / data["calibration"]["fill_GBs"] / 1000 / row["required_MBU"]
            if row["required_MBU"] is not None
            else None
        )
        limit = f"{required_us:.3f}" if required_us is not None else "相对native"
        lines.append(
            f"|{row['name']}|{row['triton']['total_device_us']:.3f}|{limit}|{row['MBU']:.2%}|"
            + "|".join(counters)
            + "|"
        )
    lines += [
        "",
        "## 分析与优化方向",
        "",
        "1. 当前实现以空间位置分配工作，邻域读取后按输出dtype执行四次乘法与三次加法。"
        "BF16乘加的逐步舍入必须保留；不能以FP32累加替换来降低指令数。"
        "应检查cast、broadcast、尾块补齐和重复邻域载入，结合具体执行模式解释计数。",
        (
            "2. A5短输入采用二维SIMT分组，复用坐标与系数并展开特征组；"
            "长视频保留flat时间复用路径。BF16逐步舍入显式保留，禁用FMA；"
            "多图按前缀和在一个kernel中分派。"
            "SIMT路径下AIV向量比例不能替代SIMT指令利用率，MTE比例接近零也不表示没有GM访存。"
            if data["environment"]["target"].startswith("npu.a5")
            else "2. A2/A3按空间位置分工，连续输入使用实际特征宽度，避免非必要填充与分段；"
            "视频复用局部空间值，在UB预算允许时使用二维时间写出。"
            "多图在一个kernel中按主机元数据写入各自切片。"
        ),
        "3. 更大二维tile、循环展开与对齐提示未必提高性能，须结合该轮原始输出核对。"
        "后续按架构检查BF16指令、特征维映射和多核分工；调整后仅在实测确认后报告收益。",
        "",
        "尚未达标的配置以测试报告逐项结果为准。任务书要求未达标配置附剖析并由验收方裁定；"
        "本报告不替代达标判定，也不宣称已获得豁免。",
    ]
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
