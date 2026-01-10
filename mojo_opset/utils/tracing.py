import json
import time


class MojoTracingGenerator:
    def __init__(self, trace_name="Custom Timeline"):
        """
        初始化 Chrome tracing 生成器

        Args:
            trace_name (str): timeline 名称
        """
        self.events = []
        self.trace_name = trace_name
        self.process_names = {}  # 存储进程名称映射
        self.thread_names = {}  # 存储线程名称映射

    def set_process_name(self, process_id, name):
        """
        设置进程名称

        Args:
            process_id (int): 进程ID
            name (str): 进程名称
        """
        self.process_names[process_id] = name

    def set_thread_name(self, process_id, thread_id, name):
        """
        设置线程名称

        Args:
            process_id (int): 进程ID
            thread_id (int): 线程ID
            name (str): 线程名称
        """
        self.thread_names[(process_id, thread_id)] = name

    def add_metadata_events(self):
        """添加进程和线程名称的元数据事件"""

        for pid, name in self.process_names.items():
            process_name_event = {
                "name": "process_name",
                "ph": "M",  # Metadata event
                "pid": pid,
                "tid": 0,  # 对于进程元数据，线程ID通常为0
                "args": {"name": name},
            }
            self.events.append(process_name_event)

        # 添加线程名称元数据:cite[7]
        for (pid, tid), name in self.thread_names.items():
            thread_name_event = {
                "name": "thread_name",
                "ph": "M",  # Metadata event
                "pid": pid,
                "tid": tid,
                "args": {"name": name},
            }
            self.events.append(thread_name_event)

        # 添加timeline名称元数据
        timeline_name_event = {
            "name": "trace_name",
            "ph": "M",
            "pid": 0,
            "tid": 0,
            "args": {"name": self.trace_name},
        }
        self.events.append(timeline_name_event)

    def add_event(
        self,
        name,
        categories,
        event_type,
        timestamp,
        duration=None,
        process_id=0,
        thread_id=0,
        args=None,
    ):
        """
        添加一个事件到 timeline

        Args:
            name (str): 事件名称
            categories (str or list): 事件类别
            event_type (str): 事件类型 ('B'=开始, 'E'=结束, 'X'=完整事件, 'i'=即时事件等)
            timestamp (float): 时间戳（秒）
            duration (float, optional): 持续时间（秒），仅对 'X' 类型事件需要
            process_id (int): 进程ID
            thread_id (int): 线程ID
            args (dict, optional): 额外参数
        """
        event = {
            "name": name,
            "cat": ",".join(categories) if isinstance(categories, list) else categories,
            "ph": event_type,
            "ts": timestamp * 1000000,  # 转换为微秒
            "pid": process_id,
            "tid": thread_id,
        }

        if event_type == "X" and duration is not None:
            event["dur"] = duration * 1000000  # 转换为微秒

        if args:
            event["args"] = args

        self.events.append(event)

    def save_to_file(self, filename="trace.json"):
        """
        保存 timeline 到 JSON 文件

        Args:
            filename (str): 输出文件名
        """
        # 添加元数据事件
        self.add_metadata_events()

        trace_data = {
            "traceEvents": self.events,
            "displayTimeUnit": "ms",
            "systemTraceEvents": "SystemTraceData",
            "otherData": {"version": "My Custom Tracer v1.0"},
        }

        with open(filename, "w") as f:
            json.dump(trace_data, f, indent=2)

        print(f"Timeline '{self.trace_name}' 已保存到 {filename}")


# 使用示例
if __name__ == "__main__":
    # 创建 timeline 生成器
    tracer = MojoTracingGenerator(trace_name="Test Example")

    # 设置进程名称:cite[7]
    tracer.set_process_name(0, "Core0")
    tracer.set_process_name(1, "Core1")

    # 设置线程名称:cite[7]
    tracer.set_thread_name(0, 1, "Cube")
    tracer.set_thread_name(0, 2, "Vector")
    tracer.set_thread_name(0, 3, "MTE2")
    tracer.set_thread_name(1, 1, "Cube")
    tracer.set_thread_name(1, 2, "Vector")
    tracer.set_thread_name(1, 3, "MTE2")

    # 获取当前时间作为基准
    start_time = time.time()

    tracer.add_event(
        "rms_norm",
        "vector",
        "X",
        start_time + 0.1,
        duration=0.15,
        process_id=0,
        thread_id=2,
        args={
            "op_name": "input_rms_norm",
            "layer_id": 0,
        },
    )

    tracer.add_event(
        "gemm",
        "cube",
        "X",
        start_time + 0.2,
        duration=0.15,
        process_id=0,
        thread_id=1,
        args={
            "op_name": "qkv_projection",
            "layer_id": 0,
        },
    )

    tracer.add_event("memory", "mte", "X", start_time + 0.3, duration=0.25, process_id=0, thread_id=3)

    tracer.add_event(
        "rms_norm",
        "vector",
        "X",
        start_time + 0.11,
        duration=0.15,
        process_id=1,
        thread_id=2,
        args={
            "op_name": "input_rms_norm",
            "layer_id": 0,
        },
    )

    tracer.add_event(
        "gemm",
        "cube",
        "X",
        start_time + 0.21,
        duration=0.15,
        process_id=1,
        thread_id=1,
        args={
            "op_name": "qkv_projection",
            "layer_id": 0,
        },
    )

    tracer.add_event("memory", "mte", "X", start_time + 0.31, duration=0.25, process_id=1, thread_id=3)

    # 保存到文件
    tracer.save_to_file("custom_process_thread_timeline.json")

    print("\n使用说明：")
    print("1. 在 Chrome 浏览器中打开 chrome://tracing")
    print("2. 点击 'Load' 按钮")
    print("3. 选择生成的 custom_process_thread_timeline.json 文件")
    print("4. 在时间线中可以看到自定义的进程和线程名称")
