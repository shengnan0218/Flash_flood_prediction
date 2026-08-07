"""Intentional guard for the not-yet-defined Zhejiang formal data contract."""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="浙江正式微调尚未启用；当前项目只执行湖南训练与测试"
    )
    parser.add_argument("--dataset-root")
    parser.add_argument("--checkpoint")
    parser.parse_args()
    parser.error(
        "已停止：浙江数据适配器、事件级微调划分和独立TEST尚未实现。"
        "为避免把synthetic或同一loader误作浙江训练/验证，本入口不会启动训练。"
    )


if __name__ == "__main__":
    main()
