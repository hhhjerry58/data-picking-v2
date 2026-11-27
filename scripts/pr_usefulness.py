#!/usr/bin/env python3
"""八组固定预算 PR 继续训练与独立 T 评测入口。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from picking.pr_selection.continuation.runner import main

if __name__ == '__main__':
    main()
