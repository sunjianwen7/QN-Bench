"""
qnbench.utils.logging
======================

集中式日志配置。

设计原则:
    1. 即使不显式调用 setup_logging(), 日志也能正常输出 (通过 ensure_logging 自动初始化)。
    2. 仿真日志 (qnbench.env/engine) 与训练日志 (qnbench.ppo/train) 独立控制级别。
    3. 训练进度用 print() 保证可见性, 调试信息走 logger。
"""

import logging
import sys

# 模块级标记, 防止重复初始化
_initialized = False


def setup_logging(level: str = "INFO", sim_level: str = "WARNING"):
    """
    配置 qnbench 包的日志系统。

    Parameters
    ----------
    level : str
        训练 / 评估日志级别 ("DEBUG", "INFO", "WARNING").
        控制 qnbench.ppo, qnbench.train, qnbench.eval 等。
    sim_level : str
        仿真引擎日志级别。
        控制 qnbench.env, qnbench.engine。
        训练时通常设为 "WARNING" 避免刷屏。
    """
    global _initialized

    numeric = getattr(logging, level.upper(), logging.INFO)
    sim_numeric = getattr(logging, sim_level.upper(), logging.WARNING)

    # ── 根 logger: qnbench ────────────────────────────────────
    root = logging.getLogger("qnbench")
    root.setLevel(logging.DEBUG)  # 允许所有消息传递, 由 handler 过滤
    root.handlers.clear()         # 清除旧 handler, 防止重复
    root.propagate = False        # 不传递到 Python 根 logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(handler)

    # ── 仿真 logger: 独立控制级别 ─────────────────────────────
    # 训练时通常不想看到每条 link 的生成/过期日志
    for name in ("qnbench.env", "qnbench.engine"):
        lg = logging.getLogger(name)
        lg.setLevel(sim_numeric)

    _initialized = True


def ensure_logging():
    """
    确保日志系统已初始化。

    如果还没有调用过 setup_logging(), 自动以默认参数初始化。
    PPO trainer 和 runner 在启动时调用此函数, 保证即使用户
    没有显式 setup_logging(), 日志也能正常输出。
    """
    global _initialized
    if not _initialized:
        setup_logging()
