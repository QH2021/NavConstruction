#!/usr/bin/env python3
"""
视觉语言导航系统 - 完整VLM导航任务执行
========================================

完整的2阶段导航实现：
- 阶段1: 路径规划 - 场景图读取 → A* 3条候选路径 → Agent1 VLM选择最优路径
- 阶段2: 迭代导航 - RGB帧捕获 → VLM调用 → 5步动作输出 → 执行循环 → 直到到达目标

功能：
1. 完整配置加载（6个YAML配置文件）
2. 场景图初始化和路径规划
3. Agent1进行路径选择和动作生成
4. Agent2进行重规划（可选）
5. RGB帧采集和视频生成
6. VLM对话、输入输出保存
7. 完整的导航指标记录

使用：
    python run.py --start Start01 --end R309 --use-habitat --enable-viz
"""

import sys
import os
import argparse
import logging
import json
from collections import deque
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


def _maybe_reexec_with_nvidia_egl_fix() -> None:
    """在部分conda环境中，habitat-sim 的 EGL/GLVND 可能默认走 Mesa vendor，导致 probe 报
    `unable to find CUDA device ... among ... EGL devices` 并回退到 Mock。

    这里在**尽量早**的位置（尚未导入 habitat_sim 相关模块前）检查并通过 re-exec 注入环境变量：
    - `__EGL_VENDOR_LIBRARY_FILENAMES`：强制使用系统 NVIDIA EGL vendor json
    - `__GLX_VENDOR_LIBRARY_NAME`：强制 GLX 走 nvidia
    - `LD_PRELOAD`：优先加载系统 `libGLdispatch.so.0`
    - `LD_LIBRARY_PATH`：将系统 lib 目录置前（尽量减少 conda GLVND 干扰）

    若条件不满足则不做任何事。
    """

    # 避免循环 re-exec
    if os.environ.get("CONSTRUCTIONNAV_EGL_FIX_REEXEC") == "1":
        return

    # 默认使用 Habitat；仅当显式 --no-habitat 时使用 Mock
    if "--no-habitat" in sys.argv:
        return

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    # 仅当 conda 环境里存在 mesa vendor 配置时才尝试修复
    conda_vendor_dir = Path(conda_prefix) / "share" / "glvnd" / "egl_vendor.d"
    conda_mesa_vendor = conda_vendor_dir / "50_mesa.json"
    if not conda_mesa_vendor.exists():
        return

    system_nvidia_vendor = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    system_gldispatch = Path("/lib/x86_64-linux-gnu/libGLdispatch.so.0")
    if not (system_nvidia_vendor.exists() and system_gldispatch.exists()):
        return

    # 若用户已显式设置 EGL vendor，则尊重用户设置
    if os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES"):
        return

    env = os.environ.copy()
    env["CONSTRUCTIONNAV_EGL_FIX_REEXEC"] = "1"
    env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(system_nvidia_vendor)
    env.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

    # LD_PRELOAD: 让系统 libGLdispatch 优先生效（避免 conda GLVND/mesa 组合导致 OpenGL version 获取失败）
    existing_preload = env.get("LD_PRELOAD", "").strip()
    preload_items = [str(system_gldispatch)]
    if existing_preload:
        preload_items.append(existing_preload)
    env["LD_PRELOAD"] = ":".join(preload_items)

    # LD_LIBRARY_PATH: 系统目录置前（不覆盖原值）
    existing_ld = env.get("LD_LIBRARY_PATH", "").strip()
    sys_ld_prefix = "/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"
    env["LD_LIBRARY_PATH"] = (
        f"{sys_ld_prefix}:{existing_ld}" if existing_ld else sys_ld_prefix
    )

    script_path = str(Path(__file__).resolve())
    argv = [sys.executable, script_path, *sys.argv[1:]]
    os.execve(sys.executable, argv, env)


# 确保可从任意工作目录运行：把项目根目录加入 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 必须在导入 habitat_sim/相关绑定前尽早执行
_maybe_reexec_with_nvidia_egl_fix()

# 项目导入
from src.config_loader import UnifiedConfigLoader
from src.scene_graph import SceneGraph
from src.agents import Agent1, Agent2
from src.habitat_integration import (
    EnvironmentFactory,
    HabitatVersionInfo,
    MockEnvironment,
)
from src.video_recorder import BackCameraRecorder
from src.visualizer import RealtimeVisualizer


def _require_cv2():
    if cv2 is None:
        raise ImportError(
            "缺少依赖 opencv-python（cv2）。请先执行: pip install opencv-python"
        )


def _configure_logging(
    log_dir: Path,
    *,
    level: str = "INFO",
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    return log_file


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="VLM导航系统 - 完整视觉语言导航任务执行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例使用：
  基础运行: python run.py --start Start01 --end R309
  启用可视化: python run.py --start Start01 --end R309 --enable-viz
    使用虚拟环境: python run.py --start Start01 --end R309 --no-habitat
  自定义配置目录: python run.py --start Start01 --end R309 --config-dir ./my_configs
        """,
    )

    parser.add_argument(
        "--start",
        type=str,
        default="Start01",
        help="起点房间ID (默认: Start01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="R309",
        help="终点房间ID (默认: R309)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="./configs",
        help="配置目录路径 (默认: ./configs)",
    )
    habitat_group = parser.add_mutually_exclusive_group()
    habitat_group.add_argument(
        "--use-habitat",
        action="store_true",
        help="使用Habitat环境 (默认: 启用Habitat)",
    )
    habitat_group.add_argument(
        "--no-habitat",
        action="store_true",
        help="强制使用虚拟环境 (Mock)",
    )
    viz_group = parser.add_mutually_exclusive_group()
    viz_group.add_argument(
        "--enable-viz",
        action="store_true",
        help="启用可视化界面",
    )
    viz_group.add_argument(
        "--disable-viz",
        action="store_true",
        help="禁用可视化界面（覆盖配置）",
    )

    agent2_group = parser.add_mutually_exclusive_group()
    agent2_group.add_argument(
        "--enable-agent2",
        action="store_true",
        help="启用Agent2重规划（覆盖配置）",
    )
    agent2_group.add_argument(
        "--disable-agent2",
        action="store_true",
        help="禁用Agent2重规划（覆盖配置）",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20, # None
        help="最大导航步数 (默认: 从配置读取)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: 从配置读取)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录 (默认: 从配置读取)",
    )

    # 亮度（渲染后处理）覆盖：run.py 只负责覆盖配置，不直接改 habitat_integration 实现
    brightness_group = parser.add_mutually_exclusive_group()
    brightness_group.add_argument(
        "--brightness-factor",
        type=float,
        default=None,
        help="覆盖 environment.rendering.brightness_factor（1.0=不变；>1 更亮）",
    )
    brightness_group.add_argument(
        "--disable-brightness",
        action="store_true",
        help="禁用亮度后处理（等价于 brightness_factor=1.0）",
    )

    # VLM 覆盖项（便于快速切换服务/模型）
    parser.add_argument(
        "--vlm-endpoint",
        type=str,
        default=None,
        help="覆盖 VLM API endpoint（默认: 从配置读取）",
    )
    parser.add_argument(
        "--vlm-model",
        type=str,
        default=None,
        help="覆盖 VLM model name/path（默认: 从配置读取）",
    )
    parser.add_argument(
        "--vlm-trace-path",
        type=str,
        default=None,
        help="覆盖 VLM 单文件 trace 路径（默认: 输出目录下 vlm_io.json）",
    )

    args = parser.parse_args()
    # 默认使用 Habitat；仅当显式 --no-habitat 时使用 Mock
    if not getattr(args, "no_habitat", False):
        args.use_habitat = True
    return args


class VLMNavigationRunner:
    """
    VLM导航系统的完整执行器

    负责：
    1. 初始化所有模块和配置
    2. 管理2阶段导航流程
    3. 保存所有输出（VLM对话、RGB帧、视频、指标）
    4. 处理错误和异常
    """

    def __init__(self, args):
        """
        初始化导航执行器

        Args:
            args: 命令行参数
        """
        self.args = args

        # 先使用简易日志（后续会在输出目录创建后 force 重新配置）
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

        # 加载配置
        self.config_loader = UnifiedConfigLoader(args.config_dir)
        self._load_configs()

        # 解析运行期参数（CLI 优先，其次配置，最后默认值）
        self._resolve_runtime_settings()

        # 创建输出目录并配置日志到 run_dir
        self._setup_output_directories()
        log_file = _configure_logging(
            self.dirs["logs"],
            level=self.log_level,
            fmt=self.system_config.get("logging", {}).get(
                "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ),
            datefmt=self.system_config.get("logging", {}).get(
                "date_format", "%Y-%m-%d %H:%M:%S"
            ),
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 VLM导航系统启动")
        self.logger.info(f"   配置目录: {args.config_dir}")
        self.logger.info(f"   输出目录: {self.run_dir}")
        self.logger.info(f"   日志文件: {log_file}")

        # NOTE: 本文件实现了独立的两阶段导航流程；
        # NavigationSystem 会单独创建输出目录并产生副作用，这里不再在 __init__ 中初始化。

    def _load_configs(self):
        """加载所有配置文件

        优先加载统一的unified_config.yaml，如果不存在，则使用独立的配置文件
        """
        try:
            unified_config_path = Path(self.args.config_dir) / "unified_config.yaml"
            self.unified_config = (
                self.config_loader.load_unified_config()
                if unified_config_path.exists()
                else None
            )

            # 始终通过 loader 的 legacy 派生接口加载：保证字段结构一致
            self.env_config = self.config_loader.load_environment_config()
            self.agent_config = self.config_loader.load_agent_config()
            self.vlm_config = self.config_loader.load_vlm_config()
            self.system_config = self.config_loader.load_system_config()
            self.paths_config = self.config_loader.load_paths_config()
            self.prompts_config = self.config_loader.load_prompts_config()
            try:
                self.navigation_config = self.config_loader.load_config(
                    "navigation_config"
                )
            except Exception:
                self.navigation_config = {}

            if self.unified_config is not None:
                self.logger.info("✅ unified_config.yaml 已加载并用于派生配置")
            else:
                self.logger.info("✅ 使用独立配置文件（或 defaults）加载成功")

            self.logger.debug(f"   环境配置: {self.env_config.get('scene', {})}")
            self.logger.debug(f"   Agent配置: {self.agent_config.get('agents', {})}")
            self.logger.debug(f"   VLM配置: {self.vlm_config.get('model', {})}")

        except Exception as e:
            self.logger.error(f"❌ 配置加载失败: {e}")
            raise

    def _resolve_runtime_settings(self):
        """解析运行期参数：CLI 覆盖配置，配置覆盖默认值。"""

        sys_out = self.system_config.get("output", {})
        base_dir = self.args.output_dir or sys_out.get("base_dir") or "./output"
        self.output_base_dir = Path(base_dir)
        self.use_timestamp_dir = bool(sys_out.get("use_timestamp_dir", True))

        # 日志级别
        cfg_level = self.system_config.get("logging", {}).get("level", "INFO")
        self.log_level = (self.args.log_level or cfg_level or "INFO").upper()

        # max_steps：优先 CLI，其次 unified_config.navigation.navigation_loop，其次默认
        cfg_max_steps = None
        try:
            if isinstance(getattr(self, "unified_config", None), dict):
                cfg_max_steps = (
                    self.unified_config.get("navigation", {})
                    .get("navigation_loop", {})
                    .get("max_steps")
                )
        except Exception:
            cfg_max_steps = None
        self.max_steps = int(self.args.max_steps or cfg_max_steps or 500)

        # 亮度后处理：CLI 覆盖 environment.rendering.brightness_factor
        try:
            if not isinstance(self.env_config, dict):
                self.env_config = {}
            rendering_cfg = self.env_config.get("rendering")
            if not isinstance(rendering_cfg, dict):
                rendering_cfg = {}
                self.env_config["rendering"] = rendering_cfg

            if getattr(self.args, "disable_brightness", False):
                rendering_cfg["enabled"] = False
                rendering_cfg["brightness_factor"] = 1.0
            elif getattr(self.args, "brightness_factor", None) is not None:
                rendering_cfg["enabled"] = True
                rendering_cfg["brightness_factor"] = float(self.args.brightness_factor)
        except Exception:
            pass

        # 可视化：CLI 覆盖配置
        cfg_viz = bool(
            self.system_config.get("visualization", {}).get("enabled", False)
        )
        if getattr(self.args, "disable_viz", False):
            self.enable_viz = False
        elif getattr(self.args, "enable_viz", False):
            self.enable_viz = True
        else:
            self.enable_viz = cfg_viz

        # Agent2：CLI 覆盖配置
        cfg_agent2 = bool(self.agent_config.get("agent2", {}).get("enabled", False))
        if getattr(self.args, "disable_agent2", False):
            self.enable_agent2 = False
        elif getattr(self.args, "enable_agent2", False):
            self.enable_agent2 = True
        else:
            self.enable_agent2 = cfg_agent2

    def run(self) -> bool:
        """
        完整导航流程执行

        返回：
            是否成功完成导航
        """
        try:
            # 打印Habitat版本信息
            try:
                version_info = HabitatVersionInfo()
                self.logger.info(version_info.get_framework_summary())
            except Exception as e:
                self.logger.warning(f"⚠️ 无法检测Habitat版本: {e}")

            # 输出目录已在 __init__ 中创建并用于配置日志；这里避免重复生成新的 timestamp 目录，
            # 以确保日志/视频/VLM trace 全部落在同一个 run_dir。

            # 阶段1: 路径规划
            self.logger.info("\n" + "=" * 70)
            self.logger.info("📍 阶段1: 路径规划")
            self.logger.info("=" * 70)

            if not self._phase1_path_planning():
                self.logger.error("❌ 路径规划失败")
                return False

            # 阶段2: 迭代导航
            self.logger.info("\n" + "=" * 70)
            self.logger.info("🚀 阶段2: 迭代导航")
            self.logger.info("=" * 70)

            if not self._phase2_navigation():
                self.logger.error("❌ 导航执行失败")
                return False

            self.logger.info("\n" + "=" * 70)
            self.logger.info("✅ 导航任务完成!")
            self.logger.info("=" * 70)

            return True

        except KeyboardInterrupt:
            self.logger.warning("⚠️ 用户中断导航")
            return False
        except Exception as e:
            self.logger.error(f"❌ 导航异常: {e}")
            import traceback

            traceback.print_exc()
            return False
        finally:
            self.logger.info("🔌 导航系统关闭")

    def _setup_output_directories(self):
        """设置输出目录结构"""
        base_dir = self.output_base_dir
        if self.use_timestamp_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = base_dir / f"run_{timestamp}"
        else:
            self.run_dir = base_dir

        # 创建所有必要的子目录（优先使用配置中的 subdirs，避免重复硬编码）
        subdirs = (
            self.system_config.get("output", {}).get("subdirs", {})
            if isinstance(self.system_config, dict)
            else {}
        )
        frames_dir = self.run_dir / (subdirs.get("frames") or "frames")
        videos_dir = self.run_dir / (subdirs.get("videos") or "videos")
        logs_dir = self.run_dir / (subdirs.get("logs") or "logs")
        paths_dir = self.run_dir / (subdirs.get("paths") or "paths")
        metrics_dir = self.run_dir / (subdirs.get("metrics") or "metrics")

        self.dirs = {
            "frames": frames_dir,
            "videos": videos_dir,
            "logs": logs_dir,
            "metrics": metrics_dir,
            "paths": paths_dir,
        }

        for dir_name, dir_path in self.dirs.items():
            dir_path.mkdir(parents=True, exist_ok=True)

        # 单文件 VLM I/O 记录（不再创建 vlm_inputs/vlm_outputs 目录）
        self.vlm_io_path = self.run_dir / "vlm_io.json"

        self.logger.info(f"✅ 输出目录创建完成: {self.run_dir}")

    def _phase1_path_planning(self) -> bool:
        """
        阶段1: 路径规划

        流程：
        1. 初始化场景图
        2. A*算法生成3条候选路径
        3. Agent1基于平面图选择最优路径
        4. 保存路径信息和候选路径

        返回：
            是否成功
        """
        try:
            # 1. 初始化场景图
            self.logger.info("初始化场景图...")

            # 获取Excel文件路径（兼容 legacy/统一配置派生结构）
            data_paths = (
                self.paths_config.get("data", {})
                if isinstance(self.paths_config, dict)
                else {}
            )
            excel_dir = self.paths_config.get("excel_dir") or "./data"
            door_excel = (
                self.paths_config.get("door_table_path")
                or data_paths.get("door_table")
                or f"{excel_dir}/door_table.xlsx"
            )
            comp_excel = (
                self.paths_config.get("component_table_path")
                or data_paths.get("component_table")
                or f"{excel_dir}/component_table.xlsx"
            )

            # 检查Excel文件是否存在
            if not Path(door_excel).exists() or not Path(comp_excel).exists():
                self.logger.warning("⚠️ Excel文件不存在，使用虚拟路径")
                self.logger.info(f"   door_excel: {door_excel}")
                self.logger.info(f"   comp_excel: {comp_excel}")
                # 创建虚拟路径（3条候选路径）
                self.selected_path = [self.args.start, self.args.end]
                self.scene_graph = None
                self.logger.info("✅ 使用虚拟路径进行导航")
                return True

            try:
                scene_graph = SceneGraph(
                    door_excel=door_excel,
                    comp_excel=comp_excel,
                )
            except ImportError as e:
                # pandas 读取 xlsx 需要 openpyxl；缺失时降级到虚拟路径
                self.logger.warning(f"⚠️  Excel 依赖缺失，降级到虚拟路径: {e}")
                self.selected_path = [self.args.start, self.args.end]
                self.scene_graph = None
                return True
            self.logger.info("✅ 场景图初始化成功")

            # 2. 生成候选路径
            self.logger.info(f"生成候选路径: {self.args.start} → {self.args.end}")
            paths = scene_graph.find_k_shortest_paths(
                self.args.start, self.args.end, k=3
            )

            if not paths:
                self.logger.error("❌ 无法生成候选路径")
                return False

            self.logger.info(f"✅ 生成 {len(paths)} 条候选路径")
            for i, (rooms, doors, steps) in enumerate(paths, 1):
                self.logger.info(f"   路径{i}: {' → '.join(rooms)} ({steps}步)")

            # 保存候选路径
            candidates = [
                {
                    "id": i + 1,
                    "rooms": rooms,
                    "doors": doors,
                    "steps": steps,
                }
                for i, (rooms, doors, steps) in enumerate(paths)
            ]
            candidates_file = self.dirs["paths"] / "candidates.json"
            with open(candidates_file, "w", encoding="utf-8") as f:
                json.dump(candidates, f, ensure_ascii=False, indent=2)
            self.logger.info("✅ 候选路径已保存: candidates.json")

            # 3. Agent1选择最优路径
            self.logger.info("Agent1正在选择最优路径...")

            # 保存scene_graph供后续使用
            self.scene_graph = scene_graph

            # 创建虚拟平面图（实际应从文件加载）
            floor_maps = self._load_floor_maps(paths)

            # 初始化Agent1
            agent1 = Agent1(
                config_dir=self.args.config_dir,
                output_dir=str(self.run_dir),
            )

            # Agent1选择路径
            try:
                selected_idx = agent1.select_best_path(
                    paths, floor_maps, self.args.start, self.args.end
                )
                selected_path_rooms = paths[selected_idx][0]
                self.logger.info(
                    f"✅ Agent1选择路径{selected_idx + 1}: {' → '.join(selected_path_rooms)}"
                )
            except Exception as e:
                self.logger.warning(f"⚠️ Agent1选择异常: {e}，使用默认路径")
                selected_idx = 0
                selected_path_rooms = paths[0][0]

            # 保存选中路径
            selected_path_info = {
                "start": self.args.start,
                "end": self.args.end,
                "path": selected_path_rooms,
                "selected_index": selected_idx + 1,
                "total_steps": len(selected_path_rooms) - 1,
                "timestamp": datetime.now().isoformat(),
            }
            selected_path_file = self.dirs["paths"] / "selected_path.json"
            with open(selected_path_file, "w", encoding="utf-8") as f:
                json.dump(selected_path_info, f, ensure_ascii=False, indent=2)
            self.logger.info("✅ 选中路径已保存: selected_path.json")

            # 保存场景图信息用于后续阶段
            self.selected_path = selected_path_rooms
            self.scene_graph = scene_graph

            return True

        except Exception as e:
            self.logger.error(f"❌ 路径规划异常: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _phase2_navigation(self) -> bool:
        """
        阶段2: 迭代导航

        流程：
        1. 初始化环境（Habitat或虚拟）
        2. 重置环境到起点
        3. 循环直到到达目标：
           a. 捕获RGB帧（前置摄像头）
           b. 调用VLM生成5步动作序列
           c. 执行每步动作
           d. 检查是否到达目标
           e. 如遇障碍，触发Agent2重规划
        4. 保存RGB帧和视频
        5. 保存完整的VLM对话记录

        返回：
            是否成功
        """
        try:
            # 1. 初始化环境
            self.logger.info("初始化模拟环境...")
            env = self._create_environment()
            if env is None:
                self.logger.error("❌ 环境初始化失败")
                return False

            self.logger.info("✅ 模拟环境初始化成功")

            # 2. 初始化Agent和录像机
            if getattr(self.args, "vlm_trace_path", None):
                trace_path = Path(self.args.vlm_trace_path).expanduser()
                if not trace_path.is_absolute():
                    trace_path = self.run_dir / trace_path
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                vlm_trace_path = str(trace_path)
            else:
                vlm_trace_path = str(
                    getattr(self, "vlm_io_path", self.run_dir / "vlm_io.json")
                )
            agent1 = Agent1(
                config_dir=self.args.config_dir,
                output_dir=str(self.run_dir),
                vlm_trace_path=vlm_trace_path,
            )

            # VLM endpoint/model 覆盖
            if getattr(self.args, "vlm_endpoint", None):
                agent1.vlm_url = str(self.args.vlm_endpoint)
            if getattr(self.args, "vlm_model", None):
                agent1.model_name = str(self.args.vlm_model)

            agent2 = None
            if self.enable_agent2:
                agent2 = Agent2(
                    config_dir=self.args.config_dir,
                    output_dir=str(self.run_dir),
                    vlm_trace_path=vlm_trace_path,
                )
                if getattr(self.args, "vlm_endpoint", None):
                    agent2.vlm_url = str(self.args.vlm_endpoint)
                if getattr(self.args, "vlm_model", None):
                    agent2.model_name = str(self.args.vlm_model)
                self.logger.info("✅ Agent2重规划已启用")

            # 3. 初始化可视化
            visualizer = RealtimeVisualizer(enable=self.enable_viz)

            # 4. 导航循环
            step_count = 0
            # 当前位置信息必须由 VLM 判定；这里仅保留一个“提示/回退值”
            current_room_hint = self.args.start or (
                self.selected_path[0] if self.selected_path else "unknown"
            )
            current_floor_hint = None

            # 路径进度提示：当 VLM 的 room 不在规划路径中时，不要把进度重置为 0。
            # 保留上一轮最可信的 path_idx，避免提示在路径起点来回跳导致策略振荡。
            path_idx = 0

            # 楼层锚点：用于楼梯间阶段稳定 floorplan 选择（避免模型在楼梯内提前跳楼层）
            floor_anchor = None
            try:
                if getattr(self, "scene_graph", None) is not None and isinstance(
                    current_room_hint, str
                ):
                    floor_anchor = int(
                        self.scene_graph.extract_floor(current_room_hint)
                    )
            except Exception:
                floor_anchor = None
            in_stairwell = False
            target_room = (
                self.selected_path[-1] if self.selected_path else self.args.end
            )

            with self._create_video_recorder() as video_recorder:
                # 重置环境
                obs = env.reset()
                self.logger.info("✅ 环境已重置")

                enable_front_depth = False
                try:
                    sensors_cfg = (
                        self.env_config.get("sensors", {})
                        if isinstance(self.env_config, dict)
                        else {}
                    )
                    depth_cfg = (
                        sensors_cfg.get("front_depth", {})
                        if isinstance(sensors_cfg, dict)
                        else {}
                    )
                    enable_front_depth = bool(depth_cfg.get("enabled", False))
                except Exception:
                    enable_front_depth = False

                # 动作计数（用于可视化信息）
                action_counts = {
                    "move_forward": 0,
                    "move_backward": 0,
                    "turn_left": 0,
                    "turn_right": 0,
                }

                # 最近执行动作（用于 VLM 避免震荡/回环）
                last_actions: deque = deque(maxlen=20)

                # 碰撞/卡住跟踪：用于下一次 VLM 提示（增强稳定性，避免重复撞墙）
                collision_state = {
                    "had_collision": False,
                    "action": None,
                    "dist_moved": None,
                    "consecutive": 0,
                }

                while step_count < self.max_steps:
                    self.logger.info(f"\n{'=' * 60}")
                    self.logger.info(f"步骤 {step_count + 1}/{self.max_steps}")
                    self.logger.info(f"{'=' * 60}")

                    # 4a. 捕获观测
                    rgb_front = obs.get(
                        "rgb_front", np.zeros((720, 1280, 3), dtype=np.uint8)
                    )
                    depth_front = obs.get(
                        "depth_front", np.zeros((720, 1280), dtype=np.float32)
                    )
                    rgb_back = obs.get(
                        "rgb_back", np.zeros((480, 640, 3), dtype=np.uint8)
                    )
                    agent_pos = obs.get("agent_pos", np.array([0, 0, 0]))

                    # 重要：不要在“决策开始”重复append，否则下一轮会出现 t-1==t。
                    # 本系统使用“最后一次动作后的观测”(t) 进行 VLM 决策：不再构造两帧(t-1,t)输入。

                    # 4a1. 保存后置摄像头帧到视频（全程）
                    # 后置摄像头从第一步开始就录制全程
                    metrics = {
                        "step": step_count,
                        "room": current_room_hint,
                        "position": f"({agent_pos[0]:.2f}, {agent_pos[1]:.2f})",
                        "status": "Recording",
                    }
                    video_recorder.write_frame(
                        rgb_back,
                        floorplan=None,
                        metrics=metrics,
                    )

                    # 当前房间/楼层由 VLM 判定；这里仅打印“提示/回退值”
                    self.logger.info(f"📍 当前房间(提示): {current_room_hint}")
                    if current_floor_hint is not None:
                        self.logger.info(f"🏢 当前楼层(提示): {current_floor_hint}")
                    self.logger.info(f"🎯 目标房间: {target_room}")
                    if self.selected_path:
                        try:
                            path_idx = self.selected_path.index(current_room_hint)
                        except ValueError:
                            # 当前房间提示不在路径里：保留上一轮 path_idx
                            pass
                        self.logger.info(
                            f"📊 路径进度(基于VLM房间估计): {path_idx + 1}/{len(self.selected_path)}"
                        )
                    else:
                        path_idx = 0

                    # 4b. 获取平面图（用于VLM）
                    # 楼梯间阶段：优先使用 floor_anchor（最近一次高置信“非楼梯间”楼层）
                    floorplan_floor_hint = current_floor_hint
                    try:
                        if in_stairwell and floor_anchor is not None:
                            floorplan_floor_hint = floor_anchor
                        elif floorplan_floor_hint is None and floor_anchor is not None:
                            floorplan_floor_hint = floor_anchor
                    except Exception:
                        pass

                    floorplan_vlm = self._get_floorplan_vlm(
                        current_room_hint=current_room_hint,
                        current_floor_hint=floorplan_floor_hint,
                    )

                    # 4c. Agent1生成5步动作（仅使用前置摄像头进行VLM决策）
                    self.logger.info("🤖 Agent1正在分析场景...")

                    # 【关键】仅此时保存前置摄像头帧（VLM决策时）
                    rgb_front_path = self._save_rgb_frame(
                        rgb_front, "front_rgb", step_count
                    )

                    depth_front_path = None
                    if enable_front_depth:
                        try:
                            depth_vis = agent1._depth_to_vis_rgb(depth_front)
                            depth_front_path = self._save_rgb_frame(
                                depth_vis, "front_depth", step_count
                            )
                        except Exception:
                            depth_front_path = None

                    depth_last = None
                    if enable_front_depth:
                        depth_last = [depth_front]

                    result = agent1.generate_action_batch(
                        rgb_image=rgb_front,
                        depth_images=depth_last,
                        floorplan=floorplan_vlm,
                        current_room=current_room_hint,
                        target_room=target_room,
                        path_rooms=self.selected_path,
                        context={
                            "step": step_count,
                            "path_index_hint": path_idx,
                            "next_room_hint": (
                                self.selected_path[path_idx + 1]
                                if self.selected_path
                                and isinstance(path_idx, int)
                                and path_idx + 1 < len(self.selected_path)
                                else None
                            ),
                            "remaining_path_rooms": (
                                self.selected_path[path_idx:]
                                if self.selected_path
                                and isinstance(path_idx, int)
                                and path_idx < len(self.selected_path)
                                else None
                            ),
                            "target_floor_hint": (
                                int(self.scene_graph.extract_floor(target_room))
                                if getattr(self, "scene_graph", None) is not None
                                else None
                            ),
                            "front_rgb_path": str(rgb_front_path)
                            if rgb_front_path
                            else None,
                            "front_depth_path": str(depth_front_path)
                            if depth_front_path
                            else None,
                            "current_room_hint": current_room_hint,
                            "current_floor_hint": current_floor_hint,
                            "stairs": {
                                "robot_can_use_stairs": True,
                                "rule": "floor += 1 only after completing a full stairwell (two flights)",
                                "in_stairwell": bool(in_stairwell),
                                "floor_anchor": floor_anchor,
                                "floorplan_floor_used": floorplan_floor_hint,
                                "next_room_hint": (
                                    self.selected_path[path_idx + 1]
                                    if self.selected_path
                                    and isinstance(path_idx, int)
                                    and path_idx + 1 < len(self.selected_path)
                                    else None
                                ),
                                "remaining_path_rooms": (
                                    self.selected_path[path_idx:]
                                    if self.selected_path
                                    and isinstance(path_idx, int)
                                    and path_idx < len(self.selected_path)
                                    else None
                                ),
                            },
                            "collision": collision_state.copy(),
                            "last_actions": list(last_actions),
                            "action_counts": dict(action_counts),
                        },
                    )

                    if not result:
                        self.logger.error("❌ Agent1无法生成动作")
                        return False

                    actions = result.get("actions", [])
                    reached_goal = result.get("reached_goal", False)
                    reasoning = result.get("reasoning", "")

                    # 由 VLM 回传当前位置信息；如果缺失则保留上一轮 hint
                    vlm_current_room = result.get("current_room")
                    vlm_current_floor = result.get("current_floor")
                    vlm_confidence = result.get("confidence")
                    if isinstance(vlm_current_room, str) and vlm_current_room.strip():
                        current_room_hint = vlm_current_room.strip()
                    if vlm_current_floor is not None:
                        current_floor_hint = vlm_current_floor

                    # 更新楼梯阶段与楼层锚点：
                    # - 若处于楼梯间(Sxx)则进入 stair 模式，并保持 floor_anchor 作为 floorplan 来源。
                    # - 仅当模型高置信地定位到“非楼梯间”(H/R/Start)且能解析出楼层时，才更新 floor_anchor 并退出楼梯模式。
                    try:
                        room_str = (
                            current_room_hint
                            if isinstance(current_room_hint, str)
                            else ""
                        )
                        conf = None
                        try:
                            if vlm_confidence is not None:
                                conf = float(vlm_confidence)
                        except Exception:
                            conf = None

                        if room_str.startswith("S"):
                            in_stairwell = True

                        def _extract_floor_from_label(label: str):
                            if getattr(self, "scene_graph", None) is not None:
                                try:
                                    return int(self.scene_graph.extract_floor(label))
                                except Exception:
                                    return None
                            # fallback：取第一个数字作为楼层
                            import re

                            m = re.search(r"(\d)", label)
                            if m:
                                try:
                                    return int(m.group(1))
                                except Exception:
                                    return None
                            return None

                        kind = "unknown"
                        if room_str.startswith("H"):
                            kind = "hallway"
                        elif room_str.startswith("R"):
                            kind = "room"
                        elif room_str.startswith("Start"):
                            kind = "outdoor"
                        elif room_str.startswith("S"):
                            kind = "stair"

                        if kind in ("hallway", "room", "outdoor") and (
                            conf is None or conf >= 0.7
                        ):
                            f = _extract_floor_from_label(room_str)
                            if f is not None:
                                floor_anchor = f
                                in_stairwell = False
                    except Exception:
                        pass

                    self.logger.info("💡 VLM决策:")
                    self.logger.info(f"   动作: {actions}")
                    self.logger.info(f"   到达目标/结束信号: {reached_goal}")
                    if vlm_current_room is not None or vlm_current_floor is not None:
                        self.logger.info(
                            f"   当前位置(VLM): room={vlm_current_room} floor={vlm_current_floor} conf={vlm_confidence}"
                        )
                    self.logger.info(f"   推理: {reasoning}")

                    # 【可视化】显示前后摄像头和VLM信息
                    vlm_viz_info = {
                        "actions": actions,
                        "reasoning": reasoning,
                        "current_room": current_room_hint,
                        "target_room": target_room,
                        "step": step_count,
                        "agent_pos": (
                            float(agent_pos[0]),
                            float(agent_pos[1]),
                            float(agent_pos[2]),
                        )
                        if isinstance(agent_pos, np.ndarray) and agent_pos.size >= 3
                        else None,
                        "navmesh_loaded": bool(int(obs.get("navmesh_loaded")[0]))
                        if isinstance(obs.get("navmesh_loaded"), np.ndarray)
                        else False,
                        "navigable_area": float(obs.get("navigable_area")[0])
                        if isinstance(obs.get("navigable_area"), np.ndarray)
                        else None,
                        "robot_loaded": bool(int(obs.get("robot_loaded")[0]))
                        if isinstance(obs.get("robot_loaded"), np.ndarray)
                        else False,
                        "action_counts": action_counts,
                        "show_navmesh": bool(int(obs.get("navmesh_visualization")[0]))
                        if isinstance(obs.get("navmesh_visualization"), np.ndarray)
                        else False,
                    }
                    if not visualizer.display(rgb_front, rgb_back, vlm_viz_info):
                        self.logger.warning("⚠️ 用户关闭可视化窗口，停止导航")
                        self._save_metrics(False, step_count, path_idx)
                        visualizer.close()
                        return False

                    # 4d. 执行动作序列
                    for action in actions:
                        if step_count >= self.max_steps:
                            break

                        # 检查是否已到达目标
                        if reached_goal and action == "stop":
                            self.logger.info("\n" + "=" * 70)
                            self.logger.info(f"✅ 已到达目标房间 {target_room}!")
                            self.logger.info(f"📊 总步数: {step_count}")
                            self.logger.info("🏁 导航任务成功完成!")
                            self.logger.info("=" * 70)

                            # 后置摄像头视频由BackCameraRecorder自动管理
                            self._save_metrics(True, step_count, path_idx)

                            return True

                        # 执行动作
                        self.logger.info(f"🎮 执行动作: {action}")
                        prev_pos = agent_pos.copy()
                        obs, info = env.step(action)
                        step_count += 1

                        if action in action_counts:
                            action_counts[action] += 1

                        try:
                            last_actions.append(action)
                        except Exception:
                            pass

                        # 更新状态
                        agent_pos = obs.get("agent_pos", np.array([0, 0, 0]))
                        rgb_front = obs.get(
                            "rgb_front", np.zeros((720, 1280, 3), dtype=np.uint8)
                        )
                        depth_front = obs.get(
                            "depth_front", np.zeros((720, 1280), dtype=np.float32)
                        )

                        # 重要：VLM 决策使用“当前观测”(t)，不再维护两帧历史。
                        rgb_back = obs.get(
                            "rgb_back", np.zeros((480, 640, 3), dtype=np.uint8)
                        )

                        # 【重要】每步都保存后置摄像头帧到视频
                        metrics_current = {
                            "step": step_count,
                            "room": current_room_hint,
                            "action": action,
                            "position": f"({agent_pos[0]:.2f}, {agent_pos[1]:.2f})",
                            "status": "Executing",
                        }
                        video_recorder.write_frame(
                            rgb_back,
                            floorplan=None,
                            metrics=metrics_current,
                        )

                        # 4e. 检查障碍（简单的移动距离检查）
                        if action in ("move_forward", "move_backward"):
                            dist_moved = np.linalg.norm(agent_pos - prev_pos)
                            collision_thr = self.navigation_config.get(
                                "navigation_loop", {}
                            ).get("collision_distance_threshold", 0.05)
                            if dist_moved < float(collision_thr):  # 移动过少，认为卡住
                                self.logger.warning(
                                    f"⚠️ 检测到碰撞 (移动距离: {dist_moved:.3f})"
                                )

                                collision_state["had_collision"] = True
                                collision_state["action"] = action
                                collision_state["dist_moved"] = float(dist_moved)
                                collision_state["consecutive"] = (
                                    int(collision_state.get("consecutive", 0)) + 1
                                )

                                # 触发Agent2重规划
                                if agent2 and self.scene_graph is not None:
                                    self.logger.info("🔄 呼叫Agent2进行重规划...")
                                    blocked_edge = None
                                    if self.selected_path:
                                        try:
                                            idx = self.selected_path.index(
                                                current_room_hint
                                            )
                                        except ValueError:
                                            idx = None
                                        if idx is not None and idx + 1 < len(
                                            self.selected_path
                                        ):
                                            next_room = self.selected_path[idx + 1]
                                            blocked_edge = (
                                                current_room_hint,
                                                next_room,
                                            )

                                    if blocked_edge is not None:
                                        new_path = agent2.replan_path(
                                            current_room=current_room_hint,
                                            target_room=target_room,
                                            blocked_edge=blocked_edge,
                                            scene_graph=self.scene_graph,
                                        )

                                        if new_path:
                                            self.selected_path = new_path
                                            self.logger.info(
                                                f"✅ 新路径: {' → '.join(self.selected_path)}"
                                            )
                                            break  # 重新开始VLM决策
                                        else:
                                            self.logger.error("❌ Agent2重规划失败")
                                    else:
                                        self.logger.warning(
                                            "⚠️ 无法从路径推断被阻塞边（当前房间不在路径或已到末端），跳过Agent2重规划"
                                        )
                                elif agent2 and self.scene_graph is None:
                                    self.logger.warning(
                                        "⚠️ 场景图不可用（缺少Excel元数据），跳过Agent2重规划"
                                    )
                            else:
                                # 移动正常：清空碰撞状态
                                collision_state["had_collision"] = False
                                collision_state["action"] = None
                                collision_state["dist_moved"] = None
                                collision_state["consecutive"] = 0

                        # 记录“刚走过的经历/空间变化”（可落盘，供后续决策参考）
                        try:
                            if agent1 is not None and hasattr(
                                agent1, "record_experience"
                            ):
                                agent1.record_experience(
                                    kind="executed_action",
                                    data={
                                        "step": step_count,
                                        "action": action,
                                        "dist_moved": float(
                                            np.linalg.norm(agent_pos - prev_pos)
                                        )
                                        if isinstance(agent_pos, np.ndarray)
                                        and isinstance(prev_pos, np.ndarray)
                                        else None,
                                        "had_collision": bool(
                                            collision_state.get("had_collision")
                                        ),
                                        "consecutive": int(
                                            collision_state.get("consecutive", 0) or 0
                                        ),
                                        "room_hint": current_room_hint,
                                        "floor_hint": current_floor_hint,
                                        "in_stairwell": bool(in_stairwell),
                                        "floor_anchor": floor_anchor,
                                    },
                                    importance=0.2,
                                    source="env",
                                )
                        except Exception:
                            pass

                    if step_count >= self.max_steps:
                        self.logger.warning(f"⚠️ 达到最大步数 {self.max_steps}")
                        self._save_metrics(False, step_count, path_idx)
                        return False

            return False

        except Exception as e:
            self.logger.error(f"❌ 导航执行异常: {e}")
            import traceback

            traceback.print_exc()
            return False
        finally:
            # 清理资源
            try:
                visualizer.close()
            except Exception:
                pass
            try:
                env.close()
            except Exception:
                pass

    def _create_environment(self):
        """创建模拟环境"""
        if self.args.no_habitat:
            env = MockEnvironment(self.logger)
            self.logger.info("✅ 虚拟环境已初始化（--no-habitat）")
            return env

        # 默认使用 Habitat；仅当明确 --no-habitat 时使用 Mock
        if getattr(self.args, "use_habitat", True):
            try:
                factory = EnvironmentFactory(self.logger)
                scene_path = self.env_config.get("scene", {}).get(
                    "path",
                    "./data/scene_datasets/habitat-test-scenes/3dExport1212f.glb",
                )

                config = {
                    "scene_path": scene_path,
                    "use_habitat_lab": False,
                    "agent_config": self.env_config.get("agent", {}),
                    "sim_config": {
                        "enable_physics": self.env_config.get("physics", {}).get(
                            "enabled", True
                        ),
                        "physics_config_file": self.env_config.get("physics", {}).get(
                            "config_file", "data/default.physics_config.json"
                        ),
                    },
                    "env_config": self.env_config,
                    # 避免重复窗口：run.py 使用 RealtimeVisualizer 统一显示
                    "enable_visualization": False,
                }

                env, _ = factory.create_environment(config)
                self.logger.info("✅ Habitat环境已初始化")
                return env

            except Exception as e:
                self.logger.warning(f"⚠️ Habitat初始化失败: {e}")
                self.logger.info("降级到虚拟环境...")

        # 使用虚拟环境
        env = MockEnvironment(self.logger)
        self.logger.info("✅ 虚拟环境已初始化")
        return env

    def _get_floorplan_vlm(
        self,
        *,
        current_room_hint: str,
        current_floor_hint,
    ) -> np.ndarray:
        """为 VLM 获取楼层平面图：优先使用 VLM 判定的楼层；否则回退到房间推断。"""
        floor_num = None
        if current_floor_hint is not None:
            try:
                floor_num = int(current_floor_hint)
            except Exception:
                floor_num = None

        if floor_num is not None:
            return self._get_floorplan_by_floor(floor_num)
        return self._get_floorplan(current_room_hint)

    def _get_floorplan_by_floor(self, floor_num: int) -> np.ndarray:
        """按楼层号从 data/map 加载平面图（供 VLM 使用）。"""
        # 无 cv2 时：返回纯色图像
        back_res = (
            self.env_config.get("sensors", {})
            .get("back_camera", {})
            .get("resolution", [480, 640])
        )
        try:
            h, w = int(back_res[0]), int(back_res[1])
        except Exception:
            h, w = 480, 640

        if cv2 is None:
            return np.zeros((h, w, 3), dtype=np.uint8)

        floorplan_dir = (
            self.paths_config.get("data", {}).get("floorplan_dir")
            or self.paths_config.get("floorplan_dir")
            or "data/map"
        )
        floorplan_path = Path(floorplan_dir) / f"{floor_num}F.jpg"
        if floorplan_path.exists():
            img = cv2.imread(str(floorplan_path))
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        from src.video_recorder import FloorplanGenerator

        dummy_bgr = FloorplanGenerator.create_dummy_floorplan(w, h)
        return cv2.cvtColor(dummy_bgr, cv2.COLOR_BGR2RGB)

    class _NullRecorder:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write_frame(self, *args, **kwargs):
            return None

    def _create_video_recorder(self):
        """创建视频录制器；缺少 cv2 时自动降级为 no-op。"""
        if cv2 is None:
            self.logger.warning("⚠️ 未安装 opencv-python，跳过视频录制")
            return self._NullRecorder()
        try:
            return BackCameraRecorder(str(self.dirs["videos"]))
        except Exception as e:
            self.logger.warning(f"⚠️ 视频录制初始化失败，跳过录制: {e}")
            return self._NullRecorder()

    def _load_floor_maps(self, paths: List[Tuple]) -> Dict[int, np.ndarray]:
        """加载楼层平面图"""
        from src.video_recorder import FloorplanGenerator

        floor_maps = {}
        all_floors = set()

        for rooms, _, _ in paths:
            for room in rooms:
                floor = self.scene_graph.extract_floor(room)
                all_floors.add(floor)

        for floor_num in all_floors:
            # 尝试从文件加载，如果不存在则创建虚拟平面图
            floorplan_dir = (
                self.paths_config.get("data", {}).get("floorplan_dir")
                or self.paths_config.get("floorplan_dir")
                or "data/map"
            )
            floorplan_path = Path(floorplan_dir) / f"{floor_num}F.jpg"
            if cv2 is not None and floorplan_path.exists():
                img = cv2.imread(str(floorplan_path))
                if img is not None:
                    floor_maps[floor_num] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    continue

            # 创建虚拟平面图
            dummy_bgr = FloorplanGenerator.create_dummy_floorplan(1280, 720)
            if cv2 is None:
                # 没有 cv2 就直接返回 RGB 零矩阵（保证 Agent1 仍可工作）
                floor_maps[floor_num] = np.zeros((720, 1280, 3), dtype=np.uint8)
            else:
                floor_maps[floor_num] = cv2.cvtColor(dummy_bgr, cv2.COLOR_BGR2RGB)

        return floor_maps

    def _get_floorplan(self, room_id: str) -> np.ndarray:
        """获取指定房间所在楼层的平面图"""
        # 无 cv2 时：返回纯色图像，仍可用于 VLM 结构输入
        back_res = (
            self.env_config.get("sensors", {})
            .get("back_camera", {})
            .get("resolution", [480, 640])
        )
        try:
            h, w = int(back_res[0]), int(back_res[1])
        except Exception:
            h, w = 480, 640

        if cv2 is None:
            return np.zeros((h, w, 3), dtype=np.uint8)

        from src.video_recorder import FloorplanGenerator

        floor_num = None
        try:
            if self.scene_graph is not None:
                floor_num = self.scene_graph.extract_floor(room_id)
        except Exception:
            floor_num = None

        floorplan_dir = (
            self.paths_config.get("data", {}).get("floorplan_dir")
            or self.paths_config.get("floorplan_dir")
            or "data/map"
        )
        if floor_num is not None:
            floorplan_path = Path(floorplan_dir) / f"{floor_num}F.jpg"
            if floorplan_path.exists():
                img = cv2.imread(str(floorplan_path))
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        dummy_bgr = FloorplanGenerator.create_dummy_floorplan(w, h)
        return cv2.cvtColor(dummy_bgr, cv2.COLOR_BGR2RGB)

    def _save_rgb_frame(self, rgb_frame: np.ndarray, camera_type: str, step: int):
        """保存图像帧到文件（VLM决策时使用；保存到 frames 根目录，无子文件夹）"""
        if cv2 is None:
            return None
        if rgb_frame is None or rgb_frame.size == 0:
            return None

        # 遵循配置开关
        saving_cfg = (
            self.system_config.get("data_saving", {})
            if isinstance(self.system_config, dict)
            else {}
        )
        if not bool(saving_cfg.get("save_rgb_frames", True)):
            return None

        try:
            save_dir = self.dirs["frames"]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            fmt = str(saving_cfg.get("rgb_frame_format", "jpeg")).lower()
            if fmt in ("jpg", "jpeg"):
                ext = "jpg"
            elif fmt == "png":
                ext = "png"
            else:
                ext = "jpg"
            tag = str(camera_type or "frame").strip().lower()
            filename = f"step_{step:04d}_{tag}_{timestamp}.{ext}"
            filepath = save_dir / filename

            # RGB图像转BGR用于cv2.imwrite
            bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            write_params = []
            if ext == "jpg":
                q = int(saving_cfg.get("rgb_frame_quality", 90))
                write_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(q, 1, 100))]
            cv2.imwrite(str(filepath), bgr, write_params)
            self.logger.debug(f"✅ 保存帧: {filename}")
            return filepath

        except Exception as e:
            self.logger.warning(f"⚠️ 保存RGB帧失败: {e}")
            return None

    def _save_metrics(self, success: bool, step_count: int, path_idx: int):
        """保存导航指标"""
        metrics = {
            "success": success,
            "total_steps": step_count,
            "path_length": len(self.selected_path) - 1,
            "start_room": self.args.start,
            "end_room": self.args.end,
            "path": self.selected_path,
            "path_progress": path_idx + 1,
            "timestamp": datetime.now().isoformat(),
        }

        metrics_file = (
            self.dirs["metrics"]
            / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        self.logger.info(
            f"✅ 导航指标已保存: metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )


def main():
    """主程序入口"""
    args = parse_arguments()

    # 创建运行器并执行
    runner = VLMNavigationRunner(args)
    success = runner.run()

    # 返回适当的退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
