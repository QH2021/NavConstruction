#!/usr/bin/env python3
"""
多智能体VLM导航系统 - 核心导航逻辑
=================================

设计流程：
阶段1: 路径规划
  - 读取起点和终点
  - 场景图生成3条候选路径
  - Agent1选择最优路径

阶段2: 逐步导航（单步循环）
  - 循环直到VLM判断到达目标：
    - RGB传感器采集当前场景
    - 输入RGB+楼层平面图到VLM
    - Agent1生成下一步动作（单步）
    - Agent1判断是否到达目标房间
    - 如果到达，输出"导航结束"并停止
    - 否则执行动作，保存视频帧，继续循环
"""

import os
import json
import logging
import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import time
from collections import deque

# 自定义模块
# 兼容两种导入方式：
# 1) 作为包导入：import src.navigation_system
# 2) 作为模块导入：sys.path.insert(0, "src"); import navigation_system
try:  # pragma: no cover
    from .scene_graph import SceneGraph
    from .agents import Agent1, Agent2
    from .video_recorder import BackCameraRecorder, FloorplanGenerator
    from .habitat_integration import EnvironmentFactory
except Exception:  # pragma: no cover
    from scene_graph import SceneGraph
    from agents import Agent1, Agent2
    from video_recorder import BackCameraRecorder, FloorplanGenerator
    from habitat_integration import EnvironmentFactory

logger = logging.getLogger(__name__)


def _require_cv2():
    if cv2 is None:
        raise ImportError(
            "缺少依赖 opencv-python（cv2）。如需图像/视频/可视化功能，请先执行: pip install opencv-python"
        )


class NavigationSystem:
    """多智能体VLM导航系统"""

    def __init__(
        self,
        door_excel: str = None,
        comp_excel: str = None,
        output_dir: str = None,
        enable_agent2: bool = None,
        vlm_url: str = None,
        max_steps: int = None,
        goal_distance: float = None,
        config_dir: str = "./configs",
    ):
        """
        初始化导航系统

        Args:
            door_excel: door_table.xlsx路径（None则从配置读取）
            comp_excel: component_table.xlsx路径（None则从配置读取）
            output_dir: 输出目录（None则从配置读取）
            enable_agent2: 是否启用Agent2（None则从配置读取）
            vlm_url: VLM API地址（None则从配置读取）
            max_steps: 最大导航步数（None则从配置读取）
            goal_distance: 到达目标的距离阈值（None则从配置读取）
            config_dir: 配置文件目录
        """
        # 加载配置
        try:
            from src.config_loader import UnifiedConfigLoader
        except Exception:  # pragma: no cover
            from config_loader import UnifiedConfigLoader

        self._config_loader = UnifiedConfigLoader(config_dir=config_dir)

        # 加载统一配置文件
        try:
            unified_config = self._config_loader.load_unified_config()
            paths_config = unified_config.get("paths", {})
            system_config = unified_config.get("system", {})
            vlm_config = unified_config.get("vlm", {})
            agent_config = unified_config.get("agents", {})
            env_config = unified_config.get("environment", {})
            navigation_config = unified_config.get("navigation", {})
        except Exception:
            # 回退到独立配置文件（兼容旧配置）
            paths_config = self._config_loader.load_paths_config()
            system_config = self._config_loader.load_config("system_config")
            vlm_config = self._config_loader.load_vlm_config()
            agent_config = self._config_loader.load_config("agent_config")
            env_config = self._config_loader.load_environment_config()
            navigation_config = {}

        # Depth 传感器开关（前置同位）
        try:
            sensors_cfg = (
                env_config.get("sensors", {}) if isinstance(env_config, dict) else {}
            )
            depth_cfg = (
                sensors_cfg.get("front_depth", {})
                if isinstance(sensors_cfg, dict)
                else {}
            )
            self.enable_front_depth = bool(depth_cfg.get("enabled", False))
        except Exception:
            self.enable_front_depth = False

        # 使用配置中的默认值
        door_excel = door_excel or paths_config.get("data", {}).get(
            "door_table", "./data/door_table.xlsx"
        )
        comp_excel = comp_excel or paths_config.get("data", {}).get(
            "component_table", "./data/component_table.xlsx"
        )
        output_dir = output_dir or system_config.get("output", {}).get(
            "base_dir", "./output"
        )
        enable_agent2 = (
            enable_agent2
            if enable_agent2 is not None
            else agent_config.get("agent2", {}).get("enabled", False)
        )
        vlm_url = vlm_url or vlm_config.get("api", {}).get(
            "endpoint", "http://localhost:8000/v1/chat/completions"
        )

        # 读取导航参数：优先 unified_config.navigation.navigation_loop
        nav_loop = {}
        if isinstance(navigation_config, dict):
            nav_loop = navigation_config.get("navigation_loop", {}) or {}

        # 兼容旧字段：environment_config.navigation.*
        legacy_nav = (
            env_config.get("navigation", {}) if isinstance(env_config, dict) else {}
        )

        if max_steps is None:
            max_steps = (
                nav_loop.get("max_steps") if isinstance(nav_loop, dict) else None
            )
            if max_steps is None:
                max_steps = legacy_nav.get("max_steps", 200)

        if goal_distance is None:
            # unified: goal_distance_threshold
            goal_distance = (
                nav_loop.get("goal_distance_threshold")
                if isinstance(nav_loop, dict)
                else None
            )
            if goal_distance is None:
                goal_distance = legacy_nav.get("goal_distance", 0.5)

        # 创建输出目录（Agent需要output_dir）
        self.output_dir = (
            Path(output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化场景图和智能体（检查Excel文件是否存在）
        self.scene_graph = None
        if Path(door_excel).exists() and Path(comp_excel).exists():
            try:
                self.scene_graph = SceneGraph(door_excel, comp_excel)
            except Exception as e:
                logger.warning(f"⚠️ 场景图初始化失败: {e}")
                logger.info("   将使用虚拟场景图")
                self.scene_graph = None
        else:
            logger.warning(f"⚠️ Excel文件不存在")
            logger.info(f"   door_excel: {door_excel}")
            logger.info(f"   comp_excel: {comp_excel}")
            logger.info("   将使用虚拟场景图")

        self.agent1 = Agent1(config_dir=config_dir, output_dir=str(self.output_dir))
        self.agent2 = (
            Agent2(config_dir=config_dir, output_dir=str(self.output_dir))
            if enable_agent2
            else None
        )

        # 子目录
        self.frames_dir = self.output_dir / "frames"
        self.videos_dir = self.output_dir / "videos"
        self.logs_dir = self.output_dir / "logs"
        self.metrics_dir = self.output_dir / "metrics"
        self.paths_dir = self.output_dir / "paths"

        for d in [
            self.frames_dir,
            self.videos_dir,
            self.logs_dir,
            self.metrics_dir,
            self.paths_dir,
        ]:
            d.mkdir(exist_ok=True)

        self.vlm_url = vlm_url
        self.max_steps = max_steps
        self.goal_distance = goal_distance

        # 导航状态
        self.current_path: Optional[List[str]] = None
        self.current_room: Optional[str] = None
        self.target_room: Optional[str] = None
        self.step_count = 0
        self.video_recorder: Optional[BackCameraRecorder] = None

        logger.info(f"✅ 导航系统初始化完成")
        logger.info(f"   输出目录: {self.output_dir}")
        logger.info(f"   启用Agent2: {self.agent2 is not None}")
        logger.info(f"   启用前置Depth: {self.enable_front_depth}")

    def plan_navigation(self, start_input: str, end_input: str) -> bool:
        """
        阶段1: 路径规划

        1. A*算法生成3条候选路径
        2. Agent1基于所有楼层平面图选择最优路径
        3. 保存路径信息
        """
        logger.info("=" * 70)
        logger.info("🗺️  阶段1: 路径规划（A*算法 + VLM选择）")
        logger.info("=" * 70)

        # 1. 使用A*生成候选路径
        paths = self.scene_graph.find_k_shortest_paths(start_input, end_input, k=3)
        if not paths:
            logger.error("❌ A*算法无法生成路径")
            return False

        logger.info(f"✅ A*算法生成 {len(paths)} 条候选路径")
        for i, (rooms, doors, steps) in enumerate(paths, 1):
            logger.info(f"   路径{i}: {' → '.join(rooms[:3])}... ({steps}步)")

        # 保存候选路径
        candidates = [
            {
                "id": i + 1,
                "rooms": rooms,
                "doors": doors,
                "steps": steps,
                "details": self.scene_graph.get_path_details(rooms),
            }
            for i, (rooms, doors, steps) in enumerate(paths)
        ]

        candidates_file = self.output_dir / "candidates.json"
        with open(candidates_file, "w") as f:
            json.dump(candidates, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ 候选路径已保存: {candidates_file}")

        # 2. 加载所有楼层平面图
        floor_maps = self._load_floor_maps(paths)

        # 3. Agent1选择最优路径
        try:
            selected_idx = self.agent1.select_best_path(
                paths, floor_maps, start_input, end_input
            )
        except Exception as e:
            logger.warning(f"⚠️ Agent1选择异常: {e}，使用默认路径1")
            selected_idx = 0

        self.current_path = paths[selected_idx][0]
        self.current_room = self.current_path[0]
        self.target_room = self.current_path[-1]

        logger.info(
            f"✅ Agent1选择路径{selected_idx + 1}: {' → '.join(self.current_path)}"
        )

        # 保存路径信息
        path_info = {
            "start": self.current_room,
            "end": self.target_room,
            "path": self.current_path,
            "selected_index": selected_idx + 1,
            "total_steps": len(self.current_path) - 1,
            "timestamp": datetime.now().isoformat(),
        }
        path_file = self.output_dir / "selected_path.json"
        with open(path_file, "w") as f:
            json.dump(path_info, f, ensure_ascii=False, indent=2)

        return True

    def _load_floor_maps(self, paths: List[Tuple]) -> Dict[int, np.ndarray]:
        """加载涉及到的所有楼层平面图"""
        _require_cv2()
        from src.video_recorder import FloorplanGenerator

        # 收集所有涉及的楼层
        all_floors = set()
        for rooms, _, _ in paths:
            for room in rooms:
                floor = self.scene_graph.extract_floor(room)
                all_floors.add(floor)

        # 加载或生成平面图
        floor_maps = {}
        for floor_num in all_floors:
            # TODO: 从实际文件加载，这里使用占位符
            dummy_map_bgr = FloorplanGenerator.create_dummy_floorplan(1280, 720)
            # FloorplanGenerator 返回 BGR，这里统一转换为 RGB
            floor_maps[floor_num] = cv2.cvtColor(dummy_map_bgr, cv2.COLOR_BGR2RGB)
            logger.debug(f"加载楼层 {floor_num}F 平面图")

        return floor_maps

    def _save_rgb_image(self, rgb_image: np.ndarray, camera_type: str, step: int):
        """
        保存RGB传感器图像到文件

        Args:
            rgb_image: RGB图像数组
            camera_type: 摄像头类型 ('front' 或 'back')
            step: 当前步数
        """
        if rgb_image is None or rgb_image.size == 0:
            return

        _require_cv2()

        try:
            # 保存到 frames 根目录（不创建子文件夹）
            save_dir = self.frames_dir

            # 生成文件名: step_<步数>_<camera>_rgb_<时间戳>.jpg
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            cam = "front" if camera_type == "front" else "back"
            filename = f"step_{step:04d}_{cam}_rgb_{timestamp}.jpg"
            filepath = save_dir / filename

            # 保存图像 (BGR格式用于cv2)
            cv2.imwrite(str(filepath), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
            logger.debug(f"✅ 保存RGB图像: {filepath.name}")
            return filepath
        except Exception as e:
            logger.warning(f"⚠️ 保存RGB图像失败: {e}")
        return None

    def _save_front_depth_vis(self, depth_front: np.ndarray, step: int):
        """保存前置深度（用于VLM的可视化图）到 frames 根目录。"""
        if depth_front is None or (
            hasattr(depth_front, "size") and depth_front.size == 0
        ):
            return None
        _require_cv2()

        try:
            # 尽量复用 Agent 的 depth 可视化规则，保证“传入VLM的深度信息”与落盘一致
            depth_vis = None
            try:
                depth_vis = self.agent1._depth_to_vis_rgb(depth_front)
            except Exception:
                depth_vis = None

            if not isinstance(depth_vis, np.ndarray) or depth_vis.size == 0:
                return None

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"step_{step:04d}_front_depth_{timestamp}.jpg"
            filepath = self.frames_dir / filename
            cv2.imwrite(str(filepath), cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))
            return filepath
        except Exception as e:
            logger.warning(f"⚠️ 保存Depth图像失败: {e}")
            return None

    def _get_floorplan(self, floor_num: int) -> np.ndarray:
        """获取指定楼层的平面图"""
        _require_cv2()
        try:
            from src.video_recorder import FloorplanGenerator
        except ModuleNotFoundError:
            # 兼容 tests 把 ./src 加入 sys.path 的导入方式
            from video_recorder import FloorplanGenerator

        # TODO: 从实际文件加载楼层平面图
        # 当前使用占位符
        floorplan_path = f"data/map/{floor_num}F.jpg"
        if Path(floorplan_path).exists():
            floorplan = cv2.imread(floorplan_path)
            if floorplan is not None:
                return cv2.cvtColor(floorplan, cv2.COLOR_BGR2RGB)

        # 如果文件不存在，创建虚拟平面图
        logger.debug(f"使用虚拟平面图（{floor_num}F）")
        dummy_bgr = FloorplanGenerator.create_dummy_floorplan(1280, 720)
        return cv2.cvtColor(dummy_bgr, cv2.COLOR_BGR2RGB)

    def _save_video_frame(
        self,
        rgb_back: np.ndarray,
        floorplan: np.ndarray,
        navigable_map: np.ndarray,
        agent_pos: np.ndarray,
        current_room: str,
        status: str,
    ):
        """保存视频帧到录制器"""
        if not self.video_recorder:
            return

        # 准备指标数据
        metrics = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "position": f"({agent_pos[0]:.2f}, {agent_pos[1]:.2f})",
            "room": current_room,
            "battery": 85.0,  # 模拟电池电量
            "steps": self.step_count,
            "status": status,
        }

        # 调整尺寸并保存
        rgb_back_resized = cv2.resize(rgb_back, (1280, 960))

        # 混合平面图和可通行地图 (简单的叠加或并排，这里选择叠加或优先显示可通行图)
        # 为了清晰，我们使用可通行地图作为主要地图显示
        if navigable_map is not None and navigable_map.size > 0:
            # 确保尺寸匹配
            if navigable_map.shape[:2] != (720, 1280):
                try:
                    nav_map_resized = cv2.resize(navigable_map, (1280, 720))
                    # 如果是灰度图，转为RGB
                    if len(nav_map_resized.shape) == 2:
                        nav_map_resized = cv2.cvtColor(
                            nav_map_resized, cv2.COLOR_GRAY2BGR
                        )
                    display_map = nav_map_resized
                except:
                    display_map = floorplan
            else:
                display_map = navigable_map
        else:
            display_map = floorplan

        self.video_recorder.write_frame(
            rgb_back_resized,
            floorplan=display_map,
            robot_position=tuple(agent_pos[:2]),
            robot_heading=0.0,
            metrics=metrics,
        )

    def execute_navigation(self, env: "UnifiedEnvironment") -> bool:
        """
        阶段2: 逐步导航（VLM批量决策 + 执行）

        循环执行：
        1. 采集RGB图像（前置摄像头）
        2. Agent1基于RGB+平面图生成4步动作序列
        3. 依次执行动作，每步检查碰撞和目标状态
        4. 如果遇到障碍，触发Agent2重规划
        5. 如果到达目标，结束导航
        """
        logger.info("=" * 70)
        logger.info("🚀 阶段2: 逐步导航（VLM批量决策）")
        logger.info("=" * 70)

        if not self.current_path:
            logger.error("❌ 未规划路径，无法开始导航")
            return False

        # 初始化视频录制器
        with BackCameraRecorder(str(self.videos_dir)) as self.video_recorder:
            # 重置环境
            obs = env.reset()
            path_idx = 0  # 当前在路径中的索引（从0开始）

            # 维护最近两帧（t-1,t），用于VLM判断动作（避免只输入最后一帧）
            # 注意：系统已切换为“最后一次动作后的观测”(t) 单帧输入给 VLM。
            # 这里保留变量声明仅用于未来扩展/兼容，但不再用于 VLM 输入拼接。
            front_rgb_history: deque = deque(maxlen=2)
            front_depth_history: deque = deque(maxlen=2)

            logger.info(f"导航路径: {' → '.join(self.current_path)}")
            logger.info(f"路径长度: {len(self.current_path)} 个房间")
            logger.info(f"最大步数: {self.max_steps}\n")

            while self.step_count < self.max_steps:
                logger.info(
                    f"\n{'=' * 60}\n步骤 {self.step_count + 1}/{self.max_steps}\n{'=' * 60}"
                )

                # 1. 采集数据（注意：历史帧只在动作执行后更新；此处不重复append，避免 t-1==t）
                rgb_front = obs.get(
                    "rgb_front", np.zeros((720, 1280, 3), dtype=np.uint8)
                )
                depth_front = obs.get(
                    "depth_front", np.zeros((720, 1280), dtype=np.float32)
                )
                agent_pos = obs.get("agent_pos", np.array([0, 0, 0]))

                # 单帧策略：不再依赖历史帧构造 (t-1,t) 输入。

                # 保存前置RGB图像（VLM使用）
                front_rgb_path = self._save_rgb_image(
                    rgb_front, "front", self.step_count
                )

                # 保存前置Depth可视化（VLM使用）
                front_depth_path = None
                if self.enable_front_depth:
                    front_depth_path = self._save_front_depth_vis(
                        depth_front, self.step_count
                    )

                # 获取当前房间
                current_room = (
                    self.current_path[path_idx]
                    if path_idx < len(self.current_path)
                    else self.current_path[-1]
                )

                logger.info(f"📍 当前房间: {current_room}")
                logger.info(f"🎯 目标房间: {self.target_room}")
                logger.info(f"📊 路径进度: {path_idx + 1}/{len(self.current_path)}")

                # 获取平面图和可通行地图
                current_floor = self.scene_graph.extract_floor(current_room)
                floorplan = self._get_floorplan(current_floor)
                navigable_map = env.get_navigable_map()

                # 2. Agent1: 生成动作序列（4步）
                logger.info("\n🤖 Agent1 正在分析场景...")

                depth_last = None
                if self.enable_front_depth:
                    depth_last = [depth_front]

                result = self.agent1.generate_action_batch(
                    rgb_image=rgb_front,
                    depth_images=depth_last,
                    floorplan=floorplan,
                    current_room=current_room,
                    target_room=self.target_room,
                    path_rooms=self.current_path,
                    context={
                        "step": self.step_count,
                        "path_index": path_idx,
                        "total_rooms": len(self.current_path),
                        "front_rgb_path": str(front_rgb_path)
                        if front_rgb_path
                        else None,
                        "front_depth_path": str(front_depth_path)
                        if front_depth_path
                        else None,
                    },
                )

                if not result:
                    logger.error("❌ Agent1无法生成动作，VLM响应失败")
                    return False

                actions = result.get("actions", [])
                reached_goal = result.get("reached_goal", False)
                reasoning = result.get("reasoning", "无推理信息")

                logger.info(f"\n💡 VLM决策:")
                logger.info(f"   动作序列: {actions}")
                logger.info(f"   到达目标: {reached_goal}")
                logger.info(f"   推理: {reasoning}")

                # 3. 依次执行动作
                batch_interrupted = False
                for action in actions:
                    if self.step_count >= self.max_steps:
                        break

                    # 检查是否到达目标（VLM判断）
                    if reached_goal and action == "stop":
                        logger.info("\n" + "=" * 70)
                        logger.info(f"✅ VLM判断: 已到达目标房间 {self.target_room}！")
                        logger.info(f"📝 推理依据: {reasoning}")
                        logger.info(f"📊 总步数: {self.step_count + 1}")
                        logger.info("🏁 导航任务结束！")
                        logger.info("=" * 70)

                        # 保存最终帧
                        rgb_back = obs.get(
                            "rgb_back", np.zeros((480, 640, 3), dtype=np.uint8)
                        )
                        self._save_video_frame(
                            rgb_back,
                            floorplan,
                            navigable_map,
                            agent_pos,
                            current_room,
                            "Arrived",
                        )
                        self._save_metrics(True)
                        return True

                    # 执行动作
                    logger.info(f"\n🎮 执行动作: {action}")
                    prev_pos = agent_pos.copy()
                    obs, info = env.step(action)
                    self.step_count += 1

                    # 更新状态
                    agent_pos = obs.get("agent_pos", np.array([0, 0, 0]))
                    rgb_front = obs.get(
                        "rgb_front", np.zeros((720, 1280, 3), dtype=np.uint8)
                    )
                    depth_front = obs.get(
                        "depth_front", np.zeros((720, 1280), dtype=np.float32)
                    )
                    rgb_back = obs.get(
                        "rgb_back", np.zeros((480, 640, 3), dtype=np.uint8)
                    )
                    navigable_map = env.get_navigable_map()

                    # 更新历史帧（用于下一轮 VLM：t-1,t）
                    try:
                        if isinstance(rgb_front, np.ndarray) and rgb_front.size > 0:
                            front_rgb_history.append(rgb_front.copy())
                    except Exception:
                        pass
                    if self.enable_front_depth:
                        try:
                            if (
                                isinstance(depth_front, np.ndarray)
                                and depth_front.size > 0
                            ):
                                front_depth_history.append(depth_front.copy())
                        except Exception:
                            pass

                    # 保存后置RGB图像
                    self._save_rgb_image(rgb_back, "back", self.step_count)

                    # 保存视频帧
                    self._save_video_frame(
                        rgb_back,
                        floorplan,
                        navigable_map,
                        agent_pos,
                        current_room,
                        "Navigating",
                    )

                    # 检查碰撞/卡死 (简单的位置变化检查)
                    if action == "move_forward":
                        dist_moved = np.linalg.norm(agent_pos - prev_pos)
                        if dist_moved < 0.05:  # 移动距离过小，认为卡住
                            logger.warning(
                                f"⚠️ 检测到碰撞或卡死 (移动距离: {dist_moved:.3f})"
                            )

                            # 触发Agent2重规划
                            if self.agent2_enabled:
                                logger.info("🔄 呼叫 Agent2 进行重规划...")
                                # 假设当前房间到下一个房间的路径受阻
                                next_room_idx = path_idx + 1
                                if next_room_idx < len(self.current_path):
                                    next_room = self.current_path[next_room_idx]
                                    blocked_edge = (current_room, next_room)

                                    new_path = self.agent2.replan_path(
                                        current_room=current_room,
                                        target_room=self.target_room,
                                        blocked_edge=blocked_edge,
                                        scene_graph=self.scene_graph,
                                    )

                                    if new_path:
                                        self.current_path = new_path
                                        path_idx = 0  # 重置索引（因为路径变了，需要重新匹配当前位置，这里简化为从头开始匹配或假设当前就在new_path[0]）
                                        # 实际上应该找到当前位置在new_path中的位置
                                        try:
                                            path_idx = new_path.index(current_room)
                                        except ValueError:
                                            path_idx = 0

                                        logger.info(
                                            f"✅ 路径已更新: {' → '.join(self.current_path)}"
                                        )
                                        batch_interrupted = True
                                        break  # 跳出动作循环，重新开始VLM决策
                                    else:
                                        logger.error("❌ Agent2 重规划失败")
                            else:
                                logger.warning("⚠️ Agent2 未启用，尝试继续...")

                    # 更新路径进度
                    if info.get("distance_to_goal", 100) < self.goal_distance:
                        # 简单的进度更新逻辑，实际可能需要更复杂的定位
                        if path_idx < len(self.current_path) - 1:
                            path_idx += 1

                if batch_interrupted:
                    continue

        self._save_metrics(False)
        return False

    def _save_metrics(self, success: bool):
        """保存导航指标和VLM记录"""
        # 安全获取Agent记忆
        agent1_memory = []
        agent2_memory = []

        try:
            if self.agent1 and hasattr(self.agent1, "memory"):
                agent1_memory = list(self.agent1.memory)
        except:
            pass

        try:
            if self.agent2 and hasattr(self.agent2, "memory"):
                agent2_memory = list(self.agent2.memory)
        except:
            pass

        # 转换记忆对象为字典
        agent1_history = []
        agent2_history = []

        try:
            agent1_history = [
                m.data if hasattr(m, "data") else m for m in agent1_memory
            ]
        except:
            pass

        try:
            agent2_history = [
                m.data if hasattr(m, "data") else m for m in agent2_memory
            ]
        except:
            pass

        metrics = {
            "success": success,
            "total_steps": self.step_count,
            "path_length": len(self.current_path) - 1 if self.current_path else 0,
            "start_room": self.current_room,
            "end_room": self.target_room,
            "path": self.current_path,
            "timestamp": datetime.now().isoformat(),
        }

        # 安全添加Agent统计
        try:
            if self.agent1 and hasattr(self.agent1, "get_stats"):
                metrics["agent1_stats"] = self.agent1.get_stats()
        except:
            pass

        try:
            if self.agent2 and hasattr(self.agent2, "get_stats"):
                metrics["agent2_stats"] = self.agent2.get_stats()
        except:
            pass

        # 添加VLM历史
        if agent1_history or agent2_history:
            metrics["vlm_history"] = {
                "agent1": agent1_history,
                "agent2": agent2_history,
            }

        metrics_file = (
            self.metrics_dir
            / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(metrics_file, "w") as f:
            # 使用default=str处理可能的非序列化对象
            json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"✅ 导航指标和VLM记录已保存: {metrics_file}")

    def run(
        self,
        start_input: str,
        end_input: str,
        use_habitat: bool = True,
        enable_visualization: bool = False,
    ) -> bool:
        """
        完整导航流程

        Args:
            start_input: 起点
            end_input: 终点
            use_habitat: 是否使用Habitat环境（False则使用虚拟环境）
            enable_visualization: 是否启用GUI可视化

        Returns:
            是否成功
        """
        logger.info(f"\n🎯 开始导航任务: {start_input} -> {end_input}\n")

        # 阶段1: 路径规划
        if not self.plan_navigation(start_input, end_input):
            return False

        # 初始化环境
        env = None
        if use_habitat:
            try:
                from src.habitat_integration import EnvironmentFactory

                factory = EnvironmentFactory(logger)

                # 从environment_config读取场景路径和机器人配置
                env_cfg = self._config_loader.load_environment_config()
                scene_path = env_cfg.get("scene", {}).get(
                    "path", "./data/scene_datasets/habitat-test-scenes/3dExport1208.glb"
                )

                config = {
                    "scene_path": scene_path,
                    "use_habitat_lab": False,
                    "agent_config": env_cfg.get("agent", {}),
                    "sim_config": {
                        "enable_physics": env_cfg.get("physics", {}).get(
                            "enabled", True
                        ),
                        "physics_config_file": "data/default.physics_config.json",
                    },
                    "env_config": env_cfg,  # 传递完整的environment配置
                    "enable_visualization": enable_visualization,  # 传递可视化参数
                }
                env, _ = factory.create_environment(config)
                logger.info(f"✅ Habitat环境初始化成功 (场景: {scene_path})")
                if enable_visualization:
                    logger.info("🖼️  可视化界面已启用")
            except Exception as e:
                logger.warning(f"⚠️ Habitat环境初始化失败: {e}")
                logger.info("切换到虚拟环境模式...")
                env = None

        # 如果Habitat不可用，使用虚拟环境
        if env is None:
            from src.habitat_integration import MockEnvironment

            env = MockEnvironment(logger)
            logger.info("✅ 虚拟环境初始化成功（测试模式）")

        # 阶段2: 逐步导航
        try:
            success = self.execute_navigation(env)
        finally:
            env.close()

        return success


def setup_logging(log_dir: str):
    """设置日志"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    log_file = (
        Path(log_dir) / f"navigation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    return log_file


if __name__ == "__main__":
    # 设置日志
    log_file = setup_logging("./logs")

    logger.info("🚀 多智能体VLM导航系统启动")
    logger.info(f"日志文件: {log_file}")

    # 初始化导航系统（从配置文件读取所有参数）
    nav_system = NavigationSystem(
        config_dir="./configs",
        enable_agent2=True,  # 可以覆盖配置
    )

    # 运行导航
    try:
        success = nav_system.run(
            start_input="Start01", end_input="R309", use_habitat=True
        )

        if success:
            logger.info("✅ 导航任务完成")
        else:
            logger.error("❌ 导航任务失败")

    except KeyboardInterrupt:
        logger.info("⚠️ 用户中断导航")
    except Exception as e:
        logger.error(f"❌ 导航系统异常: {e}")
        import traceback

        traceback.print_exc()

    logger.info("🔌 导航系统关闭")
