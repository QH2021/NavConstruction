#!/usr/bin/env python3
"""
Habitat-sim 和 Habitat-lab 集成层
提供统一的环境接口，支持两种框架

功能:
1. Habitat-sim: 低级物理模拟和传感器
2. Habitat-lab: 任务定义、评估指标、工作流管理
3. 自动环境检测和兼容性管理
4. 统一的观测和动作接口
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import sys
import subprocess
from abc import ABC, abstractmethod
from enum import Enum
import json

# ============================================================================
# 第一部分: Habitat 框架检测和版本管理
# ============================================================================


class HabitatFramework(Enum):
    """支持的Habitat框架类型"""

    HABITAT_SIM_ONLY = "habitat-sim"  # 仅使用habitat-sim
    HABITAT_LAB_INTEGRATED = "habitat-lab"  # 集成habitat-lab任务框架
    HYBRID = "hybrid"  # 混合模式


class HabitatVersionInfo:
    """Habitat框架版本信息管理"""

    def __init__(self):
        """检测安装的Habitat框架"""
        self.logger = logging.getLogger("HabitatVersionInfo")
        self.sim_available = False
        self.lab_available = False
        self.sim_version = None
        self.lab_version = None
        self.detected_framework = HabitatFramework.HABITAT_SIM_ONLY

        self._detect_frameworks()

    def _detect_frameworks(self):
        """检测已安装的框架"""
        # 检测 habitat-sim
        try:
            import habitat_sim

            self.sim_available = True
            self.sim_version = getattr(habitat_sim, "__version__", "unknown")
            self.logger.info(f"✅ Habitat-sim 已安装 (v{self.sim_version})")
        except ImportError:
            self.logger.warning("⚠️  Habitat-sim 未安装")

        # 检测 habitat-lab
        try:
            import habitat

            self.lab_available = True
            self.lab_version = getattr(habitat, "__version__", "unknown")
            self.logger.info(f"✅ Habitat-lab 已安装 (v{self.lab_version})")
        except ImportError:
            self.logger.info("ℹ️  Habitat-lab 未安装 (可选)")

        # 确定最佳框架组合
        if self.lab_available:
            self.detected_framework = HabitatFramework.HABITAT_LAB_INTEGRATED
        elif self.sim_available:
            self.detected_framework = HabitatFramework.HABITAT_SIM_ONLY
        else:
            raise RuntimeError("❌ 必须安装 Habitat-sim 或 Habitat-lab")

    def get_framework_summary(self) -> str:
        """获取框架摘要信息"""
        summary = f"""
┌─────────────────────────────────────────────────────────┐
│ Habitat 框架检测结果                                     │
├─────────────────────────────────────────────────────────┤
│ 检测到的框架: {self.detected_framework.value:30} │
│ Habitat-sim:  {("v" + self.sim_version if self.sim_available else "未安装"):30} │
│ Habitat-lab:  {("v" + self.lab_version if self.lab_available else "未安装"):30} │
└─────────────────────────────────────────────────────────┘
"""
        return summary


# ============================================================================
# 第二部分: 统一环境接口
# ============================================================================


class UnifiedEnvironment(ABC):
    """统一的环境接口 (抽象基类)

    提供一致的接口供上层导航系统使用，隐藏 habitat-sim 和 habitat-lab 的差异
    """

    def __init__(self, logger: logging.Logger):
        """初始化环境

        Args:
            logger: 日志记录器
        """
        self.logger = logger
        self.sim = None
        self.agent = None
        self.pathfinder = None

    @abstractmethod
    def reset(
        self, start_position: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """重置环境

        Args:
            start_position: 起始位置坐标 (可选)

        Returns:
            初始观测字典
        """
        pass

    @abstractmethod
    def step(self, action: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """执行一步动作

        Args:
            action: 动作名称 ('move_forward', 'turn_left', 'turn_right')

        Returns:
            (观测字典, 信息字典)
        """
        pass

    @abstractmethod
    def get_observations(self) -> Dict[str, np.ndarray]:
        """获取当前观测

        Returns:
            观测字典，包含:
            - 'rgb_front': 前置RGB图像 (H, W, 3)
            - 'depth_front': 前置Depth图像 (H, W)（可选，启用depth传感器时提供；单位通常为米）
            - 'rgb_back': 后置RGB图像 (H, W, 3)
            - 'agent_pos': Agent位置 (3,)
            - 'agent_rot': Agent旋转 (4,) 四元数
        """
        pass

    @abstractmethod
    def get_agent_state(self) -> Dict[str, np.ndarray]:
        """获取Agent状态 (位置和旋转)

        Returns:
            {'position': (3,), 'rotation': (4,)}
        """
        pass

    @abstractmethod
    def set_agent_state(self, position: np.ndarray, rotation: np.ndarray) -> bool:
        """设置Agent状态

        Args:
            position: 位置 (3,)
            rotation: 旋转四元数 (4,)

        Returns:
            是否设置成功
        """
        pass

    @abstractmethod
    def compute_shortest_path(
        self, start: np.ndarray, end: np.ndarray
    ) -> Tuple[List[np.ndarray], float]:
        """计算最短路径

        Args:
            start: 起始位置 (3,)
            end: 目标位置 (3,)

        Returns:
            (路径点列表, 总距离)
        """
        pass

    @abstractmethod
    def close(self):
        """关闭环境，释放资源"""
        pass


# ============================================================================
# 第三部分: Habitat-sim 环境实现
# ============================================================================


class HabitatSimEnvironment(UnifiedEnvironment):
    """Habitat-sim 低级模拟环境实现"""

    def __init__(
        self,
        scene_path: str,
        agent_config: Dict[str, Any],
        sim_config: Dict[str, Any],
        logger: logging.Logger,
        env_config: Optional[Dict[str, Any]] = None,
        enable_visualization: bool = False,
    ):
        """初始化 Habitat-sim 环境

        Args:
            scene_path: 场景文件路径 (.glb 或 .gltf)
            agent_config: Agent 配置字典
            sim_config: 模拟器配置字典
            logger: 日志记录器
            env_config: 环境配置字典（包含robot, scene等配置）
            enable_visualization: 是否启用GUI可视化
        """
        super().__init__(logger)
        self.scene_path = scene_path
        self.agent_config = agent_config
        self.sim_config = sim_config
        self.env_config = env_config or {}  # 保存env_config以供robot加载等使用
        self.enable_visualization = enable_visualization
        self.vis_window_name = "Habitat Navigation"

        # Robot object/reference (best-effort, depends on habitat_sim version)
        self.robot_obj = None
        self.robot_obj_id = None

        # Enhanced viewer style robot-follow rotation state
        self._robot_initial_rotation_wxyz = None
        self._agent_initial_rotation_wxyz = None
        self._robot_rotation_correction_wxyz = None
        self._robot_align_deg = None

        # 动画相关
        self._leg_animation_data = {}
        self._animation_frame = 0
        self._animation_config = None
        
        # 从配置加载动画
        self._load_leg_animation()
        
        self._init_simulator()

    def _load_leg_animation(self):
        """加载腿部动画数据"""
        try:
            robot_cfg = self.env_config.get("robot", {})
            anim_cfg = robot_cfg.get("leg_animation", {})
            
            if not anim_cfg.get("enabled", False):
                self.logger.info("ℹ️  腿部动画未启用")
                return
                
            checkpoint = anim_cfg.get("checkpoint")
            if not checkpoint or not os.path.exists(checkpoint):
                self.logger.warning(f"⚠️  动画文件不存在: {checkpoint}")
                return
                
            self._animation_config = anim_cfg
            use_range = anim_cfg.get("use_range", [0, 10000])
            
            # 读取CSV动画数据
            import csv
            time_i = 0
            with open(checkpoint, 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                next(reader)  # 跳过表头
                
                for row in reader:
                    if use_range[0] <= time_i < use_range[1]:
                        # 解析关节角度
                        joint_angs = row[0].split(',')[1:13]
                        joint_angs = [float(x) for x in joint_angs]
                        
                        # ===== 新增验证 =====
                        if len(joint_angs) != 12:
                            self.logger.warning(f"⚠️  第{time_i}帧数据异常: {len(joint_angs)}个关节")
                            continue
                        
                        self._leg_animation_data[time_i - use_range[0]] = joint_angs
                    time_i += 1
            
            if self._leg_animation_data:
                self.logger.info(
                    f"✅ 加载腿部动画: {len(self._leg_animation_data)} 帧 "
                    f"(将应用到关节 8-19)"
                )
            else:
                self.logger.warning("⚠️  未加载任何动画帧")
            
        except Exception as e:
            self.logger.warning(f"⚠️  动画加载失败: {e}")

    def _init_simulator(self):
        """初始化 Habitat-sim 模拟器"""
        import habitat_sim

        try:
            # 创建摄像头规格
            camera_specs = self._create_camera_specs()

            # 创建动作空间
            action_space = self._create_action_space()

            # 创建 Agent 配置
            agent_cfg = habitat_sim.agent.AgentConfiguration(
                height=self.agent_config.get("height", 0.55),
                radius=self.agent_config.get("radius", 0.28),
                sensor_specifications=camera_specs,
                action_space=action_space,
            )

            # 创建模拟器配置
            sim_cfg = habitat_sim.SimulatorConfiguration()
            sim_cfg.scene_id = self.scene_path
            sim_cfg.enable_physics = self.sim_config.get("enable_physics", True)
            sim_cfg.physics_config_file = self.sim_config.get(
                "physics_config_file", "data/default.physics_config.json"
            )

            # GPU/EGL 设备选择：在部分机器/远程环境中，默认 device=0 可能导致 EGL 上下文创建失败并直接退出进程。
            # 优先级：环境变量 HABITAT_SIM_GPU_DEVICE_ID > unified_config.environment.simulator.gpu_device_id > 默认 0
            try:
                gpu_id = None
                env_gpu = os.environ.get("HABITAT_SIM_GPU_DEVICE_ID")
                if env_gpu is not None and str(env_gpu).strip() != "":
                    gpu_id = int(str(env_gpu).strip())
                else:
                    sim_section = (
                        self.env_config.get("simulator", {})
                        if isinstance(self.env_config, dict)
                        else {}
                    )
                    if (
                        isinstance(sim_section, dict)
                        and sim_section.get("gpu_device_id") is not None
                    ):
                        gpu_id = int(sim_section.get("gpu_device_id"))

                if gpu_id is not None:
                    sim_cfg.gpu_device_id = int(gpu_id)
            except Exception:
                pass

            # 灯光：默认与之前一致；允许通过 unified_config.environment.lighting 覆盖
            lighting_cfg = (
                self.env_config.get("lighting", {})
                if isinstance(self.env_config, dict)
                else {}
            )
            override_lights = bool(
                lighting_cfg.get("override_scene_light_defaults", True)
            )
            sim_cfg.override_scene_light_defaults = override_lights
            light_key = lighting_cfg.get("scene_light_setup", "DEFAULT_LIGHTING_KEY")
            if light_key in (None, "DEFAULT_LIGHTING_KEY", "default", "DEFAULT"):
                sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY
            else:
                sim_cfg.scene_light_setup = str(light_key)

            # 创建模拟器
            cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
            self.sim = habitat_sim.Simulator(cfg)
            self.agent = self.sim.get_agent(0)
            self.pathfinder = self.sim.pathfinder

            # 计算/加载 NavMesh（参考 tests/test_habitat_interactive_enhanced.py）
            try:
                self._ensure_navmesh()
            except Exception as e:
                self.logger.warning(f"⚠️  NavMesh 初始化失败，但继续运行: {e}")

            self.logger.info("✅ Habitat-sim 模拟器初始化完成")

            # 尝试加载机器人模型以便在仿真中可见 (best-effort)
            try:
                self._try_load_robot_model()
                # 确保首帧即可见并与 Agent 对齐
                self._sync_robot_follow_agent()
            except Exception as e:
                self.logger.debug(f"ℹ️  尝试加载机器人模型时发生异常: {e}")

        except Exception as e:
            self.logger.error(f"❌ 模拟器初始化失败: {e}")
            raise

    def _create_camera_specs(self) -> List:
        """创建摄像头规格（参考test_habitat_interactive_enhanced.py的正确实现）"""
        import habitat_sim
        import numpy as np

        specs = []

        sensors_cfg = (
            self.env_config.get("sensors", {})
            if isinstance(self.env_config, dict)
            else {}
        )
        front_cfg = (
            sensors_cfg.get("front_camera", {}) if isinstance(sensors_cfg, dict) else {}
        )
        back_cfg = (
            sensors_cfg.get("back_camera", {}) if isinstance(sensors_cfg, dict) else {}
        )
        depth_cfg = (
            sensors_cfg.get("front_depth", {}) if isinstance(sensors_cfg, dict) else {}
        )

        # 【参考test_habitat_interactive_enhanced.py】前置RGB摄像头 - 第一人称视角
        enable_front = front_cfg.get("enabled")
        if enable_front is None:
            enable_front = self.agent_config.get("enable_front_camera", True)

        if enable_front:
            front_spec = habitat_sim.CameraSensorSpec()
            front_spec.uuid = front_cfg.get("uuid", "front_rgb")
            front_spec.sensor_type = habitat_sim.SensorType.COLOR
            front_spec.resolution = front_cfg.get(
                "resolution",
                self.agent_config.get("front_camera_resolution", [720, 1280]),
            )
            # 【参考代码】位置在agent前方下方 (y 使用 agent.height)
            default_front_pos = [
                0.0,
                float(self.agent_config.get("height", 0.55)),
                -0.6,
            ]
            cfg_pos = front_cfg.get("position", default_front_pos)
            try:
                # 保持 x/z 可配置，但 y 强制与 agent.height 一致
                front_spec.position = [
                    float(cfg_pos[0]),
                    float(default_front_pos[1]),
                    float(cfg_pos[2]),
                ]
            except Exception:
                front_spec.position = default_front_pos
            front_spec.hfov = front_cfg.get(
                "hfov", self.agent_config.get("front_camera_hfov", 90.0)
            )
            # 可选：支持配置欧拉角 orientation=[pitch, roll, yaw]
            try:
                ori = front_cfg.get("orientation")
                if ori is not None:
                    front_spec.orientation = np.array(ori, dtype=float)
            except Exception:
                pass
            specs.append(front_spec)

        # 前置深度摄像头 - 与前置RGB同位（可选）
        enable_depth = depth_cfg.get("enabled")
        if enable_depth is None:
            enable_depth = False

        if enable_depth:
            depth_spec = habitat_sim.CameraSensorSpec()
            depth_spec.uuid = depth_cfg.get("uuid", "front_depth")
            depth_spec.sensor_type = habitat_sim.SensorType.DEPTH

            # 默认继承前置RGB参数，允许单独覆盖
            depth_spec.resolution = depth_cfg.get(
                "resolution",
                front_cfg.get(
                    "resolution",
                    self.agent_config.get("front_camera_resolution", [720, 1280]),
                ),
            )
            default_front_pos = [
                0.0,
                float(self.agent_config.get("height", 0.55)),
                -0.6,
            ]
            cfg_pos = depth_cfg.get(
                "position",
                front_cfg.get("position", default_front_pos),
            )
            try:
                depth_spec.position = [
                    float(cfg_pos[0]),
                    float(default_front_pos[1]),
                    float(cfg_pos[2]),
                ]
            except Exception:
                depth_spec.position = default_front_pos

            depth_spec.hfov = depth_cfg.get(
                "hfov",
                front_cfg.get("hfov", self.agent_config.get("front_camera_hfov", 90.0)),
            )
            try:
                ori = depth_cfg.get("orientation")
                if ori is not None:
                    depth_spec.orientation = np.array(ori, dtype=float)
            except Exception:
                pass

            specs.append(depth_spec)

        # 【参考test_habitat_interactive_enhanced.py】后置俯视摄像头 - 俯视图
        enable_back = back_cfg.get("enabled")
        if enable_back is None:
            enable_back = self.agent_config.get("enable_back_camera", True)

        if enable_back:
            back_spec = habitat_sim.CameraSensorSpec()
            back_spec.uuid = back_cfg.get("uuid", "top_down_view")
            back_spec.sensor_type = habitat_sim.SensorType.COLOR
            back_spec.resolution = back_cfg.get(
                "resolution",
                self.agent_config.get("back_camera_resolution", [480, 640]),
            )
            back_spec.hfov = back_cfg.get(
                "hfov", self.agent_config.get("back_camera_hfov", 90.0)
            )
            # 【参考代码】俯视图的位置：在agent上方1.6米，后方1米
            back_spec.position = back_cfg.get("position", [0.0, 1.6, 1.0])
            # 【参考代码】俯视图的朝向：向下看(pitch=-π/5 ≈ -36度)
            orientation = back_cfg.get("orientation")
            if orientation is None:
                back_spec.orientation = np.array([-np.pi / 5, 0.0, 0.0])
            else:
                back_spec.orientation = np.array(orientation, dtype=float)
            specs.append(back_spec)

        return specs

    def _create_action_space(self) -> Dict:
        """创建动作空间"""
        import habitat_sim

        actions_cfg = (
            self.env_config.get("actions", {})
            if isinstance(self.env_config, dict)
            else {}
        )
        mf_amount = (
            (actions_cfg.get("move_forward", {}) or {}).get("amount")
            if isinstance(actions_cfg, dict)
            else None
        )
        if mf_amount is None:
            mf_amount = self.agent_config.get("action_forward_amount", 0.25)

        tl_amount = (
            (actions_cfg.get("turn_left", {}) or {}).get("amount")
            if isinstance(actions_cfg, dict)
            else None
        )
        if tl_amount is None:
            tl_amount = self.agent_config.get("action_turn_left_amount", 10.0)

        tr_amount = (
            (actions_cfg.get("turn_right", {}) or {}).get("amount")
            if isinstance(actions_cfg, dict)
            else None
        )
        if tr_amount is None:
            tr_amount = self.agent_config.get("action_turn_right_amount", 10.0)

        # 兼容 enhanced viewer：move_backward 通过 move_forward 的负步长实现
        return {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward",
                habitat_sim.agent.ActuationSpec(amount=float(mf_amount)),
            ),
            "move_backward": habitat_sim.agent.ActionSpec(
                "move_forward",
                habitat_sim.agent.ActuationSpec(amount=-float(mf_amount)),
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left",
                habitat_sim.agent.ActuationSpec(amount=float(tl_amount)),
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right",
                habitat_sim.agent.ActuationSpec(amount=float(tr_amount)),
            ),
        }

    def _try_load_robot_model(self):
        """Best-effort: 从 `data/robots/hab_spot_arm` 加载 URDF 或 mesh，并记录结果。"""
        try:
            import glob
            import os

            robot_cfg = (
                self.env_config.get("robot", {})
                if isinstance(self.env_config, dict)
                else {}
            )
            if not bool(robot_cfg.get("enabled", True)):
                self.logger.info("ℹ️  robot.enabled=false，跳过机器人模型加载")
                return False

            # 优先使用配置中的 urdf_path
            urdf_path = self.env_config.get("robot", {}).get("urdf_path")
            if urdf_path and os.path.exists(urdf_path):
                self.logger.info(f"🔧  尝试加载指定URDF: {urdf_path}")
                try:
                    if hasattr(self.sim, "get_articulated_object_manager"):
                        aom = self.sim.get_articulated_object_manager()
                        if hasattr(aom, "add_articulated_object_from_urdf"):
                            # 尽量使用与 enhanced viewer 一致的参数（不同 habitat_sim 版本可能不支持这些 kwargs）
                            try:
                                self.robot_obj = aom.add_articulated_object_from_urdf(
                                    filepath=urdf_path,
                                    fixed_base=False,
                                    global_scale=1.0,
                                    mass_scale=1.0,
                                    force_reload=True,
                                )
                            except TypeError:
                                self.robot_obj = aom.add_articulated_object_from_urdf(
                                    urdf_path
                                )

                            # 兼容：部分版本返回对象，部分返回 id/handle
                            try:
                                self.robot_obj_id = getattr(
                                    self.robot_obj, "object_id", None
                                )
                                if self.robot_obj_id is None:
                                    self.robot_obj_id = getattr(
                                        self.robot_obj, "handle", None
                                    )
                            except Exception:
                                self.robot_obj_id = None
                            self.logger.info(
                                f"✅ 机器狗模型已加载 (URDF), id={self.robot_obj_id}"
                            )

                            # 初始化机器人初始旋转（参考 enhanced viewer：相对旋转跟随）
                            try:
                                self._init_robot_follow_state()
                            except Exception as e:
                                self.logger.debug(f"robot 初始旋转初始化失败: {e}")
                            return True
                except Exception as e:
                    self.logger.warning(f"⚠️ 指定URDF加载失败: {e}")

            # 优先使用传入的 sim_config/agent_config 中指定的路径
            robot_dir = None
            try:
                robot_dir = self.sim_config.get(
                    "robot_model_dir"
                ) or self.agent_config.get("robot_model_dir")
            except Exception:
                robot_dir = None

            # 如果未在实例配置中提供，则使用默认 data 目录下的路径（不依赖外部 get_global_config）
            if not robot_dir:
                robot_dir = os.path.join(os.getcwd(), "data", "robots", "hab_spot_arm")

            if not os.path.isdir(robot_dir):
                self.logger.info(f"ℹ️  未找到机器人模型目录: {robot_dir}")
                return False

            urdf_list = glob.glob(os.path.join(robot_dir, "urdf", "*.urdf"))
            mesh_dirs = [
                os.path.join(robot_dir, "meshesColored"),
                os.path.join(robot_dir, "meshes"),
            ]

            loaded_id = None

            # 优先尝试 URDF
            if urdf_list:
                urdf = urdf_list[0]
                self.logger.info(f"🔧  尝试通过 URDF 加载机器人模型: {urdf}")
                try:
                    # Articulated object manager (不同版本 API 可能不同)
                    if hasattr(self.sim, "get_articulated_object_manager"):
                        aom = self.sim.get_articulated_object_manager()
                        # 常见方法名尝试
                        for fn in [
                            "add_articulated_object_from_urdf",
                            "load_articulated_object_from_urdf",
                            "load_urdf",
                        ]:
                            if hasattr(aom, fn):
                                try:
                                    loaded_id = getattr(aom, fn)(urdf)
                                    break
                                except Exception:
                                    continue

                    # 尝试 simulator 级别的加载接口
                    if loaded_id is None and hasattr(self.sim, "add_object_from_file"):
                        try:
                            loaded_id = self.sim.add_object_from_file(urdf)
                        except Exception:
                            pass

                except Exception as e:
                    self.logger.debug(f"⚠️ URDF 加载尝试失败: {e}")

            # 若 URDF 未成功，尝试加载 meshes 中的第一个 mesh
            if loaded_id is None:
                for mesh_dir in mesh_dirs:
                    if not os.path.isdir(mesh_dir):
                        continue
                    meshes = glob.glob(
                        os.path.join(mesh_dir, "**", "*.glb"), recursive=True
                    )
                    meshes += glob.glob(
                        os.path.join(mesh_dir, "**", "*.gltf"), recursive=True
                    )
                    meshes += glob.glob(
                        os.path.join(mesh_dir, "**", "*.obj"), recursive=True
                    )
                    if not meshes:
                        continue
                    mesh_file = meshes[0]
                    self.logger.info(f"🔧  尝试通过 mesh 加载机器人模型: {mesh_file}")
                    try:
                        if hasattr(self.sim, "add_object_from_file"):
                            loaded_id = self.sim.add_object_from_file(mesh_file)
                            break
                        # 其他管理器尝试
                        if hasattr(self.sim, "get_rigid_object_manager"):
                            rom = self.sim.get_rigid_object_manager()
                            if hasattr(rom, "load_object"):
                                loaded_id = rom.load_object(mesh_file)
                                break
                    except Exception as e:
                        self.logger.debug(f"⚠️ mesh 加载尝试失败: {e}")

            if loaded_id is not None:
                try:
                    # 记录 ID 以供上层使用（例如渲染或定位）
                    self.robot_obj_id = loaded_id
                    self.logger.info(f"✅ 机器狗模型已加载, id={loaded_id}")
                    return True
                except Exception:
                    self.logger.info("✅ 机器狗模型加载成功 (无法获取 id)")
                    return True

            self.logger.info(
                "ℹ️ 未能通过自动方式加载机器人模型 (请检查文件或 API 兼容性)"
            )
            return False
        except Exception as e:
            self.logger.debug(f"⚠️ 尝试加载机器人模型时出现异常: {e}")
            return False

    @staticmethod
    def _to_wxyz(q_in) -> np.ndarray:
        """将四元数转换为 numpy.array([w, x, y, z])，兼容 magnum/habitat_sim 返回值。"""
        if q_in is None:
            raise ValueError("四元数为空")
        if isinstance(q_in, np.ndarray):
            arr = q_in.astype(float).reshape(-1)
            if arr.shape[0] == 4:
                return arr

        # objects with w/x/y/z
        try:
            w = getattr(q_in, "w", None)
            x = getattr(q_in, "x", None)
            y = getattr(q_in, "y", None)
            z = getattr(q_in, "z", None)
            if None not in (w, x, y, z):
                return np.array([w, x, y, z], dtype=float)
        except Exception:
            pass

        # Magnum quaternion: scalar()/vector() or scalar/vector
        try:
            scalar = getattr(q_in, "scalar", None)
            vector = getattr(q_in, "vector", None)
            s = scalar() if callable(scalar) else scalar
            v = vector() if callable(vector) else vector
            if s is not None and v is not None and len(v) >= 3:
                return np.array([s, v[0], v[1], v[2]], dtype=float)
        except Exception:
            pass

        # Iterable
        try:
            seq = list(q_in)
            if len(seq) == 4:
                return np.array(seq, dtype=float)
        except Exception:
            pass

        raise ValueError("无法解析四元数")

    @staticmethod
    def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = HabitatSimEnvironment._to_wxyz(a)
        b = HabitatSimEnvironment._to_wxyz(b)
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            dtype=float,
        )

    @staticmethod
    def _quat_conjugate(q: np.ndarray) -> np.ndarray:
        q = HabitatSimEnvironment._to_wxyz(q)
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)

    @staticmethod
    def _quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        q = HabitatSimEnvironment._to_wxyz(q)
        v = np.array(v, dtype=float).reshape(3)
        qv = np.array([0.0, v[0], v[1], v[2]], dtype=float)
        return HabitatSimEnvironment._quat_mul(
            HabitatSimEnvironment._quat_mul(q, qv),
            HabitatSimEnvironment._quat_conjugate(q),
        )[1:]

    @staticmethod
    def _set_robot_rotation(robot_obj, rotation_wxyz: np.ndarray) -> None:
        """安全设置机器人 rotation：将 wxyz 转为 Magnum Quaternion。"""
        import magnum as mn

        q = HabitatSimEnvironment._to_wxyz(rotation_wxyz)
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        robot_obj.rotation = mn.Quaternion(mn.Vector3(x, y, z), w)

    def _init_robot_follow_state(self) -> None:
        """初始化机器人初始朝向与跟随状态（参考 enhanced viewer：相对旋转跟随）。"""
        if self.robot_obj is None:
            return

        robot_cfg = (
            self.env_config.get("robot", {})
            if isinstance(self.env_config, dict)
            else {}
        )

        agent_rot = getattr(self.agent.scene_node, "rotation", None)
        if agent_rot is None:
            return
        agent_q = self._to_wxyz(agent_rot)
        self._agent_initial_rotation_wxyz = agent_q.copy()

        # 1) 明确指定 initial_rotation_wxyz
        init_rot_cfg = robot_cfg.get("initial_rotation_wxyz", None)
        if isinstance(init_rot_cfg, (list, tuple)) and len(init_rot_cfg) == 4:
            robot_init = self._to_wxyz(np.array(init_rot_cfg, dtype=float))
            self._robot_initial_rotation_wxyz = robot_init.copy()
            self._set_robot_rotation(self.robot_obj, robot_init)
            return

        # 2) 指定 initial_yaw_deg
        init_yaw_deg = robot_cfg.get("initial_yaw_deg", None)
        if init_yaw_deg is not None:
            try:
                from habitat_sim.utils.common import quat_from_angle_axis

                yaw = float(init_yaw_deg)
                q_yaw = quat_from_angle_axis(np.deg2rad(yaw), np.array([0.0, 1.0, 0.0]))
                robot_init = self._to_wxyz(q_yaw)
                self._robot_initial_rotation_wxyz = robot_init.copy()
                self._set_robot_rotation(self.robot_obj, robot_init)
                return
            except Exception:
                pass

        # 3) yaw_align_deg / 自动对齐：得到 correction，然后 robot_init = correction * agent_init
        try:
            from habitat_sim.utils.common import quat_from_angle_axis

            cfg_align = robot_cfg.get("yaw_align_deg", None)
            if cfg_align is not None:
                self._robot_align_deg = float(cfg_align)
                corr = quat_from_angle_axis(
                    np.deg2rad(self._robot_align_deg),
                    np.array([0.0, 1.0, 0.0]),
                )
                self._robot_rotation_correction_wxyz = self._to_wxyz(corr)
            else:
                local_forward = np.array([0.0, 0.0, -1.0])
                try:
                    robot_q0 = self._to_wxyz(getattr(self.robot_obj, "rotation"))
                except Exception:
                    robot_q0 = agent_q.copy()

                agent_fwd = self._quat_rotate_vector(agent_q, local_forward)
                robot_fwd = self._quat_rotate_vector(robot_q0, local_forward)
                a = np.array([agent_fwd[0], agent_fwd[2]])
                r = np.array([robot_fwd[0], robot_fwd[2]])
                a = a / (np.linalg.norm(a) + 1e-8)
                r = r / (np.linalg.norm(r) + 1e-8)
                det = r[0] * a[1] - r[1] * a[0]
                dot = float(np.clip(np.dot(r, a), -1.0, 1.0))
                yaw_delta = float(np.arctan2(det, dot))
                corr = quat_from_angle_axis(yaw_delta, np.array([0.0, 1.0, 0.0]))
                self._robot_rotation_correction_wxyz = self._to_wxyz(corr)

            if self._robot_rotation_correction_wxyz is None:
                self._robot_rotation_correction_wxyz = np.array([1.0, 0.0, 0.0, 0.0])

            robot_init = self._quat_mul(self._robot_rotation_correction_wxyz, agent_q)
            self._robot_initial_rotation_wxyz = robot_init.copy()
            self._set_robot_rotation(self.robot_obj, robot_init)
        except Exception:
            # 回退：直接使用 agent 初始朝向
            self._robot_initial_rotation_wxyz = agent_q.copy()
            try:
                self._set_robot_rotation(self.robot_obj, agent_q)
            except Exception:
                pass

    def _sync_robot_follow_agent(self) -> None:
        """让机器狗跟随 Agent（参考 enhanced viewer，best-effort）。

        关键点：在渲染传感器观测前更新 robot 位姿，保证视频/可视化中能看到机器人跟随。
        """
        try:
            if self.robot_obj is None and self.robot_obj_id is None:
                return

            robot_cfg = (
                self.env_config.get("robot", {})
                if isinstance(self.env_config, dict)
                else {}
            )
            height_offset = float(robot_cfg.get("height_offset", 0.6))

            agent_pos = getattr(self.agent.scene_node, "translation", None)
            agent_rot = getattr(self.agent.scene_node, "rotation", None)
            if agent_pos is None:
                agent_pos = self._get_agent_position()

            pos = np.array(agent_pos, dtype=float) + np.array(
                [0.0, height_offset, 0.0], dtype=float
            )

            # 优先：robot_obj 具备 translation/rotation 属性
            if self.robot_obj is not None and hasattr(self.robot_obj, "translation"):
                try:
                    self.robot_obj.translation = pos
                except Exception:
                    pass
                try:
                    if agent_rot is not None:
                        # 与 enhanced viewer 一致：使用相对旋转，让机器人身体朝向跟随 agent body
                        if (
                            self._robot_initial_rotation_wxyz is not None
                            and self._agent_initial_rotation_wxyz is not None
                        ):
                            cur_agent = self._to_wxyz(agent_rot)
                            delta = self._quat_mul(
                                cur_agent,
                                self._quat_conjugate(self._agent_initial_rotation_wxyz),
                            )
                            corrected = self._quat_mul(
                                delta, self._robot_initial_rotation_wxyz
                            )
                            self._set_robot_rotation(self.robot_obj, corrected)
                        else:
                            self.robot_obj.rotation = agent_rot
                except Exception:
                    pass
                return

            # 兼容：仅有 id/handle 的情况
            rid = self.robot_obj_id
            if rid is None:
                return

            moved = False
            try:
                if hasattr(self.sim, "get_articulated_object_manager"):
                    aom = self.sim.get_articulated_object_manager()
                    if hasattr(aom, "set_root_state"):
                        aom.set_root_state(rid, pos, agent_rot)
                        moved = True
            except Exception:
                moved = False

            if not moved:
                try:
                    if hasattr(self.sim, "set_object_transformation"):
                        self.sim.set_object_transformation(rid, pos, agent_rot)
                except Exception:
                    pass
        except Exception:
            return

    def reset(
        self, start_position: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """重置环境"""
        self.sim.reset()

        if start_position is not None:
            self.set_agent_state(start_position, np.array([0, 0, 0, 1]))

        # reset 后重新记录初始旋转基准（参考 enhanced viewer）
        try:
            self._init_robot_follow_state()
        except Exception:
            pass

        # reset 后先同步机器人到起点（避免首帧不可见/不同步）
        try:
            self._sync_robot_follow_agent()
        except Exception:
            pass

        obs = self.get_observations()

        # 显示可视化
        self._display_visualization(obs)

        return obs

    def step(self, action: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """执行一步动作"""
        try:
            self.agent.act(action)

            # 先同步机器人，再渲染传感器观测（保证“看到机器人跟随”）
            try:
                self._sync_robot_follow_agent()
            except Exception:
                pass

            # 2. 应用腿部动画（如果启用且机器人在移动）
            is_moving = action in ["move_forward", "move_backward"]
            if is_moving:
                self._apply_leg_animation()
            # else:
            #     self._reset_leg_pose()
                
            obs = self.sim.get_sensor_observations()

            info = {
                "agent_pos": self._get_agent_position(),
                "distance_to_goal": 0.0,  # 由导航系统计算
                "navmesh_loaded": bool(getattr(self.pathfinder, "is_loaded", False)),
                "navigable_area": float(
                    getattr(self.pathfinder, "navigable_area", 0.0) or 0.0
                ),
                "robot_loaded": bool(
                    (self.robot_obj is not None) or (self.robot_obj_id is not None)
                ),
            }

            # 机器人跟随已在渲染前处理，这里不再重复移动

            # 处理观测并显示可视化
            processed_obs = self._process_observations(obs)
            self._display_visualization(processed_obs)

            return processed_obs, info

        except Exception as e:
            self.logger.error(f"❌ 执行动作失败: {e}")
            return self.get_observations(), {"error": str(e)}

    def _apply_leg_animation(self):
        """应用腿部动画到机器人关节"""
        if not self._leg_animation_data or self.robot_obj is None:
            return
            
        try:
            # 获取当前帧（循环播放）
            num_frames = len(self._leg_animation_data)
            if num_frames == 0:
                return
                
            frame_idx = int(self._animation_frame % num_frames)
            joint_angles = self._leg_animation_data[frame_idx]
            
            # ===== 关键修正：Spot机器人的腿部关节是 8-19 =====
            if hasattr(self.robot_obj, 'joint_positions'):
                current_pos = self.robot_obj.joint_positions.copy()
                
                # 验证关节数量
                if len(current_pos) < 20:
                    self.logger.warning(f"⚠️  关节数量不足: {len(current_pos)}, 需要至少20个")
                    return
                
                if len(joint_angles) != 12:
                    self.logger.warning(f"⚠️  动画数据异常: {len(joint_angles)}个关节，需要12个")
                    return
                
                # 将12个腿部关节角度应用到关节 8-19
                current_pos[8:20] = joint_angles
                self.robot_obj.joint_positions = current_pos
                
                # 调试日志（首次执行时打印）
                if self._animation_frame == 0:
                    self.logger.info(f"✅ 应用腿部动画: 关节8-19 = {joint_angles[:3]}...")
                
            # 方法2: 如果有单独的腿部关节访问接口（通常在RearrangeSim中）
            elif hasattr(self.robot_obj, 'leg_joint_pos'):
                self.robot_obj.leg_joint_pos = joint_angles
                
            # 更新帧索引
            play_speed = self._animation_config.get("play_i_perframe", 1)
            self._animation_frame += play_speed
            
        except Exception as e:
            self.logger.debug(f"应用动画失败: {e}")

    def _reset_leg_pose(self):
        """重置腿部到初始姿态（停止时）"""
        if self.robot_obj is None:
            return
            
        try:
            robot_cfg = self.env_config.get("robot", {})
            
            # 从 spot_robot.py 获取的默认初始姿态
            default_leg_init = [
                0.0, 0.7, -1.5,  # 前左腿
                0.0, 0.7, -1.5,  # 前右腿
                0.0, 0.7, -1.5,  # 后左腿
                0.0, 0.7, -1.5,  # 后右腿
            ]
            
            init_pose = robot_cfg.get("leg_init_params", default_leg_init)
            
            if hasattr(self.robot_obj, 'joint_positions'):
                current_pos = self.robot_obj.joint_positions.copy()
                if len(current_pos) >= 20 and len(init_pose) == 12:
                    # 重置关节 8-19
                    current_pos[8:20] = init_pose
                    self.robot_obj.joint_positions = current_pos
                    self._animation_frame = 0
                    self.logger.debug("腿部姿态已重置")
                    
            elif hasattr(self.robot_obj, 'leg_joint_pos'):
                self.robot_obj.leg_joint_pos = init_pose
                self._animation_frame = 0
                
        except Exception as e:
            self.logger.debug(f"重置腿部姿态失败: {e}")
        
    def _display_visualization(self, obs: Dict[str, np.ndarray]):
        """
        显示Habitat可视化窗口

        Args:
            obs: 观测字典，包含rgb_front和rgb_back
        """
        if not self.enable_visualization:
            return

        try:
            import cv2

            # 获取前置和后置RGB图像
            rgb_front = obs.get("rgb_front", np.zeros((720, 1280, 3), dtype=np.uint8))
            rgb_back = obs.get("rgb_back", np.zeros((480, 640, 3), dtype=np.uint8))

            # 调整后置图像大小以匹配前置图像宽度
            h_back, w_back = rgb_back.shape[:2]
            h_front, w_front = rgb_front.shape[:2]

            # 按比例缩放后置图像
            scale = w_front / w_back
            new_h = int(h_back * scale)
            rgb_back_resized = cv2.resize(rgb_back, (w_front, new_h))

            # 垂直拼接前置和后置图像
            combined = np.vstack([rgb_front, rgb_back_resized])

            # 添加文本信息
            agent_pos = obs.get("agent_pos", np.array([0, 0, 0]))
            pos_text = f"Position: ({agent_pos[0]:.2f}, {agent_pos[1]:.2f}, {agent_pos[2]:.2f})"
            cv2.putText(
                combined,
                pos_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                combined,
                "Front Camera",
                (10, h_front - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                combined,
                "Back Camera (Overhead)",
                (10, h_front + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # 显示窗口
            cv2.imshow(self.vis_window_name, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)  # 1ms延迟，允许窗口刷新

        except Exception as e:
            self.logger.debug(f"可视化显示失败: {e}")

    def get_observations(self) -> Dict[str, np.ndarray]:
        """获取当前观测"""
        try:
            obs = self.sim.get_sensor_observations()
            return self._process_observations(obs)
        except Exception as e:
            self.logger.error(f"❌ 获取观测失败: {e}")
            return {
                "rgb_front": np.zeros((720, 1280, 3), dtype=np.uint8),
                "depth_front": np.zeros((720, 1280), dtype=np.float32),
                "rgb_back": np.zeros((480, 640, 3), dtype=np.uint8),
            }

    def _process_observations(self, obs: Dict) -> Dict[str, np.ndarray]:
        """处理原始观测（将Habitat的观测转换为标准格式）

        【重要】此方法返回原始RGB图像，不做任何处理。
        Habitat返回RGBA格式，此方法仅移除Alpha通道以获得RGB。
        """
        processed = {}

        # 读取配置中的 UUID（增强鲁棒性：即使用户改了 uuid，也能正确映射）
        sensors_cfg = (
            self.env_config.get("sensors", {})
            if isinstance(self.env_config, dict)
            else {}
        )
        front_cfg = (
            sensors_cfg.get("front_camera", {}) if isinstance(sensors_cfg, dict) else {}
        )
        back_cfg = (
            sensors_cfg.get("back_camera", {}) if isinstance(sensors_cfg, dict) else {}
        )
        depth_cfg = (
            sensors_cfg.get("front_depth", {}) if isinstance(sensors_cfg, dict) else {}
        )
        front_uuid = front_cfg.get("uuid", "front_rgb")
        back_uuid = back_cfg.get("uuid", "top_down_view")
        depth_uuid = depth_cfg.get("uuid", "front_depth")

        # 前置摄像头 (Habitat返回RGBA或RGB格式)
        if front_uuid in obs:
            img = obs[front_uuid]
            # 如果是RGBA (4通道)，仅移除Alpha通道，返回原始RGB数据
            if len(img.shape) == 3 and img.shape[2] == 4:
                # 直接提取前三个通道（原始RGB数据，不做任何其他处理）
                processed["rgb_front"] = img[:, :, :3].copy()
            else:
                # 已经是RGB格式，直接使用
                processed["rgb_front"] = img.copy()
        else:
            processed["rgb_front"] = np.zeros((720, 1280, 3), dtype=np.uint8)

        # 前置深度（可选）。Habitat 通常返回 float32 的 (H,W) 或 (H,W,1)
        if bool(depth_cfg.get("enabled", False)) and depth_uuid in obs:
            d = obs[depth_uuid]
            try:
                if isinstance(d, np.ndarray):
                    if d.ndim == 3 and d.shape[2] == 1:
                        d = d[:, :, 0]
                    processed["depth_front"] = d.astype(np.float32, copy=False)
                else:
                    processed["depth_front"] = np.zeros((720, 1280), dtype=np.float32)
            except Exception:
                processed["depth_front"] = np.zeros((720, 1280), dtype=np.float32)
        else:
            # 保持键存在但为0，有助于上层统一处理（VLM是否使用由启用开关控制）
            processed["depth_front"] = np.zeros((720, 1280), dtype=np.float32)

        # 后置摄像头 - 俯视图 (Habitat返回RGBA或RGB格式)
        if back_uuid in obs:
            img = obs[back_uuid]
            # 如果是RGBA (4通道)，仅移除Alpha通道，返回原始RGB数据
            if len(img.shape) == 3 and img.shape[2] == 4:
                # 直接提取前三个通道（原始RGB数据，不做任何其他处理）
                processed["rgb_back"] = img[:, :, :3].copy()
            else:
                # 已经是RGB格式，直接使用
                processed["rgb_back"] = img.copy()
        else:
            processed["rgb_back"] = np.zeros((480, 640, 3), dtype=np.uint8)

        # 添加Agent状态信息
        processed["agent_pos"] = self._get_agent_position()
        processed["agent_rot"] = self._get_agent_rotation()

        # NavMesh / robot 状态（用于上层可视化/日志）
        try:
            processed["navmesh_loaded"] = np.array(
                [1 if bool(getattr(self.pathfinder, "is_loaded", False)) else 0],
                dtype=np.uint8,
            )
        except Exception:
            pass
        try:
            processed["navigable_area"] = np.array(
                [float(getattr(self.pathfinder, "navigable_area", 0.0) or 0.0)],
                dtype=np.float32,
            )
        except Exception:
            pass
        try:
            processed["navmesh_visualization"] = np.array(
                [1 if bool(getattr(self.sim, "navmesh_visualization", False)) else 0],
                dtype=np.uint8,
            )
        except Exception:
            pass
        try:
            processed["robot_loaded"] = np.array(
                [
                    1
                    if bool(
                        (self.robot_obj is not None) or (self.robot_obj_id is not None)
                    )
                    else 0
                ],
                dtype=np.uint8,
            )
        except Exception:
            pass

        # 亮度增益（与 enhanced viewer 一致的更“明亮”观感；用于显示与 VLM 输入）
        try:
            rendering_cfg = (
                self.env_config.get("rendering", {})
                if isinstance(self.env_config, dict)
                else {}
            )
            enabled = bool(rendering_cfg.get("enabled", True))
            brightness = float(rendering_cfg.get("brightness_factor", 1.0))
            if enabled and brightness and abs(brightness - 1.0) > 1e-3:
                for key in ("rgb_front", "rgb_back"):
                    img = processed.get(key)
                    if (
                        isinstance(img, np.ndarray)
                        and img.ndim == 3
                        and img.dtype == np.uint8
                    ):
                        processed[key] = np.clip(
                            img.astype(np.float32) * brightness, 0, 255
                        ).astype(np.uint8)
        except Exception:
            pass

        return processed

    def _ensure_navmesh(self):
        """加载或计算 NavMesh 可通行区域（参考 enhanced viewer）。"""
        import habitat_sim

        scene_cfg = (
            self.env_config.get("scene", {})
            if isinstance(self.env_config, dict)
            else {}
        )
        navmesh_path = scene_cfg.get("navmesh_path")

        # 1) 若提供预编译 navmesh，则优先加载
        if navmesh_path and os.path.exists(str(navmesh_path)):
            try:
                ok = self.pathfinder.load_nav_mesh(str(navmesh_path))
                if ok and self.pathfinder.is_loaded:
                    self.logger.info(f"✅ NavMesh 已加载: {navmesh_path}")
                    return
            except Exception as e:
                self.logger.warning(f"⚠️  NavMesh 加载失败，转为重新计算: {e}")

        # 2) 否则重新计算
        self.logger.info("🗺️  开始计算NavMesh可通行区域...")

        navmesh_cfg = (
            self.env_config.get("navmesh", {})
            if isinstance(self.env_config, dict)
            else {}
        )
        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()
        try:
            agent_cfg = self.sim.config.agents[0]
            settings.agent_height = agent_cfg.height
            settings.agent_radius = agent_cfg.radius
        except Exception:
            # 回退到配置
            settings.agent_height = float(self.agent_config.get("height", 0.55))
            settings.agent_radius = float(self.agent_config.get("radius", 0.28))
        settings.include_static_objects = bool(
            navmesh_cfg.get("include_static_objects", True)
        )

        success = self.sim.recompute_navmesh(self.pathfinder, settings)
        if success and self.pathfinder.is_loaded:
            self.logger.info("✅ NavMesh计算成功")
            try:
                self.logger.info(
                    f"   可通行面积: {self.pathfinder.navigable_area:.2f} m²"
                )
            except Exception:
                pass
            # 默认关闭 navmesh 渲染覆盖（可通过外部配置/交互打开）
            try:
                self.sim.navmesh_visualization = bool(
                    navmesh_cfg.get("visualization_default", False)
                )
            except Exception:
                pass
        else:
            self.logger.warning("⚠️  NavMesh计算失败，但继续运行")

    def get_agent_state(self) -> Dict[str, np.ndarray]:
        """获取Agent状态"""
        return {
            "position": self._get_agent_position(),
            "rotation": self._get_agent_rotation(),
        }

    def set_agent_state(self, position: np.ndarray, rotation: np.ndarray) -> bool:
        """设置Agent状态"""
        try:
            self.agent.set_state(position, rotation)
            return True
        except Exception as e:
            self.logger.warning(f"⚠️  设置Agent状态失败: {e}")
            return False

    def _get_agent_position(self) -> np.ndarray:
        """获取Agent位置"""
        try:
            return self.agent.get_state().position.astype(np.float32)
        except Exception as e:
            self.logger.debug(f"获取 Agent 位置失败: {e}")
            return np.array([0, 0, 0], dtype=np.float32)

    def _get_agent_rotation(self) -> np.ndarray:
        """获取Agent旋转 (四元数)"""
        try:
            return self.agent.get_state().rotation.astype(np.float32)
        except Exception as e:
            self.logger.debug(f"获取 Agent 旋转失败: {e}")
            return np.array([0, 0, 0, 1], dtype=np.float32)

    def compute_shortest_path(
        self, start: np.ndarray, end: np.ndarray
    ) -> Tuple[List[np.ndarray], float]:
        """计算最短路径"""
        try:
            from habitat_sim.utils.common import compute_euclid_distance

            path = habitat_sim.ShortestPath()
            path.requested_start = start
            path.requested_end = end

            self.pathfinder.find_path(path)

            if path.points:
                points = [np.array(p) for p in path.points]
                distance = path.geodesic_distance
                return points, distance
            else:
                return [], 0.0

        except Exception as e:
            self.logger.warning(f"⚠️  路径规划失败: {e}")
            return [], 0.0

    def close(self):
        """关闭环境"""
        # 关闭可视化窗口
        if self.enable_visualization:
            try:
                import cv2

                cv2.destroyWindow(self.vis_window_name)
                cv2.waitKey(1)
            except Exception:
                pass

        # 关闭模拟器
        if self.sim:
            self.sim.close()
            self.logger.info("✅ Habitat-sim 环境已关闭")


# ============================================================================
# 第四部分: Habitat-lab 集成
# ============================================================================


class HabitatLabTaskIntegration:
    """轻量的 Habitat-lab 集成包装（best-effort 实现）。

    该类为可选集成：当用户安装了 habitat-lab 时会尝试初始化，
    否则本类保持不可用状态但不会抛出异常。
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.env = None
        self.task = None
        self.measurements = None
        self.available = False
        self._try_init_habitatlab()

    def _try_init_habitatlab(self):
        try:
            import habitat  # type: ignore

            self.available = True
            self.logger.info("✅ Habitat-lab 可用，已启用集成接口（轻量级）")
        except Exception:
            self.available = False
            self.logger.info("ℹ️  Habitat-lab 不可用，跳过 Habitat-lab 集成")

    def load_task_config(self, config_file: str) -> bool:
        """加载任务配置（轻量实现）。

        此函数会尝试读取配置并执行对机器人模型的最佳尝试加载。
        """
        if not self.available:
            self.logger.warning("⚠️  Habitat-lab 不可用，无法加载任务配置")
            return False

        try:
            self.logger.info(f"📋 加载 Habitat-lab 任务配置: {config_file}")
            # best-effort 加载 robot 模型用于可视化
            try:
                self._try_load_robot_model()
            except Exception as e:
                self.logger.debug(f"ℹ️  机器人模型加载尝试结束: {e}")

            return True
        except Exception as e:
            self.logger.error(f"❌ 加载任务配置失败: {e}")
            return False

    def _try_load_robot_model(self) -> bool:
        """尝试从 `data/robots/hab_spot_arm` 加载 URDF / meshes（轻量实现）。

        返回是否成功加载至少一个可视对象。
        """
        try:
            import glob
            import os

            # 优先从外部配置尝试读取路径，否则使用项目内默认路径
            robot_dir = None
            try:
                # 在轻量集成中我们不import外部get_global_config，优先检测常见位置
                robot_dir = os.path.join(os.getcwd(), "data", "robots", "hab_spot_arm")
            except Exception:
                robot_dir = os.path.join(os.getcwd(), "data", "robots", "hab_spot_arm")

            if not os.path.isdir(robot_dir):
                self.logger.info(f"ℹ️  未找到机器人模型目录: {robot_dir}")
                return False

            # 仅记录发现情况，实际加载在 HabitatSimEnvironment 中执行更可靠
            urdfs = glob.glob(os.path.join(robot_dir, "urdf", "*.urdf"))
            meshes = []
            for d in [
                os.path.join(robot_dir, "meshesColored"),
                os.path.join(robot_dir, "meshes"),
            ]:
                if os.path.isdir(d):
                    meshes += glob.glob(os.path.join(d, "**", "*.glb"), recursive=True)
                    meshes += glob.glob(os.path.join(d, "**", "*.gltf"), recursive=True)

            if urdfs or meshes:
                self.logger.info(
                    "✅ 发现机器人模型文件，建议在 Habitat-sim 中加载以获得最佳兼容性"
                )
                return True
            else:
                self.logger.info("ℹ️  未发现机器人 URDF 或 meshes 文件")
                return False
        except Exception as e:
            self.logger.debug(f"⚠️ 尝试检测机器人模型时出错: {e}")
            return False

    def get_evaluation_metrics(self) -> Dict[str, float]:
        if not self.available or not self.env:
            return {}
        try:
            metrics = self.env.get_metrics()
            return metrics
        except Exception as e:
            self.logger.warning(f"⚠️ 获取 Habitat-lab 指标失败: {e}")
            return {}

    def close(self):
        if self.env:
            try:
                self.env.close()
            except Exception:
                pass
        self.logger.info("✅ Habitat-lab 集成已关闭（如有）")

    def get_evaluation_metrics(self) -> Dict[str, float]:
        """获取评估指标

        Returns:
            指标字典 (如果 habitat-lab 可用)
        """
        if not self.available or not self.env:
            return {}

        try:
            metrics = self.env.get_metrics()
            return metrics
        except Exception as e:
            self.logger.warning(f"⚠️  获取指标失败: {e}")
            return {}

    def close(self):
        """关闭 Habitat-lab 环境"""
        if self.env:
            self.env.close()
            self.logger.info("✅ Habitat-lab 环境已关闭")


# ============================================================================
# 第五部分: 统一环境工厂
# ============================================================================


class EnvironmentFactory:
    """环境创建工厂

    根据配置和已安装框架自动创建合适的环境
    """

    def __init__(self, logger: logging.Logger):
        """初始化工厂

        Args:
            logger: 日志记录器
        """
        self.logger = logger
        self.version_info = HabitatVersionInfo()

    def create_environment(
        self, config: Dict[str, Any]
    ) -> Tuple[UnifiedEnvironment, Optional[HabitatLabTaskIntegration]]:
        """创建环境

        Args:
            config: 环境配置字典，应包含:
                - scene_path: 场景路径
                - agent_config: Agent配置
                - sim_config: 模拟器配置

        Returns:
            (环境对象, Habitat-lab集成对象或None)
        """
        self.logger.info(self.version_info.get_framework_summary())

        # 如果配置中显式要求使用 Habitat-lab，则尝试创建；否则优先使用 Habitat-sim
        force_lab = bool(config.get("use_habitat_lab", False))

        if self.version_info.lab_available and force_lab:
            self.logger.info("🔄 使用 Habitat-lab 集成环境 (由配置强制)...")
            try:
                env = self._create_lab_environment(config)
                lab_integration = HabitatLabTaskIntegration(self.logger)
                return env, lab_integration
            except Exception:
                # 如果创建失败则降级到 Habitat-sim
                self.logger.info("🔄 Habitat-lab 创建失败，降级到 Habitat-sim...")

        # 缺省或未强制使用 Habitat-lab 时，优先使用 Habitat-sim
        self.logger.info("🔄 使用 Habitat-sim 低级环境...")

        # 安全探测：某些EGL/GL错误会直接 exit(1)/abort，无法被 try/except 捕获。
        # 这里先在子进程中探测能否创建 Simulator，失败则回退到 MockEnvironment，避免整个pytest/主进程被杀。
        enable_visualization = bool(config.get("enable_visualization", False))
        safe_probe = bool(config.get("safe_habitat_probe", True))
        if safe_probe and not enable_visualization:
            if not self._probe_habitat_sim_safe(config):
                self.logger.warning(
                    "⚠️  Habitat-sim 探测失败（可能是EGL/GPU设备问题），回退到 MockEnvironment"
                )
                return MockEnvironment(self.logger, config=config), None

        env = HabitatSimEnvironment(
            scene_path=config.get("scene_path", ""),
            agent_config=config.get("agent_config", {}),
            sim_config=config.get("sim_config", {}),
            logger=self.logger,
            env_config=config.get("env_config", {}),  # 传递环境配置
            enable_visualization=enable_visualization,  # 传递可视化参数
        )
        return env, None

    def _probe_habitat_sim_safe(self, config: Dict[str, Any]) -> bool:
        """在子进程中探测Habitat-sim是否能创建Simulator。

        返回 False 表示不安全（子进程非零退出/异常），此时主进程应避免直接创建Simulator。
        """
        scene_path = str(config.get("scene_path", "") or "")
        if not scene_path:
            return False

        # 探测必须尽量“触发渲染后端初始化”，否则可能出现：Simulator可建，但一旦创建相机传感器就 exit(1)。
        # 因此这里会在子进程里创建一个最小 RGB CameraSensorSpec 并拉一次观测。
        # 注意：此代码可能触发 EGL 失败并直接退出，因此必须在子进程中运行。
        code = (
            "import os\n"
            "import habitat_sim\n"
            "scene=os.environ.get('HAB_SCENE','')\n"
            "sim_cfg=habitat_sim.SimulatorConfiguration()\n"
            "sim_cfg.scene_id=scene\n"
            "sim_cfg.enable_physics=False\n"
            "gpu=os.environ.get('HAB_GPU')\n"
            "if gpu not in (None,''): sim_cfg.gpu_device_id=int(gpu)\n"
            "sensor=habitat_sim.CameraSensorSpec()\n"
            "sensor.uuid='probe_rgb'\n"
            "sensor.sensor_type=habitat_sim.SensorType.COLOR\n"
            "sensor.sensor_subtype=habitat_sim.SensorSubType.PINHOLE\n"
            "sensor.resolution=[32,32]\n"
            "sensor.position=[0.0,0.0,0.0]\n"
            "agent_cfg=habitat_sim.agent.AgentConfiguration()\n"
            "agent_cfg.sensor_specifications=[sensor]\n"
            "cfg=habitat_sim.Configuration(sim_cfg,[agent_cfg])\n"
            "sim=habitat_sim.Simulator(cfg)\n"
            "sim.reset()\n"
            "_ = sim.get_sensor_observations()\n"
            "sim.close()\n"
            "print('OK')\n"
        )

        env = os.environ.copy()
        env["HAB_SCENE"] = scene_path
        # 传递 GPU 选择（与主进程逻辑一致）
        gpu_id = ""
        try:
            env_gpu = os.environ.get("HABITAT_SIM_GPU_DEVICE_ID")
            if env_gpu is not None and str(env_gpu).strip() != "":
                gpu_id = str(int(str(env_gpu).strip()))
            else:
                env_cfg = config.get("env_config", {})
                sim_section = (
                    env_cfg.get("simulator", {}) if isinstance(env_cfg, dict) else {}
                )
                if (
                    isinstance(sim_section, dict)
                    and sim_section.get("gpu_device_id") is not None
                ):
                    gpu_id = str(int(sim_section.get("gpu_device_id")))
        except Exception:
            gpu_id = ""
        env["HAB_GPU"] = gpu_id

        # 修复部分 conda 环境下 GLVND 默认走 mesa vendor 导致的 EGL/CUDA 映射失败。
        # 仅对 probe 子进程注入，不影响用户手工配置（若用户已设置则尊重用户设置）。
        try:
            if not env.get("__EGL_VENDOR_LIBRARY_FILENAMES"):
                nvidia_vendor = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
                if os.path.exists(nvidia_vendor):
                    env["__EGL_VENDOR_LIBRARY_FILENAMES"] = nvidia_vendor
            env.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

            # 优先使用系统 libGLdispatch，避免 conda GLVND/mesa 组合触发 OpenGL version 获取失败
            system_gldispatch = "/lib/x86_64-linux-gnu/libGLdispatch.so.0"
            if os.path.exists(system_gldispatch):
                preload = env.get("LD_PRELOAD", "").strip()
                if preload:
                    env["LD_PRELOAD"] = f"{system_gldispatch}:{preload}"
                else:
                    env["LD_PRELOAD"] = system_gldispatch

            # 系统库路径置前（不覆盖原值）
            sys_ld_prefix = "/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"
            old_ld = env.get("LD_LIBRARY_PATH", "").strip()
            env["LD_LIBRARY_PATH"] = (
                f"{sys_ld_prefix}:{old_ld}" if old_ld else sys_ld_prefix
            )
        except Exception:
            pass

        try:
            p = subprocess.run(
                [sys.executable, "-c", code],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=45,
            )
            if p.returncode != 0:
                self.logger.debug(
                    f"Habitat-sim probe failed rc={p.returncode}; stderr={p.stderr[-400:]}"
                )
                return False
            return "OK" in (p.stdout or "")
        except Exception as e:
            self.logger.debug(f"Habitat-sim probe exception: {e}")
            return False

    def _create_lab_environment(self, config: Dict[str, Any]) -> UnifiedEnvironment:
        """从 Habitat-lab 创建环境包装器"""
        try:
            import habitat
            from habitat import Env

            # 创建 Habitat-lab 环境
            # 返回一个包装了 habitat-lab Env 的 UnifiedEnvironment 适配器
            return HabitatLabEnvironmentAdapter(config=config, logger=self.logger)

        except Exception as e:
            self.logger.warning(f"⚠️  Habitat-lab 环境创建失败: {e}")
            self.logger.info("🔄 降级到 Habitat-sim...")

            return HabitatSimEnvironment(
                scene_path=config.get("scene_path", ""),
                agent_config=config.get("agent_config", {}),
                sim_config=config.get("sim_config", {}),
                logger=self.logger,
                env_config=config.get("env_config", {}),  # 传递环境配置
                enable_visualization=config.get("enable_visualization", False),
            )


# ============================================================================
# 第六部分: Habitat-lab 适配器 (可选)
# ============================================================================


class HabitatLabEnvironmentAdapter(UnifiedEnvironment):
    """Habitat-lab 环境适配器

    包装 habitat-lab Env，提供统一接口
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """初始化适配器

        Args:
            config: 环境配置
            logger: 日志记录器
        """
        super().__init__(logger)
        self.config = config

        try:
            import habitat

            # 尝试创建 Habitat-lab 环境
            # 具体实现取决于 habitat-lab 版本
            self.logger.info("ℹ️  Habitat-lab 环境适配器创建成功")

        except Exception as e:
            self.logger.error(f"❌ Habitat-lab 适配器初始化失败: {e}")
            raise

    def reset(
        self, start_position: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """重置环境"""
        raise NotImplementedError("Habitat-lab 适配器需要具体实现")

    def step(self, action: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """执行一步动作"""
        raise NotImplementedError("Habitat-lab 适配器需要具体实现")

    def get_observations(self) -> Dict[str, np.ndarray]:
        """获取观测"""
        raise NotImplementedError("Habitat-lab 适配器需要具体实现")

    def get_agent_state(self) -> Dict[str, np.ndarray]:
        """获取Agent状态"""
        raise NotImplementedError("Habitat-lab 适配器需要具体实现")

    def set_agent_state(self, position: np.ndarray, rotation: np.ndarray) -> bool:
        """设置Agent状态"""
        raise NotImplementedError("Habitat-lab 适配器需要具体实现")

    def compute_shortest_path(
        self, start: np.ndarray, end: np.ndarray
    ) -> Tuple[List[np.ndarray], float]:
        """计算最短路径"""
        raise NotImplementedError("Habitat-lab 适配器需要具体实现")

    def get_navigable_map(self, height: float = None) -> np.ndarray:
        """获取虚拟可通行地图"""
        # 返回一个简单的网格图
        map_size = 200
        dummy_map = np.ones((map_size, map_size), dtype=np.uint8) * 255  # 白色背景
        # 绘制一些障碍物
        cv2.rectangle(dummy_map, (50, 50), (80, 80), 0, -1)
        cv2.rectangle(dummy_map, (120, 100), (150, 150), 0, -1)
        return dummy_map

    def close(self):
        """关闭环境"""
        pass


# ============================================================================
# 虚拟环境Mock（用于测试和无Habitat fallback）
# ============================================================================


class MockEnvironment(UnifiedEnvironment):
    """虚拟环境Mock - 用于测试或Habitat不可用时的fallback"""

    def __init__(self, logger: logging.Logger, config: Optional[Dict] = None):
        """初始化虚拟环境

        Args:
            logger: 日志记录器
            config: 配置字典（可选）
        """
        super().__init__(logger)
        self.config = config or {}
        self.agent_position = np.array([0.0, 0.0, 0.0])
        self.agent_rotation = np.array([0.0, 0.0, 0.0, 1.0])  # 四元数 [x, y, z, w]
        self.agent_heading = 0.0  # 弧度
        self.step_size = 0.25  # 前进步长
        self.turn_angle = np.deg2rad(10)  # 转弯角度
        self.step_count = 0

        self.logger.info("✅ 虚拟环境Mock已创建（用于测试）")

    def reset(
        self, start_position: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """重置环境"""
        if start_position is not None:
            self.agent_position = np.array(start_position)
        else:
            self.agent_position = np.array([0.0, 0.0, 0.0])

        self.agent_heading = 0.0
        self.agent_rotation = np.array([0.0, 0.0, 0.0, 1.0])
        self.step_count = 0

        self.logger.info(f"虚拟环境已重置，起始位置: {self.agent_position}")
        return self.get_observations()

    def step(self, action: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """执行一步动作

        Args:
            action: 'move_forward', 'turn_left', 'turn_right', 'stop'

        Returns:
            (观测, 信息)
        """
        self.step_count += 1

        # 执行动作
        if action == "move_forward":
            # 向前移动
            dx = self.step_size * np.cos(self.agent_heading)
            dy = self.step_size * np.sin(self.agent_heading)
            self.agent_position[0] += dx
            self.agent_position[2] += dy  # Habitat使用z轴作为前进方向
            self.logger.debug(
                f"前进 → 位置: ({self.agent_position[0]:.2f}, {self.agent_position[2]:.2f})"
            )

        elif action == "move_backward":
            dx = -self.step_size * np.cos(self.agent_heading)
            dy = -self.step_size * np.sin(self.agent_heading)
            self.agent_position[0] += dx
            self.agent_position[2] += dy
            self.logger.debug(
                f"后退 → 位置: ({self.agent_position[0]:.2f}, {self.agent_position[2]:.2f})"
            )

        elif action == "turn_left":
            # 左转
            self.agent_heading += self.turn_angle
            self.logger.debug(f"左转 → 朝向: {np.rad2deg(self.agent_heading):.1f}°")

        elif action == "turn_right":
            # 右转
            self.agent_heading -= self.turn_angle
            self.logger.debug(f"右转 → 朝向: {np.rad2deg(self.agent_heading):.1f}°")

        elif action == "stop":
            self.logger.info("停止")
        else:
            self.logger.warning(f"未知动作: {action}")

        # 更新旋转（简化的四元数）
        self.agent_rotation = self._heading_to_quaternion(self.agent_heading)

        # 获取观测
        obs = self.get_observations()

        # 计算信息
        info = {
            "step": self.step_count,
            "position": self.agent_position.copy(),
            "heading": np.rad2deg(self.agent_heading),
            "distance_to_goal": 0.5,  # 模拟距离
            "collision": False,
        }

        return obs, info

    def get_observations(self) -> Dict[str, np.ndarray]:
        """获取模拟观测"""
        # 生成模拟RGB图像
        rgb_front = self._generate_mock_rgb((720, 1280, 3))
        rgb_back = self._generate_mock_rgb((480, 640, 3))

        # 模拟深度：默认给一个“远处为主”的平面（单位米）
        depth_front = np.full((720, 1280), 5.0, dtype=np.float32)

        return {
            "rgb_front": rgb_front,
            "depth_front": depth_front,
            "rgb_back": rgb_back,
            "agent_pos": self.agent_position.copy(),
            "agent_rot": self.agent_rotation.copy(),
        }

    def get_agent_state(self) -> Dict[str, np.ndarray]:
        """获取Agent状态"""
        return {
            "position": self.agent_position.copy(),
            "rotation": self.agent_rotation.copy(),
        }

    def set_agent_state(self, position: np.ndarray, rotation: np.ndarray) -> bool:
        """设置Agent状态"""
        self.agent_position = np.array(position)
        self.agent_rotation = np.array(rotation)
        self.agent_heading = self._quaternion_to_heading(rotation)
        return True

    def compute_shortest_path(
        self, start: np.ndarray, end: np.ndarray
    ) -> Tuple[List[np.ndarray], float]:
        """计算模拟最短路径（直线）"""
        distance = np.linalg.norm(end - start)
        path = [start, end]
        return path, distance

    def get_navigable_map(self, height: float = None) -> np.ndarray:
        """获取虚拟可通行地图"""
        # 返回一个简单的网格图
        map_size = 200
        dummy_map = np.ones((map_size, map_size), dtype=np.uint8) * 255  # 白色背景
        # 绘制一些障碍物
        try:
            import cv2  # type: ignore

            cv2.rectangle(dummy_map, (50, 50), (80, 80), 0, -1)
            cv2.rectangle(dummy_map, (120, 100), (150, 150), 0, -1)
        except Exception:
            # 无 OpenCV 时用 numpy 直接画矩形
            dummy_map[50:81, 50:81] = 0
            dummy_map[100:151, 120:151] = 0
        return dummy_map

    def close(self):
        """关闭虚拟环境"""
        self.logger.info("虚拟环境已关闭")

    # 辅助方法
    def _generate_mock_rgb(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """生成模拟RGB图像（渐变色背景+位置信息）"""
        # 创建渐变色背景 (使用numpy广播优化)
        h, w, _ = shape
        img = np.zeros(shape, dtype=np.uint8)

        # 生成坐标网格
        y, x = np.ogrid[:h, :w]

        # 向量化计算
        img[..., 0] = (128 + 64 * np.sin(y / 50)).astype(np.uint8)  # R
        img[..., 1] = (128 + 64 * np.cos(x / 50)).astype(np.uint8)  # G
        img[..., 2] = 100  # B

        # 添加位置文字（使用简单的方式）
        try:
            import cv2  # type: ignore

            text = f"Pos:({self.agent_position[0]:.1f},{self.agent_position[2]:.1f})"
            cv2.putText(
                img,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
        except Exception:
            # 无 OpenCV 时跳过文字绘制
            pass

        return img

    def _heading_to_quaternion(self, heading: float) -> np.ndarray:
        """朝向角转四元数（简化版，绕Y轴旋转）"""
        # 四元数: [x, y, z, w]
        return np.array([0.0, np.sin(heading / 2), 0.0, np.cos(heading / 2)])

    def _quaternion_to_heading(self, q: np.ndarray) -> float:
        """四元数转朝向角（简化版）"""
        # 简化：仅考虑绕Y轴的旋转
        return 2 * np.arctan2(q[1], q[3])
