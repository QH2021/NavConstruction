#!/usr/bin/env python3
"""
Habitat环境增强交互式测试 - 参考官方viewer实现
功能:
- NavMesh可通行区域计算与可视化
- 机器狗模型跟随Agent移动
- 键盘控制 (WASD)
- 实时统计信息显示

控制说明:
- W/S: 前进/后退
- A/D: 左转/右转
- N: 切换NavMesh可视化
- N+SHIFT: 重新计算NavMesh
- N+ALT: 在NavMesh上随机采样新位置
- ESC: 退出
"""

import sys
import os
from pathlib import Path

# 这个文件本质上是“交互式Viewer脚本”，不适合在自动化 pytest 中默认运行。
# 在CI/无EGL/GPU/无显示环境中导入它可能导致崩溃或挂起。
if "PYTEST_CURRENT_TEST" in os.environ and os.environ.get(
    "RUN_HABITAT_INTERACTIVE_ENHANCED", ""
).strip().lower() not in {"1", "true", "yes"}:
    try:
        import pytest  # type: ignore

        pytest.skip(
            "Skip interactive Habitat enhanced viewer module during automated pytest. "
            "Set RUN_HABITAT_INTERACTIVE_ENHANCED=1 to enable.",
            allow_module_level=True,
        )
    except Exception:
        # 如果pytest不可用（例如作为普通脚本运行），就不要跳过
        pass

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from typing import Dict, Any, Optional
import math
import numpy as np
import cv2

# 导入Habitat
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

# 导入项目模块
from src.config_loader import UnifiedConfigLoader


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """四元数乘法，输入可以是多种四元数类型，返回 numpy 数组 [w,x,y,z]"""
    a_arr = to_wxyz(a)
    b_arr = to_wxyz(b)
    aw, ax, ay, az = a_arr
    bw, bx, by, bz = b_arr
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    """四元数共轭，输入 [w, x, y, z]"""
    qa = to_wxyz(q)
    return np.array([qa[0], -qa[1], -qa[2], -qa[3]], dtype=float)


def _quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """用四元数 q ([w,x,y,z]) 旋转向量 v"""
    # v' = q * (0, v) * q_conj
    qa = to_wxyz(q)
    qv = np.concatenate([[0.0], v]).astype(float)
    return _quat_mul(_quat_mul(qa, qv), _quat_conjugate(qa))[1:]


def to_wxyz(q_in) -> np.ndarray:
    """将各种四元数表示转换为 numpy.array([w,x,y,z])

    支持输入类型: numpy.ndarray (len=4), objects with attributes w/x/y/z,
    objects with scalar()/vector() (Magnum), numpy.quaternion, or iterable of length 4.
    """
    import numpy as _np

    # numpy array
    if isinstance(q_in, _np.ndarray):
        arr = q_in.astype(float)
        if arr.shape[0] == 4:
            return arr

    # numpy quaternion (module 'quaternion') has attributes .w,.x,.y,.z
    try:
        w = getattr(q_in, "w", None)
        x = getattr(q_in, "x", None)
        y = getattr(q_in, "y", None)
        z = getattr(q_in, "z", None)
        if None not in (w, x, y, z):
            return _np.array([w, x, y, z], dtype=float)
    except Exception:
        pass

    # Magnum quaternion: scalar() and vector() or scalar and vector attributes
    try:
        scalar = getattr(q_in, "scalar", None)
        vector = getattr(q_in, "vector", None)
        if callable(scalar):
            s = scalar()
        else:
            s = scalar
        if callable(vector):
            v = vector()
        else:
            v = vector
        if v is not None and len(v) >= 3 and s is not None:
            return _np.array([s, v[0], v[1], v[2]], dtype=float)
    except Exception:
        pass

    # Try common attribute name permutations
    for attrs in ("w,x,y,z", "r,i,j,k", "s,x,y,z", "x,y,z,w"):
        try:
            parts = [getattr(q_in, a) for a in attrs.split(",")]
            return _np.array(parts, dtype=float)
        except Exception:
            continue

    # Iterable of length 4
    try:
        seq = list(q_in)
        if len(seq) == 4:
            return _np.array(seq, dtype=float)
    except Exception:
        pass

    raise ValueError("不能解析四元数对象")


def set_robot_rotation(robot_obj, rotation_quat: np.ndarray):
    """安全设置机器狗旋转的标准方法

    1. 创建标准四元数 (NumPy格式) [w,x,y,z]
    2. 自动转换为Magnum格式
    3. 安全赋值

    Args:
        robot_obj: 机器狗对象
        rotation_quat: 四元数 (numpy.ndarray [w,x,y,z] 或其他支持的格式)
    """
    try:
        import magnum as mn

        # 1. 确保输入是标准的 numpy 四元数 [w,x,y,z]
        np_quat = to_wxyz(rotation_quat)

        # 2. 从 [w,x,y,z] 提取标量和向量部分
        w, x, y, z = np_quat[0], np_quat[1], np_quat[2], np_quat[3]

        # 3. 创建Magnum Quaternion
        # Magnum Quaternion(Vector3 imag, float real)
        magnum_quat = mn.Quaternion(mn.Vector3(x, y, z), w)

        # 4. 安全赋值
        robot_obj.rotation = magnum_quat
    except Exception as e:
        raise ValueError(f"设置机器狗旋转失败: {e}")


class HabitatEnhancedViewer:
    """增强版Habitat交互式测试器，支持NavMesh和机器狗跟随"""

    def __init__(self, config_dir: str = "configs"):
        """初始化测试器

        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir)
        self.logger = self._setup_logger()

        # 加载配置
        self.loader = UnifiedConfigLoader(config_dir=str(self.config_dir))
        self.env_config = self.loader.load_environment_config()

        # Habitat模拟器
        self.sim: Optional[habitat_sim.Simulator] = None
        self.agent: Optional[habitat_sim.Agent] = None
        self.robot_obj = None  # 机器狗对象
        # 机器狗与Agent朝向之间的固定旋转修正（四元数），在加载机器人时计算
        self.robot_rotation_correction = None
        # 可配置的初始yaw对齐角（度），可在配置中设置或运行时调整
        self.robot_align_deg = None
        # 保存机器狗的初始朝向，用于后续相对旋转
        self.robot_initial_rotation = None
        # 保存Agent的初始朝向，用于计算相对旋转
        self.agent_initial_rotation = None
        # 保存前置摘像头的初始高度，确保移动时高度一致
        self.front_camera_initial_height = None

        # NavMesh配置
        self.navmesh_settings = None
        self.show_navmesh = False

        # 统计信息
        self.total_steps = 0
        self.action_counts = {
            "move_forward": 0,
            "move_backward": 0,
            "turn_left": 0,
            "turn_right": 0,
        }

        self.logger.info("✅ Habitat增强查看器初始化完成")
        self.logger.info(f"📂 配置目录: {self.config_dir}")
        self.logger.info(f"🗺️  场景文件: {self.env_config.get('scene', {}).get('path')}")
        self.logger.info(
            f"🤖 机器人URDF: {self.env_config.get('robot', {}).get('urdf_path')}"
        )

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("HabitatEnhancedViewer")
        logger.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def create_simulator(self) -> bool:
        """创建Habitat模拟器

        Returns:
            创建是否成功
        """
        try:
            self.logger.info("🔄 正在创建Habitat模拟器...")

            # 获取配置
            scene_config = self.env_config.get("scene", {})
            robot_config = self.env_config.get("robot", {})
            sensor_config = self.env_config.get("sensors", {})
            action_config = self.env_config.get("actions", {})
            agent_config = self.env_config.get("agent", {})
            physics_config = self.env_config.get("physics", {})

            front_camera_spec = habitat_sim.CameraSensorSpec()
            front_camera_spec.uuid = "front_rgb"
            front_camera_spec.sensor_type = habitat_sim.SensorType.COLOR
            front_camera_spec.resolution = sensor_config.get("front_camera", {}).get(
                "resolution", [720, 1280]
            )
            front_camera_spec.position = [0.0, agent_config.get("height", 0.8), -0.6]
            # 保存前置摘像头的初始高度
            self.front_camera_initial_height = agent_config.get("height", 0.8)
            front_camera_spec.hfov = sensor_config.get("front_camera", {}).get(
                "hfov", 90.0
            )

            # 创建后置俯视摄像头规格 (从上往下看)
            top_camera_spec = habitat_sim.CameraSensorSpec()
            top_camera_spec.uuid = "top_down_view"
            top_camera_spec.sensor_type = habitat_sim.SensorType.COLOR
            top_camera_spec.resolution = [480, 640]  # 较小分辨率用于俯视图
            # 位置: 在Agent后上方2米处
            top_camera_spec.position = [
                0.0,
                1.6,
                1.0,
            ]  # (x, y, z) - y高度1.6米，z后方0.5米
            # 朝向: 使用欧拉角向下俯视45度 (pitch=-45度)
            top_camera_spec.orientation = np.array(
                [-np.pi / 5, 0.0, 0.0]
            )  # (pitch, yaw, roll) - 向下45度
            top_camera_spec.hfov = 90.0

            # 创建Agent配置
            agent_cfg = habitat_sim.agent.AgentConfiguration()
            agent_cfg.sensor_specifications = [front_camera_spec, top_camera_spec]
            agent_cfg.action_space = {
                "move_forward": habitat_sim.agent.ActionSpec(
                    "move_forward",
                    habitat_sim.agent.ActuationSpec(
                        amount=action_config.get("move_forward", {}).get("amount", 0.25)
                    ),
                ),
                "move_backward": habitat_sim.agent.ActionSpec(
                    "move_forward",
                    habitat_sim.agent.ActuationSpec(
                        amount=-action_config.get("move_forward", {}).get(
                            "amount", 0.25
                        )
                    ),
                ),
                "turn_left": habitat_sim.agent.ActionSpec(
                    "turn_left",
                    habitat_sim.agent.ActuationSpec(
                        amount=action_config.get("turn_left", {}).get("amount", 10.0)
                    ),
                ),
                "turn_right": habitat_sim.agent.ActionSpec(
                    "turn_right",
                    habitat_sim.agent.ActuationSpec(
                        amount=action_config.get("turn_right", {}).get("amount", 10.0)
                    ),
                ),
            }
            agent_cfg.height = agent_config.get("height", 0.55)
            agent_cfg.radius = agent_config.get("radius", 0.28)

            # 创建模拟器配置
            backend_cfg = habitat_sim.SimulatorConfiguration()
            backend_cfg.scene_id = scene_config.get("path", "")
            backend_cfg.enable_physics = physics_config.get("enabled", True)

            # 创建配置
            cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])

            # 创建模拟器（支持在无法创建 windowless EGL context 时尝试不同 GPU 索引）
            def _try_create_sim(cfg, max_devices=4):
                last_exc = None
                # 如果用户已经设置了可见 GPU，则尊重该设置并只尝试它
                user_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
                user_egl = os.environ.get("EGL_DEVICE_ID")
                if user_cuda is not None or user_egl is not None:
                    try:
                        self.logger.info(
                            f"尝试使用用户环境 CUDA_VISIBLE_DEVICES={user_cuda} EGL_DEVICE_ID={user_egl} 创建 Simulator"
                        )
                        return habitat_sim.Simulator(cfg)
                    except Exception as e:
                        last_exc = e
                        self.logger.warning(f"使用用户环境创建 Simulator 失败: {e}")

                # 否则循环尝试若干个设备索引
                for dev in range(max_devices):
                    try:
                        os.environ["CUDA_VISIBLE_DEVICES"] = str(dev)
                        os.environ["EGL_DEVICE_ID"] = str(dev)
                        self.logger.info(
                            f"尝试使用设备索引 dev={dev} 创建 Simulator (设置 CUDA_VISIBLE_DEVICES/EGL_DEVICE_ID={dev})"
                        )
                        sim = habitat_sim.Simulator(cfg)
                        self.logger.info(f"成功在设备索引 {dev} 上创建 Simulator")
                        return sim
                    except Exception as e:
                        last_exc = e
                        # 记录异常并继续尝试下一个设备索引
                        self.logger.debug(f"在设备 {dev} 上创建 Simulator 失败: {e}")
                        continue

                # 若尝试所有候选仍失败，抛出最后一次异常
                if last_exc is not None:
                    self.logger.error(
                        f"尝试全部设备后仍无法创建 Simulator，最后一次错误: {last_exc}"
                    )
                    raise last_exc
                return None

            self.sim = _try_create_sim(cfg, max_devices=4)
            if self.sim is None:
                raise RuntimeError("无法创建 Simulator（尝试多设备失败）")

            self.agent = self.sim.get_agent(0)

            self.logger.info("✅ Habitat模拟器创建成功")

            # 计算NavMesh
            self._compute_navmesh()

            # 尝试加载机器人URDF
            self._load_robot()

            return True

        except Exception as e:
            self.logger.error(f"❌ 模拟器创建失败: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            return False

    def _compute_navmesh(self):
        """
        计算NavMesh可通行区域 (参考官方viewer实现)
        """
        try:
            self.logger.info("🗺️  开始计算NavMesh可通行区域...")

            # 创建NavMesh设置
            self.navmesh_settings = habitat_sim.NavMeshSettings()
            self.navmesh_settings.set_defaults()

            # 使用Agent配置设置NavMesh参数
            agent_cfg = self.sim.config.agents[0]
            self.navmesh_settings.agent_height = agent_cfg.height
            self.navmesh_settings.agent_radius = agent_cfg.radius
            self.navmesh_settings.include_static_objects = True

            # 计算NavMesh
            success = self.sim.recompute_navmesh(
                self.sim.pathfinder, self.navmesh_settings
            )

            if success and self.sim.pathfinder.is_loaded:
                self.logger.info("✅ NavMesh计算成功")
                self.logger.info(
                    f"   可通行面积: {self.sim.pathfinder.navigable_area:.2f} m²"
                )
                self.logger.info(
                    f"   可通行区域是否连通: {self.sim.pathfinder.is_loaded}"
                )

                # 启用NavMesh可视化 (默认关闭)
                self.sim.navmesh_visualization = False
                self.show_navmesh = False
            else:
                self.logger.warning("⚠️  NavMesh计算失败，但继续运行")

        except Exception as e:
            self.logger.warning(f"⚠️  NavMesh计算出错: {e}")
            self.logger.warning("   继续运行，但无法使用导航功能")

    def _load_robot(self):
        """
        加载机器狗模型并设置跟随Agent
        """
        try:
            urdf_path = self.env_config.get("robot", {}).get("urdf_path")
            if not urdf_path or not os.path.exists(urdf_path):
                self.logger.warning(f"⚠️  机器人URDF文件不存在: {urdf_path}")
                return

            self.logger.info(f"🔧 正在加载机器狗URDF: {urdf_path}")

            # 获取ArticulatedObjectManager
            aom = self.sim.get_articulated_object_manager()
            # 加载URDF (固定基座=False，允许移动)
            self.robot_obj = aom.add_articulated_object_from_urdf(
                filepath=urdf_path,
                fixed_base=False,  # 允许移动
                global_scale=1.0,
                mass_scale=1.0,
                force_reload=True,
            )

            if self.robot_obj:
                # 设置机器狗初始位置（与Agent相同位置和朝向）
                agent_state = self.agent.get_state()

                # 使用Agent的实际位置（包括高度）
                self.robot_obj.translation = agent_state.position

                # 使用Agent scene node的rotation（body朝向，不是camera朝向）
                # 获取当前旋转（四元数）
                current_rotation = self.agent.scene_node.rotation

                # 使用模块级的 to_wxyz() 来转换四元数对象为 [w,x,y,z]

                # 检查配置中是否指定了初始对齐角度（度）: `configs/... environment_config.yaml -> robot: yaw_align_deg`。
                applied_cfg_align = False
                cfg_align = self.env_config.get("robot", {}).get("yaw_align_deg", None)
                if cfg_align is not None:
                    try:
                        self.robot_align_deg = float(cfg_align)
                        self.robot_rotation_correction = quat_from_angle_axis(
                            math.radians(self.robot_align_deg),
                            np.array([0.0, 1.0, 0.0]),
                        )
                        # 立即应用到机器人初始朝向
                        try:
                            agent_q_arr = to_wxyz(current_rotation)
                            corrected_init_rot = _quat_mul(
                                self.robot_rotation_correction, agent_q_arr
                            )
                            set_robot_rotation(self.robot_obj, corrected_init_rot)
                            applied_cfg_align = True
                        except Exception:
                            applied_cfg_align = False
                    except Exception:
                        applied_cfg_align = False

                # 通过比较机器人模型的世界前向与Agent的世界前向，计算绕Y轴的对齐角度
                if applied_cfg_align:
                    # 配置已指定初始对齐，跳过自动计算
                    pass
                else:
                    try:
                        local_forward = np.array([0.0, 0.0, -1.0])

                        agent_q = to_wxyz(current_rotation)
                        try:
                            robot_q = to_wxyz(self.robot_obj.rotation)
                        except Exception:
                            robot_q = agent_q.copy()

                        agent_fwd = _quat_rotate_vector(agent_q, local_forward)
                        robot_fwd = _quat_rotate_vector(robot_q, local_forward)

                        a = np.array([agent_fwd[0], agent_fwd[2]])
                        r = np.array([robot_fwd[0], robot_fwd[2]])
                        a_norm = a / (np.linalg.norm(a) + 1e-8)
                        r_norm = r / (np.linalg.norm(r) + 1e-8)

                        det = r_norm[0] * a_norm[1] - r_norm[1] * a_norm[0]
                        dot = np.clip(np.dot(r_norm, a_norm), -1.0, 1.0)
                        yaw_delta = math.atan2(det, dot)

                        self.robot_rotation_correction = quat_from_angle_axis(
                            yaw_delta, np.array([0.0, 1.0, 0.0])
                        )
                    except Exception:
                        # 默认修正（单位四元数）
                        self.robot_rotation_correction = quat_from_angle_axis(
                            0.0, np.array([0.0, 1.0, 0.0])
                        )

                # 若未由配置应用对齐，则在此处把自动/计算得到的修正应用到初始朝向
                if not applied_cfg_align:
                    try:
                        agent_q_arr = to_wxyz(current_rotation)
                        corrected_init_rot = _quat_mul(
                            self.robot_rotation_correction, agent_q_arr
                        )
                        set_robot_rotation(self.robot_obj, corrected_init_rot)
                    except Exception:
                        set_robot_rotation(self.robot_obj, current_rotation)

                # 设置机器狗的初始朝向为特定的四元数 (0.707, 0, 0.707, 0)
                # 这表示绕Y轴旋转90度
                initial_rotation = np.array([0.707, 0.0, 0.707, 0.0])
                try:
                    set_robot_rotation(self.robot_obj, initial_rotation)
                    # 保存初始朝向和Agent初始朝向，用于后续相对旋转
                    self.robot_initial_rotation = initial_rotation.copy()
                    self.agent_initial_rotation = to_wxyz(current_rotation).copy()
                    self.logger.info(f"✅ 机器狗初始朝向已设置: {initial_rotation}")
                    self.logger.info(
                        f"✅ Agent初始朝向已记录: {self.agent_initial_rotation}"
                    )
                except Exception as e:
                    self.logger.warning(f"⚠️  设置初始朝向失败: {e}")

                self.logger.info(f"✅ 机器狗模型已加载: {self.robot_obj.handle}")
                self.logger.info(f"   初始位置: {agent_state.position}")
                self.logger.info(f"   初始朝向: {self.robot_obj.rotation}")
            else:
                self.logger.warning("⚠️  机器狗对象创建失败")

        except Exception as e:
            self.logger.warning(f"⚠️  机器狗加载失败: {e}")
            import traceback

            self.logger.warning(traceback.format_exc())

    def _update_agent_camera_height(self):
        """保持前置摄像头高度与初始高度一致

        摄像头相对于Agent的位置是固定的，但我们需要通过Agent配置来确保
        整个移动过程中高度保持初始设定的值。
        注意: Habitat中摄像头位置是相对于Agent body的固定偏移，
        一旦在create_simulator中设置，就会随Agent移动。
        """
        # 摄像头高度由Agent配置决定，会自动随Agent移动
        # 因此Habitat已经保证了高度的一致性
        if self.front_camera_initial_height is not None:
            self.logger.debug(
                f"前置摄像头初始高度: {self.front_camera_initial_height} (自动保持一致)"
            )

    def _update_robot_position(self):
        """
        更新机器狗位置以跟随Agent
        使用Agent的scene_node来获取真实的body位置和朝向
        这样机器狗可以跟随Agent到任何高度二楼、三楼等）
        并且身体旋转与Agent旋转同步转向时机器狗身体也跟随转向）
        """
        # 需要保持前置摘像头高度一致
        self._update_agent_camera_height()

        try:
            # 使用Agent scene node的实际位置（包括所有楼层的高度）
            agent_position = self.agent.scene_node.translation

            # 增加机器狗离地面的高度（y轴增加0.6单位）
            height_offset = np.array([0.0, 0.6, 0.0])
            robot_position = agent_position + height_offset

            # 使用Agent scene node的rotation（body的实际朝向）
            # 这确保了当Agent转向时，机器狗也会相
            # 应地转向
            agent_rotation = self.agent.scene_node.rotation

            # 基于初始朝向进行相对旋转
            if (
                self.robot_initial_rotation is not None
                and self.agent_initial_rotation is not None
            ):
                try:
                    # 计算Agent相对于初始朝向的旋转增量
                    current_agent_rot = to_wxyz(agent_rotation)

                    # 计算相对旋转: delta = current * inverse(initial)
                    # 四元数逆 = 共轭 / 模长平方（单位四元数共轭=逆）
                    agent_initial_conj = _quat_conjugate(self.agent_initial_rotation)
                    delta_rotation = _quat_mul(current_agent_rot, agent_initial_conj)

                    # 将相对旋转应用到机器狗初始朝向
                    # robot_current = delta * robot_initial
                    corrected_rotation = _quat_mul(
                        delta_rotation, self.robot_initial_rotation
                    )
                except Exception as e:
                    self.logger.debug(f"计算相对旋转失败: {e}，使用初始朝向")
                    corrected_rotation = self.robot_initial_rotation
            else:
                # 如果没有初始朝向记录，使用默认行为
                if getattr(self, "robot_rotation_correction", None) is None:
                    self.robot_rotation_correction = quat_from_angle_axis(
                        0.0, np.array([0.0, 1.0, 0.0])
                    )
                try:
                    ar = np.array(agent_rotation)
                    corrected_rotation = _quat_mul(self.robot_rotation_correction, ar)
                except Exception:
                    corrected_rotation = agent_rotation

            # 同步机器狗位置和朝向
            self.robot_obj.translation = robot_position
            set_robot_rotation(self.robot_obj, corrected_rotation)

        except Exception as e:
            # 如果更新失败，记录错误但不影响主循环
            if hasattr(self, "logger"):
                self.logger.debug(f"机器狗位置更新失败: {e}")

    def toggle_navmesh(self):
        """切换NavMesh可视化"""
        if self.sim and self.sim.pathfinder.is_loaded:
            self.show_navmesh = not self.show_navmesh
            self.sim.navmesh_visualization = self.show_navmesh
            status = "显示" if self.show_navmesh else "隐藏"
            self.logger.info(f"🗺️  NavMesh可视化: {status}")
        else:
            self.logger.warning("⚠️  NavMesh未加载，无法切换可视化")

    def recompute_navmesh(self):
        """重新计算NavMesh"""
        self.logger.info("🔄 重新计算NavMesh...")
        self._compute_navmesh()

    def sample_random_position(self):
        """在NavMesh上随机采样新位置"""
        if not self.sim or not self.sim.pathfinder.is_loaded:
            self.logger.warning("⚠️  NavMesh未加载，无法采样位置")
            return

        try:
            # 采样随机可通行点
            new_position = self.sim.pathfinder.get_random_navigable_point()

            # 采样随机朝向
            new_rotation = quat_from_angle_axis(
                np.random.uniform(0, 2.0 * np.pi), np.array([0, 1, 0])
            )

            # 创建新状态
            new_state = habitat_sim.AgentState()
            new_state.position = new_position
            new_state.rotation = new_rotation

            # 设置Agent状态
            self.agent.set_state(new_state)

            # 更新机器狗位置
            self._update_robot_position()

            self.logger.info(
                f"📍 随机采样新位置: ({new_position[0]:.2f}, {new_position[1]:.2f}, {new_position[2]:.2f})"
            )

        except Exception as e:
            self.logger.error(f"❌ 位置采样失败: {e}")

    def run_interactive(self):
        """
        运行交互式查看器
        """
        if not self.create_simulator():
            self.logger.error("❌ 无法创建模拟器，测试终止")
            return

        self.logger.info("\n" + "=" * 60)
        self.logger.info("🎮 Habitat增强交互式测试开始 (双摄像头分开显示)")
        self.logger.info("=" * 60)
        self.logger.info("视角说明:")
        self.logger.info("  📹 窗口1: 前置RGB摄像头 (第一人称视角)")
        self.logger.info("  🦅 窗口2: 后置俯视摄像头 (鸟瞰视角)")
        self.logger.info("")
        self.logger.info("控制说明:")
        self.logger.info("  W      : 前进")
        self.logger.info("  S      : 后退")
        self.logger.info("  A      : 左转 (机器狗身体左转)")
        self.logger.info("  D      : 右转 (机器狗身体右转)")
        self.logger.info("  N      : 切换NavMesh可视化")
        self.logger.info("  SHIFT+N: 重新计算NavMesh")
        self.logger.info("  ALT+N  : 随机采样新位置")
        self.logger.info("  ESC    : 退出")
        self.logger.info("=" * 60 + "\n")

        # 显示初始状态
        agent_state = self.agent.get_state()
        self.logger.info(f"初始位置: {agent_state.position}")
        self.logger.info(f"初始旋转: {agent_state.rotation}")

        if self.robot_obj:
            self.logger.info(f"🤖 机器狗已就位: {self.robot_obj.handle}")

        # 交互循环
        try:
            self.logger.info("🚀 启动交互式控制...\n")

            running = True
            shift_pressed = False
            alt_pressed = False

            while running:
                # 获取观测
                obs = self.sim.get_sensor_observations()

                # 获取前置RGB和俯视图像
                front_rgb = obs.get("front_rgb")
                top_view = obs.get("top_down_view")

                if front_rgb is not None and top_view is not None:
                    # 转换为BGR用于OpenCV显示
                    front_bgr = cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR)
                    top_bgr = cv2.cvtColor(top_view, cv2.COLOR_RGB2BGR)

                    # 添加信息覆盖层到前置视图
                    state = self.agent.get_state()
                    pos = state.position

                    # NavMesh状态
                    navmesh_status = "ON" if self.show_navmesh else "OFF"
                    navmesh_loaded = "✓" if self.sim.pathfinder.is_loaded else "✗"

                    info_text = [
                        f"Steps: {self.total_steps}",
                        f"Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})",
                        f"NavMesh: {navmesh_status} ({navmesh_loaded})",
                        f"Robot: {'YES' if self.robot_obj else 'NO'}",
                        f"Forward: {self.action_counts['move_forward']}",
                        f"Backward: {self.action_counts['move_backward']}",
                        f"Turn L/R: {self.action_counts['turn_left']}/{self.action_counts['turn_right']}",
                    ]

                    # 在前置视图添加文字信息
                    y_offset = 30
                    for i, text in enumerate(info_text):
                        cv2.putText(
                            front_bgr,
                            text,
                            (10, y_offset + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

                    # 在前置视图添加标签
                    cv2.putText(
                        front_bgr,
                        "Front RGB View (First Person)",
                        (10, front_bgr.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2,
                    )

                    # 在俯视图添加标签
                    cv2.putText(
                        top_bgr,
                        "Top-Down View (Bird's Eye)",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                    # 在两个独立的窗口中显示
                    cv2.imshow("Front RGB Camera - First Person View", front_bgr)
                    cv2.imshow("Top-Down Camera - Bird's Eye View", top_bgr)

                    # 键盘输入
                    key = cv2.waitKey(1) & 0xFF

                    # 检测修饰键 (OpenCV限制，需要在动作键之前检测)
                    if key == 225:  # Left Shift
                        shift_pressed = True
                    elif key == 233:  # Left Alt
                        alt_pressed = True

                    # 处理动作
                    if key == 27:  # ESC
                        self.logger.info("👋 退出测试")
                        running = False

                    elif key == ord("w") or key == ord("W"):
                        self.agent.act("move_forward")
                        self.total_steps += 1
                        self.action_counts["move_forward"] += 1
                        self._update_robot_position()
                        state = self.agent.get_state()
                        pos = state.position
                        robot_info = ""
                        if self.robot_obj:
                            robot_pos = self.robot_obj.translation
                            robot_info = f" | Robot: ({robot_pos[0]:6.2f}, {robot_pos[1]:6.2f}, {robot_pos[2]:6.2f})"
                        self.logger.info(
                            f"🎮 #{self.total_steps}: FORWARD  → ({pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}){robot_info}"
                        )

                    elif key == ord("s") or key == ord("S"):
                        self.agent.act("move_backward")
                        self.total_steps += 1
                        self.action_counts["move_backward"] += 1
                        self._update_robot_position()
                        state = self.agent.get_state()
                        pos = state.position
                        robot_info = ""
                        if self.robot_obj:
                            robot_pos = self.robot_obj.translation
                            robot_info = f" | Robot: ({robot_pos[0]:6.2f}, {robot_pos[1]:6.2f}, {robot_pos[2]:6.2f})"
                        self.logger.info(
                            f"🎮 #{self.total_steps}: BACKWARD → ({pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}){robot_info}"
                        )

                    elif key == ord("a") or key == ord("A"):
                        self.agent.act("turn_left")
                        self.total_steps += 1
                        self.action_counts["turn_left"] += 1
                        self._update_robot_position()
                        state = self.agent.get_state()
                        robot_rot_info = ""
                        if self.robot_obj:
                            robot_rot = self.robot_obj.rotation
                            robot_rot_info = f" | Robot rot: {robot_rot}"
                        self.logger.info(
                            f"🎮 #{self.total_steps}: TURN LEFT{robot_rot_info}"
                        )

                    elif key == ord("d") or key == ord("D"):
                        self.agent.act("turn_right")
                        self.total_steps += 1
                        self.action_counts["turn_right"] += 1
                        self._update_robot_position()
                        state = self.agent.get_state()
                        robot_rot_info = ""
                        if self.robot_obj:
                            robot_rot = self.robot_obj.rotation
                            robot_rot_info = f" | Robot rot: {robot_rot}"
                        self.logger.info(
                            f"🎮 #{self.total_steps}: TURN RIGHT{robot_rot_info}"
                        )

                    # 交互式微调机器人对齐角度
                    elif key == ord("["):
                        # 向左减小对齐角（度）
                        if self.robot_align_deg is None:
                            self.robot_align_deg = 0.0
                        self.robot_align_deg -= 15.0
                        self.robot_rotation_correction = quat_from_angle_axis(
                            math.radians(self.robot_align_deg),
                            np.array([0.0, 1.0, 0.0]),
                        )
                        self._update_robot_position()
                        self.logger.info(
                            f"🔧 调整机器人对齐角: {self.robot_align_deg:.1f}°"
                        )

                    elif key == ord("]"):
                        # 向右增大对齐角（度）
                        if self.robot_align_deg is None:
                            self.robot_align_deg = 0.0
                        self.robot_align_deg += 15.0
                        self.robot_rotation_correction = quat_from_angle_axis(
                            math.radians(self.robot_align_deg),
                            np.array([0.0, 1.0, 0.0]),
                        )
                        self._update_robot_position()
                        self.logger.info(
                            f"🔧 调整机器人对齐角: {self.robot_align_deg:.1f}°"
                        )

                    elif key == ord("p") or key == ord("P"):
                        self.logger.info(
                            f"🔍 当前机器人对齐角 (deg): {self.robot_align_deg}"
                        )

                    elif key == ord("n") or key == ord("N"):
                        # N: 切换NavMesh可视化
                        # SHIFT+N: 重新计算
                        # ALT+N: 随机采样位置
                        if shift_pressed:
                            self.recompute_navmesh()
                            shift_pressed = False
                        elif alt_pressed:
                            self.sample_random_position()
                            alt_pressed = False
                        else:
                            self.toggle_navmesh()

                    # 重置修饰键 (简化版，实际应该监听keyup)
                    if key != 225:
                        shift_pressed = False
                    if key != 233:
                        alt_pressed = False

            cv2.destroyAllWindows()

        except KeyboardInterrupt:
            self.logger.info("\n⚠️  测试被用户中断")

        except Exception as e:
            self.logger.error(f"❌ 查看器运行出错: {e}")
            import traceback

            self.logger.error(traceback.format_exc())

        finally:
            # 显示统计
            self.logger.info("\n" + "=" * 60)
            self.logger.info("📊 测试统计")
            self.logger.info("=" * 60)
            self.logger.info(f"总步数: {self.total_steps}")
            self.logger.info(f"前进: {self.action_counts['move_forward']}")
            self.logger.info(f"后退: {self.action_counts['move_backward']}")
            self.logger.info(f"左转: {self.action_counts['turn_left']}")
            self.logger.info(f"右转: {self.action_counts['turn_right']}")

            if self.sim and self.sim.pathfinder.is_loaded:
                self.logger.info(
                    f"NavMesh面积: {self.sim.pathfinder.navigable_area:.2f} m²"
                )

            self.logger.info("=" * 60)
            self.logger.info("✅ 测试完成")

            # 清理
            if self.sim:
                self.logger.info("🧹 清理模拟器资源...")
                self.sim.close()


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("Habitat环境增强交互式测试")
    print("=" * 70)
    print("功能:")
    print("  ✓ NavMesh可通行区域计算与可视化")
    print("  ✓ 机器狗模型跟随Agent移动")
    print("  ✓ WASD键盘控制")
    print("  ✓ 实时统计信息显示")
    print("=" * 70 + "\n")

    # 检查Habitat是否可用
    try:
        import habitat_sim

        print(f"✅ Habitat-sim 已安装 (v{habitat_sim.__version__})")
    except ImportError:
        print("❌ 未安装 Habitat-sim，无法运行测试")
        print("   请先安装: conda install habitat-sim -c conda-forge -c aihabitat")
        return

    # 创建并运行测试器
    viewer = HabitatEnhancedViewer(config_dir="configs")

    try:
        viewer.run_interactive()
    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
