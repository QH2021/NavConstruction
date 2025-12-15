#!/usr/bin/env python3
"""
实时可视化管理器 - 参考test_habitat_interactive_enhanced.py的实现

显示前置和后置摄像头的实时画面，以及VLM决策信息
"""

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None
import numpy as np
from typing import Dict, Optional, Tuple


class RealtimeVisualizer:
    """实时可视化管理器，显示前后摄像头和VLM决策信息"""

    def __init__(self, enable: bool = True):
        """
        初始化可视化器

        Args:
            enable: 是否启用可视化
        """
        # 若缺少 OpenCV，自动降级为无可视化模式（确保 run.py --help / import 不失败）
        self.enable = bool(enable) and (cv2 is not None)
        self.window_front = "Front RGB Camera - First Person View"
        self.window_back = "Top-Down Camera - Bird's Eye View"

        if self.enable:
            # 创建窗口（只创建一次）
            cv2.namedWindow(self.window_front, cv2.WINDOW_NORMAL)
            cv2.namedWindow(self.window_back, cv2.WINDOW_NORMAL)

            # 调整窗口大小以适应显示
            cv2.resizeWindow(self.window_front, 1280, 720)
            cv2.resizeWindow(self.window_back, 1280, 960)

            # 排列窗口位置
            cv2.moveWindow(self.window_front, 0, 0)
            cv2.moveWindow(self.window_back, 1280 + 20, 0)

    def display(
        self,
        rgb_front: np.ndarray,
        rgb_back: np.ndarray,
        vlm_info: Optional[Dict] = None,
        delay_ms: int = 1,
    ) -> bool:
        """
        显示前后摄像头画面和VLM信息

        Args:
            rgb_front: 前置摄像头RGB图像 (HxWx3)
            rgb_back: 后置摄像头RGB图像 (HxWx3)
            vlm_info: VLM决策信息字典，包含:
                - actions: 动作列表
                - reasoning: 推理文本
                - current_room: 当前房间
                - target_room: 目标房间
                - step: 步数
            delay_ms: 显示延迟（毫秒）

        返回：
            是否继续显示（False表示用户按下ESC）
        """
        if not self.enable:
            return True

        try:
            # 确保是RGB格式，转换为BGR用于OpenCV显示
            front_bgr = cv2.cvtColor(rgb_front, cv2.COLOR_RGB2BGR)
            back_bgr = cv2.cvtColor(rgb_back, cv2.COLOR_RGB2BGR)

            # 添加信息文本到前置摄像头
            front_bgr = self._add_front_camera_info(front_bgr, vlm_info)

            # 添加信息文本到后置摄像头
            back_bgr = self._add_back_camera_info(back_bgr, vlm_info)

            # 显示图像
            cv2.imshow(self.window_front, front_bgr)
            cv2.imshow(self.window_back, back_bgr)

            # 等待键盘输入
            key = cv2.waitKey(delay_ms) & 0xFF

            # ESC键退出
            if key == 27:
                return False

            return True

        except Exception as e:
            print(f"⚠️ 可视化显示异常: {e}")
            return True

    def _add_front_camera_info(
        self, image: np.ndarray, vlm_info: Optional[Dict] = None
    ) -> np.ndarray:
        """
        向前置摄像头图像添加信息文本

        Args:
            image: BGR格式的图像
            vlm_info: VLM决策信息

        返回：
            添加了文本的图像
        """
        img = image.copy()

        if not vlm_info:
            return img

        # 参考 enhanced viewer 的信息布局
        step = vlm_info.get("step", 0)
        agent_pos = vlm_info.get("agent_pos")
        if isinstance(agent_pos, (tuple, list)) and len(agent_pos) >= 3:
            pos_str = f"({agent_pos[0]:.2f}, {agent_pos[1]:.2f}, {agent_pos[2]:.2f})"
        else:
            pos_str = str(vlm_info.get("position", "N/A"))

        navmesh_status = "ON" if vlm_info.get("show_navmesh") else "OFF"
        navmesh_loaded = "✓" if vlm_info.get("navmesh_loaded") else "✗"
        robot_loaded = "YES" if vlm_info.get("robot_loaded") else "NO"

        nav_area = vlm_info.get("navigable_area")
        try:
            nav_area_val = float(nav_area) if nav_area is not None else None
        except Exception:
            nav_area_val = None

        action_counts = vlm_info.get("action_counts") or {}
        info_lines = [
            f"Steps: {step}",
            f"Pos: {pos_str}",
            f"NavMesh: {navmesh_status} ({navmesh_loaded})",
            f"Area: {nav_area_val:.1f} m^2"
            if nav_area_val is not None
            else "Area: N/A",
            f"Robot: {robot_loaded}",
            f"Forward: {action_counts.get('move_forward', 0)}",
            f"Backward: {action_counts.get('move_backward', 0)}",
            f"Turn L/R: {action_counts.get('turn_left', 0)}/{action_counts.get('turn_right', 0)}",
        ]

        y_offset = 30
        for i, text in enumerate(info_lines):
            cv2.putText(
                img,
                text,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # 底部标签
        cv2.putText(
            img,
            "Front RGB View (First Person)",
            (10, img.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return img

    def _add_back_camera_info(
        self, image: np.ndarray, vlm_info: Optional[Dict] = None
    ) -> np.ndarray:
        """
        向后置摄像头图像添加信息文本

        Args:
            image: BGR格式的图像
            vlm_info: VLM决策信息

        返回：
            添加了文本的图像
        """
        img = image.copy()

        # 参考 enhanced viewer：俯视图只需要标签
        cv2.putText(
            img,
            "Top-Down View (Bird's Eye)",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return img

    def close(self):
        """关闭可视化窗口"""
        if self.enable:
            cv2.destroyAllWindows()

    def __enter__(self):
        """上下文管理器支持"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器支持"""
        self.close()
