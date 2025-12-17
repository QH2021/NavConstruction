#!/usr/bin/env python3
"""
视频记录和传感器模块
====================

功能：
1. 后置RGB传感器（俯视视角）
2. 视频记录+叠加参数和可通行区域
3. 实时动态参数显示
4. 机器狗位置高亮
"""

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None
import numpy as np
import os
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class BackCameraRecorder:
    """后置相机录制器（俯视视角）"""

    def __init__(
        self,
        output_dir: str,
        fps: int = 30,
        frame_size: Tuple[int, int] = (1280, 960),
        codec: str = "avc1",
        show_floorplan_overlay: bool = False,
    ):
        """
        初始化后置相机录制器

        Args:
            output_dir: 输出目录
            fps: 帧率
            frame_size: 帧尺寸 (W, H)
            codec: 视频编码器
        """
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        self.show_floorplan_overlay = bool(show_floorplan_overlay)

        if cv2 is None:
            raise ImportError(
                "BackCameraRecorder 需要 opencv-python（cv2）。请先执行: pip install opencv-python"
            )

        # 视频文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = self.output_dir / f"back_camera_{timestamp}.mp4"

        # 初始化视频编写器
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(str(self.video_path), fourcc, fps, frame_size)

        if not self.writer.isOpened():
            raise RuntimeError("❌ 无法初始化视频编写器")

        logger.info(f"✅ 后置相机录制器初始化: {self.video_path}")
        logger.info(f"   帧率: {fps} fps, 尺寸: {frame_size}")

    def write_frame(
        self,
        frame: np.ndarray,
        floorplan: Optional[np.ndarray] = None,
        robot_position: Optional[Tuple[float, float]] = None,
        robot_heading: Optional[float] = None,
        metrics: Optional[Dict[str, any]] = None,
    ) -> bool:
        """
        写入并注释帧到视频

        Args:
            frame: RGB图像 (H, W, 3) - 从Habitat或观测中获得的RGB格式
            floorplan: 可通行区域平面图 (H, W, 3) BGR格式
            robot_position: 机器狗在平面图中的位置 (x, y)
            robot_heading: 机器狗朝向角度 (度数)
            metrics: 实时参数字典

        Returns:
            是否成功写入
        """
        try:
            # 【关键】输入是RGB格式，需要转换为BGR用于cv2.VideoWriter
            import cv2

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 确保帧大小正确
            if frame_bgr.shape[:2] != (self.frame_size[1], self.frame_size[0]):
                frame_bgr = cv2.resize(frame_bgr, self.frame_size)

            # 制作注释帧的副本
            annotated_frame = frame_bgr.copy()

            # 按需求：视频不需要叠加平面图（默认关闭；保留参数以兼容旧调用）
            if self.show_floorplan_overlay and floorplan is not None:
                annotated_frame = self._draw_floorplan_overlay(
                    annotated_frame, floorplan
                )

            # 绘制机器狗位置
            if robot_position is not None:
                annotated_frame = self._draw_robot_position(
                    annotated_frame, robot_position, robot_heading
                )

            # 绘制实时参数（右上角）
            if metrics is not None:
                annotated_frame = self._draw_metrics(annotated_frame, metrics)

            # 写入视频
            self.writer.write(annotated_frame)
            return True

        except Exception as e:
            logger.error(f"❌ 写入视频帧失败: {e}")
            return False

    @staticmethod
    def _draw_floorplan_overlay(
        frame: np.ndarray, floorplan: np.ndarray, alpha: float = 0.3
    ) -> np.ndarray:
        """绘制可通行区域平面图覆盖层"""
        try:
            # 调整平面图大小
            if floorplan.shape[:2] != frame.shape[:2]:
                floorplan = cv2.resize(floorplan, (frame.shape[1], frame.shape[0]))

            # Alpha混合
            frame = cv2.addWeighted(frame, 1 - alpha, floorplan, alpha, 0)

            # 添加标签
            cv2.putText(
                frame,
                "Floorplan Overlay",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            return frame
        except Exception as e:
            logger.warning(f"⚠️ 绘制平面图失败: {e}")
            return frame

    @staticmethod
    def _draw_robot_position(
        frame: np.ndarray,
        position: Tuple[float, float],
        heading: Optional[float] = None,
        radius: int = 15,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """绘制机器狗位置和朝向"""
        try:
            x, y = int(position[0]), int(position[1])

            # 绘制位置圆点
            cv2.circle(frame, (x, y), radius, color, -1)
            cv2.circle(frame, (x, y), radius, (255, 255, 255), 2)

            # 绘制朝向箭头
            if heading is not None:
                # 将角度转换为弧度
                angle_rad = np.radians(heading)
                arrow_length = radius * 2
                end_x = int(x + arrow_length * np.cos(angle_rad))
                end_y = int(y + arrow_length * np.sin(angle_rad))
                cv2.arrowedLine(
                    frame, (x, y), (end_x, end_y), (255, 0, 0), 2, tipLength=0.3
                )

            return frame
        except Exception as e:
            logger.warning(f"⚠️ 绘制机器狗位置失败: {e}")
            return frame

    @staticmethod
    def _draw_metrics(frame: np.ndarray, metrics: Dict[str, any]) -> np.ndarray:
        """绘制实时参数"""
        try:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (255, 255, 255)
            thickness = 1

            # 右上角布局
            padding = 12
            line_h = 20
            start_y = 30
            frame_h, frame_w = frame.shape[:2]

            # 参数列表
            step = metrics.get("step", metrics.get("steps", "N/A"))
            room = metrics.get("room", "N/A")
            pos = metrics.get("position", "N/A")
            action = metrics.get("action", "")
            status = metrics.get("status", "Running")

            lines = [
                f"Step: {step}",
                f"Room: {room}",
                f"Pos: {pos}",
            ]
            if action:
                lines.append(f"Action: {action}")
            lines.append(f"Status: {status}")

            # 计算最大文本宽度以右对齐
            max_w = 0
            for text in lines:
                (tw, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
                max_w = max(max_w, tw)
            x = max(padding, frame_w - padding - max_w)

            # 半透明背景
            box_w = max_w + padding * 2
            box_h = line_h * len(lines) + padding
            x0 = max(0, x - padding)
            y0 = max(0, start_y - line_h)
            x1 = min(frame_w - 1, x0 + box_w)
            y1 = min(frame_h - 1, y0 + box_h)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

            for i, text in enumerate(lines):
                cv2.putText(
                    frame,
                    text,
                    (x, start_y + i * line_h),
                    font,
                    font_scale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

            return frame
        except Exception as e:
            logger.warning(f"⚠️ 绘制参数失败: {e}")
            return frame

    def release(self):
        """释放视频编写器"""
        if self.writer is not None:
            self.writer.release()
            logger.info(f"✅ 视频已保存: {self.video_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class FloorplanGenerator:
    """可通行区域平面图生成器"""

    @staticmethod
    def create_passable_area_map(
        width: int,
        height: int,
        passable_points: np.ndarray,
        robot_position: Optional[Tuple[float, float]] = None,
        robot_radius: float = 0.3,
    ) -> np.ndarray:
        """
        生成可通行区域平面图

        Args:
            width: 平面图宽度
            height: 平面图高度
            passable_points: 可通行点集 (Nx2)
            robot_position: 机器狗位置 (x, y)
            robot_radius: 机器狗半径

        Returns:
            BGR图像 (H, W, 3)
        """
        # 创建底层（白色背景）
        floorplan = np.ones((height, width, 3), dtype=np.uint8) * 255

        # 绘制可通行区域（绿色）
        if len(passable_points) > 0:
            passable_points_int = passable_points.astype(np.int32)
            cv2.fillPoly(floorplan, [passable_points_int], (0, 255, 0))

        # 绘制机器狗当前位置（蓝色圆点 + 红色圆圈）
        if robot_position is not None:
            x, y = int(robot_position[0]), int(robot_position[1])
            r = int(robot_radius * 50)  # 放大倍数
            cv2.circle(floorplan, (x, y), r, (255, 0, 0), -1)  # 蓝色填充
            cv2.circle(floorplan, (x, y), r, (0, 0, 255), 2)  # 红色边框

        return floorplan

    @staticmethod
    def create_dummy_floorplan(width: int, height: int) -> np.ndarray:
        """
        创建虚拟平面图（用于测试）

        Returns:
            BGR图像 (H, W, 3)
        """
        floorplan = np.ones((height, width, 3), dtype=np.uint8) * 255

        # 绘制一个简单的可通行区域
        points = np.array(
            [[50, 50], [width - 50, 50], [width - 50, height - 50], [50, height - 50]],
            dtype=np.int32,
        )
        cv2.fillPoly(floorplan, [points], (0, 255, 0))

        # 添加一些障碍物（红色）
        cv2.rectangle(floorplan, (200, 200), (400, 400), (0, 0, 255), -1)
        cv2.circle(floorplan, (width // 2, height // 2), 100, (0, 100, 255), -1)

        return floorplan
