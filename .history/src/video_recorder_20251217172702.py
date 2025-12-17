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
import imageio
logger = logging.getLogger(__name__)

class BackCameraRecorder:
    """使用 imageio 的录制器"""
    
    def __init__(self, output_dir, fps=30, frame_size=(1280, 960)):
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.frame_size = frame_size
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = self.output_dir / f"back_camera_{timestamp}.mp4"
        
        # imageio 会自动使用 FFmpeg
        self.writer = imageio.get_writer(
            str(self.video_path),
            fps=fps,
            codec='libx264',
            quality=8,  # 1-10，10 最佳
            pixelformat='yuv420p',
            macro_block_size=1
        )
        
        logger.info(f"✅ 录制器初始化: {self.video_path}")
    
    def write_frame(self, frame: np.ndarray, **kwargs) -> bool:
        """写入帧（frame 是 RGB 格式）"""
        try:
            # imageio 接受 RGB 格式
            if frame.shape[:2] != (self.frame_size[1], self.frame_size[0]):
                import cv2
                frame = cv2.resize(frame, self.frame_size)
            
            # TODO: 添加注释（需要先转 BGR 再处理，最后转回 RGB）
            
            self.writer.append_data(frame)
            return True
        except Exception as e:
            logger.error(f"❌ 写入帧失败: {e}")
            return False
    
    def release(self):
        self.writer.close()
        logger.info(f"✅ 视频已保存: {self.video_path}")
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

def detect_available_codecs() -> Dict[str, bool]:
    """
    检测系统支持的视频编码器

    Returns:
        编码器可用性字典
    """
    if cv2 is None:
        return {}

    test_codecs = ["H264", "X264", "avc1", "mp4v", "XVID", "MJPG"]
    results = {}

    # 临时测试文件
    import tempfile
    temp_path = Path(tempfile.gettempdir()) / "codec_test.mp4"

    for codec_str in test_codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_str)
            writer = cv2.VideoWriter(str(temp_path), fourcc, 30, (640, 480))

            is_available = writer.isOpened()
            writer.release()

            results[codec_str] = is_available

        except Exception:
            results[codec_str] = False

    # 清理测试文件
    if temp_path.exists():
        temp_path.unlink()

    return results


# 在模块加载时打印可用编码器（可选）
if __name__ == "__main__":
    codecs = detect_available_codecs()
    logger.info("可用的视频编码器:")
    for codec, available in codecs.items():
        status = "✅" if available else "❌"
        logger.info(f"  {status} {codec}")