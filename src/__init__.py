#!/usr/bin/env python3
"""
VLM Navigation System - Core Module
多智能体视觉语言导航系统 - 核心模块
"""

# 导入核心模块
from .config_loader import UnifiedConfigLoader
from .navigation_system import NavigationSystem
from .habitat_integration import (
    HabitatFramework,
    HabitatVersionInfo,
    EnvironmentFactory,
)
from .video_recorder import BackCameraRecorder, FloorplanGenerator
from .agents import Agent1, Agent2
from .scene_graph import SceneGraph

__version__ = "1.0.0"
__all__ = [
    "UnifiedConfigLoader",
    "NavigationSystem",
    "HabitatFramework",
    "HabitatVersionInfo",
    "EnvironmentFactory",
    "BackCameraRecorder",
    "FloorplanGenerator",
    "Agent1",
    "Agent2",
    "SceneGraph",
]
