#!/usr/bin/env python3
"""
多智能体VLM导航系统 - 场景图和路径规划模块
========================================

功能：
- 读取Excel表格（door_table.xlsx, component_table.xlsx）
- 构建场景图（房间图+构件映射）
- A*算法路径规划
- 获取K条最短路径

设计特点：
- 支持房间号/构件号输入解析
- 原始ID管理，避免歧义
- 离线可用，无外部依赖（除pandas）
"""

import pandas as pd
import re
from collections import deque
from typing import Dict, List, Tuple, Optional, Set, Any
import logging

logger = logging.getLogger(__name__)


class SceneGraph:
    """场景图管理类 - 房间、构件、门、路径规划"""

    def __init__(self, door_excel: str, comp_excel: str):
        """
        初始化场景图

        Args:
            door_excel: door_table.xlsx 路径
            comp_excel: component_table.xlsx 路径
        """
        self.door_df = None
        self.comp_df = None
        self.graph: Dict[str, List[str]] = {}  # room_id -> [neighbors]
        self.door_info: Dict[
            Tuple[str, str], Tuple[str, float]
        ] = {}  # (u,v) -> (door_id, width)
        self.comp_to_room: Dict[str, str] = {}  # comp_id -> room_id

        self._load_excel_files(door_excel, comp_excel)
        self._build_graph()

    def _load_excel_files(self, door_excel: str, comp_excel: str):
        """加载Excel文件"""
        try:
            self.door_df = pd.read_excel(door_excel)
            self.comp_df = pd.read_excel(comp_excel)
            logger.info(f"✅ 加载Excel文件: {door_excel}, {comp_excel}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"❌ Excel文件不存在: {e}")

    def _build_graph(self):
        """构建房间图和构件映射"""
        # 收集所有房间
        all_rooms = set(self.door_df["Room_From"]).union(set(self.door_df["Room_To"]))
        if "Room_ID" in self.comp_df.columns:
            all_rooms = all_rooms.union(set(self.comp_df["Room_ID"]))

        # 初始化图
        self.graph = {room: [] for room in all_rooms}

        # 添加边和门信息
        for _, row in self.door_df.iterrows():
            u = row["Room_From"]
            v = row["Room_To"]
            self.graph[u].append(v)
            self.graph[v].append(u)
            self.door_info[(u, v)] = (row["Door_ID"], row["Clear_Width_m"])
            self.door_info[(v, u)] = (row["Door_ID"], row["Clear_Width_m"])

        # 构件映射
        if "Room_ID" in self.comp_df.columns and "Comp_ID" in self.comp_df.columns:
            for _, row in self.comp_df.iterrows():
                self.comp_to_room[row["Comp_ID"]] = row["Room_ID"]

        logger.info(
            f"✅ 场景图已构建: {len(all_rooms)} 个房间, {len(self.door_info) // 2} 条边"
        )

    @staticmethod
    def extract_floor(room_id: str) -> int:
        """从房间ID中提取楼层号"""
        match = re.search(r"\d+", str(room_id))
        if match:
            num_str = match.group()
            return int(num_str[0]) if num_str else 0
        return 0

    def parse_location(self, user_input: str) -> Optional[str]:
        """
        解析用户输入为房间ID
        支持: "Room_S101", "Comp_W101", "S101"
        """
        user_input = user_input.strip()

        if user_input.startswith("Room_"):
            return user_input[5:]
        elif user_input.startswith("Comp_"):
            comp_id = user_input[5:]
            return self.comp_to_room.get(comp_id)
        else:
            # 直接作为房间ID
            if user_input in self.graph:
                return user_input
            elif user_input in self.comp_to_room:
                return self.comp_to_room[user_input]
        return None

    def _astar_shortest_path(
        self, start: str, end: str, blocked_rooms: Set[str] = None
    ) -> Optional[Tuple[List[str], int]]:
        """A*算法求最短路径，支持避开阻挡房间"""
        if start == end:
            return ([start], 0)
        if start not in self.graph or end not in self.graph:
            return None

        blocked = blocked_rooms or set()
        if start in blocked or end in blocked:
            return None

        # A*使用优先队列: (f_score, g_score, current, path)
        import heapq

        # 启发式函数：楼层差异（简化版）
        def heuristic(room1: str, room2: str) -> int:
            floor1 = self.extract_floor(room1)
            floor2 = self.extract_floor(room2)
            return abs(floor1 - floor2)

        open_set = [(heuristic(start, end), 0, start, [start])]
        visited = set()

        while open_set:
            f_score, g_score, current, path = heapq.heappop(open_set)

            if current in visited:
                continue
            visited.add(current)

            if current == end:
                return (path, g_score)

            for neighbor in self.graph.get(current, []):
                if neighbor in visited or neighbor in blocked:
                    continue

                new_g = g_score + 1
                new_f = new_g + heuristic(neighbor, end)
                new_path = path + [neighbor]
                heapq.heappush(open_set, (new_f, new_g, neighbor, new_path))

        return None

    def _find_k_different_paths(
        self, start: str, end: str, k: int = 3, blocked_rooms: Set[str] = None
    ) -> List[Tuple[List[str], int]]:
        """找到k条不同的最短路径（通过临时阻挡已使用的路径）"""
        paths = []
        blocked = blocked_rooms or set()
        temp_blocked = blocked.copy()

        for i in range(k):
            result = self._astar_shortest_path(start, end, temp_blocked)
            if not result:
                break

            path, cost = result
            paths.append((path, cost))

            # 为了找到不同路径，临时阻挡路径中的一个中间节点
            if i < k - 1 and len(path) > 2:
                # 阻挡路径中间的某个房间（不包括起点和终点）
                mid_idx = len(path) // 2
                temp_blocked.add(path[mid_idx])

        return paths

    def find_k_shortest_paths(
        self,
        start_input: str,
        end_input: str,
        k: int = 3,
        blocked_rooms: Set[str] = None,
    ) -> Optional[List[Tuple[List[str], List[str], int]]]:
        """
        使用A*算法获取k条最短路径（候选路径）

        Args:
            start_input: 起点（支持Room_/Comp_前缀或直接ID）
            end_input: 终点
            k: 候选路径数量
            blocked_rooms: 需要避开的房间集合

        Returns:
            [(rooms, doors, steps), ...] 或 None
        """
        start_room = self.parse_location(start_input)
        end_room = self.parse_location(end_input)

        if start_room is None or end_room is None:
            logger.error(f"❌ 起点或终点无效: {start_input} -> {end_input}")
            return None
        if start_room not in self.graph or end_room not in self.graph:
            logger.error(f"❌ 起点或终点不在场景图中: {start_room} / {end_room}")
            return None
        if start_room == end_room:
            logger.info(f"✅ 起点即终点: {start_room}")
            return [([start_room], [], 0)]

        # 使用A*算法找k条不同的路径
        path_results = self._find_k_different_paths(
            start_room, end_room, k, blocked_rooms
        )

        if not path_results:
            logger.error(f"❌ 未找到从 {start_room} 到 {end_room} 的路径")
            return None

        # 提取门信息和步数
        results = []
        for rooms, steps in path_results:
            doors = []
            for i in range(len(rooms) - 1):
                u, v = rooms[i], rooms[i + 1]
                door_id, _ = self.door_info.get((u, v), ("Unknown", 0))
                doors.append(door_id)
            results.append((rooms, doors, steps))

        logger.info(f"✅ A*算法找到 {len(results)} 条候选路径")
        for i, (rooms, doors, steps) in enumerate(results, 1):
            logger.info(f"   路径{i}: {' → '.join(rooms)} ({steps}步)")

        return results

    def get_path_details(self, rooms: List[str]) -> Dict[str, Any]:
        """获取路径的详细信息"""
        doors = []
        widths = []
        for i in range(len(rooms) - 1):
            u, v = rooms[i], rooms[i + 1]
            door_id, width = self.door_info.get((u, v), ("Unknown", 0))
            doors.append(door_id)
            widths.append(width)

        return {
            "rooms": rooms,
            "doors": doors,
            "widths": widths,
            "steps": len(rooms) - 1,
            "floors": [self.extract_floor(r) for r in rooms],
        }
