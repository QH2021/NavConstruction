#!/usr/bin/env python3
"""
多智能体VLM导航系统 - 智能体基类和具体实现
==========================================

设计：
1. VLMAgent: 基类，包含记忆机制
2. Agent1 (ActionControlAgent): 动作控制，输出5步动作
3. Agent2 (PathPlanningAgent): 通行判断，路径重规划

记忆设计：
- Agent1: 记录前N次的RGB、动作、位置
- Agent2: 记录通行/不通行的位置点
"""

import logging
import json
from typing import Dict, List, Optional, Any, Deque, Tuple
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import hashlib
import re
import time
import requests
from requests.adapters import HTTPAdapter
import base64
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MemoryFrame:
    """记忆帧"""

    timestamp: str
    data: Dict[str, Any]


class VLMAgent:
    """VLM智能体基类"""

    def __init__(
        self,
        agent_id: str,
        memory_size: int = None,
        config_dir: str = "./configs",
        output_dir: str = None,
        vlm_trace_path: str = None,
    ):
        """
        初始化VLM智能体

        Args:
            agent_id: 智能体ID ("Agent1" or "Agent2")
            memory_size: 记忆缓冲区大小（None则从配置读取）
            config_dir: 配置文件目录
            output_dir: 输出目录（用于保存VLM输入输出）
        """
        # 加载配置
        try:
            # 作为包使用（推荐）：项目根目录在 sys.path
            from src.config_loader import UnifiedConfigLoader
        except ModuleNotFoundError:
            # 兼容测试/脚本把 ./src 加进 sys.path 的用法
            from config_loader import UnifiedConfigLoader

        self._config_loader = UnifiedConfigLoader(config_dir=config_dir)

        # 优先加载统一配置
        try:
            unified_config = self._config_loader.load_unified_config()
            self._vlm_config = unified_config.get("vlm", {})
            self._agent_config = unified_config.get("agents", {})
            self._prompts_config = unified_config.get("prompts", {})
        except Exception:
            # 回退到独立配置文件
            self._vlm_config = self._config_loader.load_vlm_config()
            self._agent_config = self._config_loader.load_config("agent_config")
            try:
                self._prompts_config = self._config_loader.load_prompts_config()
            except Exception:
                self._prompts_config = {}

        self.agent_id = agent_id
        self.output_dir = Path(output_dir) if output_dir else None

        self._vlm_trace_path = Path(vlm_trace_path) if vlm_trace_path else None

        # 从配置读取memory_size
        if memory_size is None:
            agent_key = agent_id.lower()
            memory_size = self._agent_config.get(agent_key, {}).get("memory_size", 10)

        self.memory: Deque[MemoryFrame] = deque(maxlen=memory_size)
        self.last_action = None
        self.success_count = 0
        self.failure_count = 0

        # ──────────────────────────────────────────────────────────────
        # 长期关键记忆（LTM: Long-Term Memory）
        # 目标：把“关键事实/高置信定位/失败模式/风险点”长期保留，并以极短摘要注入 prompt。
        # 设计：
        # - STM: self.memory（短期窗口）
        # - LTM: self._ltm_entries（去重+重要度+可落盘）
        # - retrieval: 注入 prompt 时只取少量 top-K（含 pinned）
        # ──────────────────────────────────────────────────────────────
        agent_key = (agent_id or "").lower()
        agent_cfg = (
            (self._agent_config.get(agent_key, {}) or {})
            if isinstance(self._agent_config, dict)
            else {}
        )
        ltm_cfg = (
            (agent_cfg.get("long_term_memory", {}) or {})
            if isinstance(agent_cfg, dict)
            else {}
        )

        self._ltm_enabled = bool(ltm_cfg.get("enabled", True))
        try:
            self._ltm_max_entries = int(ltm_cfg.get("max_entries", 80))
        except Exception:
            self._ltm_max_entries = 80
        self._ltm_max_entries = max(10, self._ltm_max_entries)

        try:
            self._ltm_max_chars_in_prompt = int(
                ltm_cfg.get("max_chars_in_prompt", 1200)
            )
        except Exception:
            self._ltm_max_chars_in_prompt = 1200
        self._ltm_max_chars_in_prompt = max(200, self._ltm_max_chars_in_prompt)

        # persist_path:
        # 1) 配置显式指定 -> 2) output_dir/run 输出目录 -> 3) 不落盘（仅内存）
        self._ltm_persist_path: Optional[Path] = None
        try:
            configured_path = ltm_cfg.get("persist_path")
            if configured_path:
                self._ltm_persist_path = Path(str(configured_path))
            elif self.output_dir is not None:
                self._ltm_persist_path = (
                    self.output_dir / "agent_memory" / f"{self.agent_id}_ltm.json"
                )
        except Exception:
            self._ltm_persist_path = None

        self._ltm_entries: List[Dict[str, Any]] = []
        self._ltm_index: Dict[str, int] = {}
        if self._ltm_enabled:
            self._load_ltm_from_disk()

        # ──────────────────────────────────────────────────────────────
        # 经历/空间变化记忆（Episodic Memory）
        # 目标：保存“刚走过的经历/空间变化轨迹”（如：刚转弯、刚进入楼梯间、刚发生碰撞）。
        # 用途：
        # - 让 VLM 在 prompt 中看到最近经历，减少原地转圈/反复探索
        # - 为后续调试提供可落盘的时序信息
        # ──────────────────────────────────────────────────────────────
        epi_cfg = (
            (agent_cfg.get("episodic_memory", {}) or {})
            if isinstance(agent_cfg, dict)
            else {}
        )
        self._epi_enabled = bool(epi_cfg.get("enabled", True))
        try:
            self._epi_max_entries = int(epi_cfg.get("max_entries", 120))
        except Exception:
            self._epi_max_entries = 120
        self._epi_max_entries = max(20, self._epi_max_entries)

        try:
            self._epi_max_chars_in_prompt = int(epi_cfg.get("max_chars_in_prompt", 900))
        except Exception:
            self._epi_max_chars_in_prompt = 900
        self._epi_max_chars_in_prompt = max(200, self._epi_max_chars_in_prompt)

        self._epi_persist_path: Optional[Path] = None
        try:
            configured_path = epi_cfg.get("persist_path")
            if configured_path:
                self._epi_persist_path = Path(str(configured_path))
            elif self.output_dir is not None:
                self._epi_persist_path = (
                    self.output_dir / "agent_memory" / f"{self.agent_id}_episodic.json"
                )
        except Exception:
            self._epi_persist_path = None

        self._epi_entries: List[Dict[str, Any]] = []
        if self._epi_enabled:
            self._load_episodic_from_disk()

        # 从配置读取VLM参数
        self.vlm_url = self._vlm_config.get("api", {}).get(
            "endpoint", "http://localhost:8000/v1/chat/completions"
        )
        self.model_name = self._vlm_config.get("model", {}).get(
            "name", "./model/Qwen3-VL-8B-Instruct"
        )
        self.timeout = self._vlm_config.get("api", {}).get("timeout", 60)
        self.max_retries = self._vlm_config.get("api", {}).get("max_retries", 3)
        self.max_tokens = self._vlm_config.get("inference", {}).get("max_tokens", 1024)
        self.temperature = self._vlm_config.get("inference", {}).get("temperature", 0.7)

        # 图像编码/压缩配置（用于 DataURL 发送，影响带宽与延迟）
        self._image_cfg = self._vlm_config.get("image_processing", {}) or {}
        self._image_cache: Dict[Any, str] = {}
        self._image_cache_max_entries = int(
            self._image_cfg.get("cache_max_entries", 32)
        )

        # HTTP 会话：连接复用 + 连接池，显著降低远程 VLM 的 RTT/握手开销
        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=0)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def add_memory(self, data: Dict[str, Any]):
        """添加记忆"""
        frame = MemoryFrame(timestamp=datetime.now().isoformat(), data=data)
        self.memory.append(frame)
        logger.debug(f"{self.agent_id} 添加记忆 (缓冲区: {len(self.memory)})")

    def _ltm_make_id(self, kind: str, data: Dict[str, Any]) -> str:
        """为一条关键记忆生成稳定 ID（用于去重）。"""
        try:
            payload = {"kind": str(kind), "data": data}
            raw = json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        except Exception:
            raw = f"{kind}:{str(data)}"
        return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()

    def add_key_memory(
        self,
        *,
        kind: str,
        data: Dict[str, Any],
        importance: float = 0.5,
        pinned: bool = False,
        source: str = "system",
    ) -> None:
        """写入长期关键记忆（去重、重要度、可落盘）。"""
        if not self._ltm_enabled:
            return

        try:
            importance = float(importance)
        except Exception:
            importance = 0.5
        importance = float(np.clip(importance, 0.0, 1.0))

        mem_id = self._ltm_make_id(kind, data)
        now = datetime.now().isoformat()

        entry = {
            "id": mem_id,
            "ts": now,
            "kind": str(kind),
            "importance": float(importance),
            "pinned": bool(pinned),
            "source": str(source),
            "data": data or {},
        }

        if mem_id in self._ltm_index:
            idx = self._ltm_index[mem_id]
            try:
                old = self._ltm_entries[idx]
                entry["importance"] = float(
                    max(old.get("importance", 0.0), entry["importance"])
                )
                entry["pinned"] = bool(old.get("pinned", False) or entry["pinned"])
            except Exception:
                pass
            self._ltm_entries[idx] = entry
        else:
            self._ltm_entries.append(entry)
            self._ltm_index[mem_id] = len(self._ltm_entries) - 1

        self._ltm_prune_if_needed()
        self._save_ltm_to_disk()

    def _ltm_prune_if_needed(self) -> None:
        """超过上限时淘汰：优先保留 pinned，其次保留重要度高/更新更近的。"""
        if len(self._ltm_entries) <= self._ltm_max_entries:
            return

        def score(e: Dict[str, Any]) -> float:
            try:
                imp = float(e.get("importance", 0.0))
            except Exception:
                imp = 0.0
            pin = 1.0 if bool(e.get("pinned", False)) else 0.0
            # 轻微偏向“近期更新”，但不压过重要度
            ts = str(e.get("ts", ""))
            rec = 0.0
            try:
                # 仅用字符串长度/存在性作弱信号，避免 datetime 解析失败
                rec = 0.01 if ts else 0.0
            except Exception:
                rec = 0.0
            return pin * 10.0 + imp * 1.0 + rec

        self._ltm_entries.sort(key=score, reverse=True)
        self._ltm_entries = self._ltm_entries[: self._ltm_max_entries]
        self._ltm_index = {
            e.get("id", ""): i for i, e in enumerate(self._ltm_entries) if e.get("id")
        }

    def _load_ltm_from_disk(self) -> None:
        path = self._ltm_persist_path
        if path is None:
            return
        try:
            if not path.exists():
                return
            obj = json.loads(path.read_text(encoding="utf-8"))
            entries = obj.get("entries")
            if not isinstance(entries, list):
                return
            self._ltm_entries = [e for e in entries if isinstance(e, dict)]
            self._ltm_index = {
                e.get("id", ""): i
                for i, e in enumerate(self._ltm_entries)
                if e.get("id")
            }
            self._ltm_prune_if_needed()
        except Exception as e:
            logger.debug(f"{self.agent_id} LTM加载失败: {e}")

    def _load_episodic_from_disk(self) -> None:
        path = self._epi_persist_path
        if path is None:
            return
        try:
            if not path.exists():
                return
            obj = json.loads(path.read_text(encoding="utf-8"))
            entries = obj.get("entries")
            if not isinstance(entries, list):
                return
            self._epi_entries = [e for e in entries if isinstance(e, dict)]
            self._epi_entries = self._epi_entries[-self._epi_max_entries :]
        except Exception as e:
            logger.debug(f"{self.agent_id} Episodic加载失败: {e}")

    def _save_ltm_to_disk(self) -> None:
        path = self._ltm_persist_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "agent_id": self.agent_id,
                "updated_at": datetime.now().isoformat(),
                "entries": self._ltm_entries,
            }
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            tmp.replace(path)
        except Exception as e:
            logger.debug(f"{self.agent_id} LTM保存失败: {e}")

    def _save_episodic_to_disk(self) -> None:
        path = self._epi_persist_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "agent_id": self.agent_id,
                "updated_at": datetime.now().isoformat(),
                "entries": self._epi_entries[-self._epi_max_entries :],
            }
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            tmp.replace(path)
        except Exception as e:
            logger.debug(f"{self.agent_id} Episodic保存失败: {e}")

    def record_experience(
        self,
        *,
        kind: str,
        data: Dict[str, Any],
        importance: float = 0.3,
        source: str = "system",
    ) -> None:
        """记录一条经历（时序记忆），用于“刚走过的经历/空间变化”注入 prompt。"""
        if not self._epi_enabled:
            return
        try:
            importance = float(importance)
        except Exception:
            importance = 0.3
        importance = float(np.clip(importance, 0.0, 1.0))

        entry = {
            "ts": datetime.now().isoformat(),
            "kind": str(kind),
            "importance": float(importance),
            "source": str(source),
            "data": data or {},
        }
        self._epi_entries.append(entry)
        if len(self._epi_entries) > self._epi_max_entries:
            self._epi_entries = self._epi_entries[-self._epi_max_entries :]
        self._save_episodic_to_disk()

    def get_episodic_prompt_block(self, *, target_room: Optional[str] = None) -> str:
        """返回最近经历摘要（严格控长）。"""
        if not self._epi_enabled or not self._epi_entries:
            return ""

        entries = list(self._epi_entries)

        # 简单检索：若指定 target_room，优先包含提到 target_room 的经历
        selected: List[Dict[str, Any]] = []
        if target_room:
            tr = str(target_room)
            for e in reversed(entries):
                try:
                    if tr and tr in json.dumps(e.get("data", {}), ensure_ascii=False):
                        selected.append(e)
                        if len(selected) >= 2:
                            break
                except Exception:
                    continue

        # 再追加最近 K 条
        for e in reversed(entries[-12:]):
            selected.append(e)
            if len(selected) >= 8:
                break

        # 去重（按 ts+kind 粗略）
        seen = set()
        uniq: List[Dict[str, Any]] = []
        for e in selected:
            k = f"{e.get('ts', '')}|{e.get('kind', '')}"
            if k in seen:
                continue
            seen.add(k)
            uniq.append(e)

        uniq = list(reversed(uniq))  # 时间正序

        lines: List[str] = [
            "【最近经历/空间变化（Episodic Memory，用于理解场景变化与避免原地循环）】"
        ]
        for e in uniq:
            kind = str(e.get("kind", ""))
            data = e.get("data") or {}

            if kind == "executed_action":
                try:
                    lines.append(
                        f"- step={data.get('step')} action={data.get('action')} dist={data.get('dist_moved')} collision={data.get('had_collision')} room={data.get('room_hint')} floor={data.get('floor_hint')}"
                    )
                except Exception:
                    lines.append("- executed_action")
            elif kind == "vlm_decision":
                try:
                    lines.append(
                        f"- vlm_decision: conf={data.get('confidence')} loc={data.get('current_room')}/{data.get('current_floor')} subgoal={data.get('subgoal')} actions={data.get('actions')}"
                    )
                except Exception:
                    lines.append("- vlm_decision")
            else:
                try:
                    lines.append(f"- {kind}: {str(data)}")
                except Exception:
                    lines.append(f"- {kind}")

            joined = "\n".join(lines)
            if len(joined) >= self._epi_max_chars_in_prompt:
                break

        block = "\n".join(lines)
        if len(block) > self._epi_max_chars_in_prompt:
            block = block[: self._epi_max_chars_in_prompt] + "…"
        return block

    def get_route_guidance_prompt_block(self) -> str:
        """返回最近一次“整体指路描述”（来自 LTM），用于 stage2 执行时保持全局一致性。"""
        if not self._ltm_enabled or not self._ltm_entries:
            return ""
        best = None
        for e in reversed(self._ltm_entries):
            if str(e.get("kind")) == "route_guidance":
                best = e
                break
        if not best:
            return ""
        try:
            txt = str((best.get("data") or {}).get("text", "")).strip()
        except Exception:
            txt = ""
        if not txt:
            return ""
        # 控长：单独给它一个较宽松的上限，但仍避免把 prompt 撑爆
        max_chars = 1200
        if len(txt) > max_chars:
            txt = txt[:max_chars] + "…"
        return "【整体导航指路（阶段1生成，用于执行阶段保持全局路线一致）】\n" + txt

    def get_key_memory_prompt_block(self) -> str:
        """返回用于注入 prompt 的关键记忆摘要（严格控长）。"""
        if not self._ltm_enabled or not self._ltm_entries:
            return ""

        # pinned 永远优先；其余按 importance 降序
        pinned = [e for e in self._ltm_entries if bool(e.get("pinned", False))]
        others = [e for e in self._ltm_entries if not bool(e.get("pinned", False))]
        try:
            others.sort(key=lambda e: float(e.get("importance", 0.0)), reverse=True)
        except Exception:
            pass

        selected = pinned + others

        lines: List[str] = []
        lines.append("【长期关键记忆（LTM，仅供参考；用于避免重复犯错/稳定定位）】")
        for e in selected:
            kind = str(e.get("kind", ""))
            imp = e.get("importance")
            data = e.get("data")
            # 尽量生成“短句”，不塞 JSON
            s = ""
            if kind == "planned_path":
                try:
                    s = f"- planned_path: {data.get('path', '')}"
                except Exception:
                    s = "- planned_path: (unavailable)"
            elif kind == "confirmed_location":
                try:
                    s = f"- confirmed_location: room={data.get('room')} floor={data.get('floor')} evidence={data.get('evidence', '')}"
                except Exception:
                    s = "- confirmed_location: (unavailable)"
            elif kind == "collision_hotspot":
                try:
                    s = f"- collision_hotspot: action={data.get('action')} consecutive={data.get('consecutive')} note={data.get('note', '')}"
                except Exception:
                    s = "- collision_hotspot: (unavailable)"
            elif kind == "last_subgoal":
                try:
                    s = f"- last_subgoal: {data.get('subgoal', '')}"
                except Exception:
                    s = "- last_subgoal: (unavailable)"
            else:
                try:
                    s = f"- {kind}: {str(data)}"
                except Exception:
                    s = f"- {kind}"

            if imp is not None:
                try:
                    s = f"{s} (imp={float(imp):.2f})"
                except Exception:
                    pass
            lines.append(s)

            joined = "\n".join(lines)
            if len(joined) >= self._ltm_max_chars_in_prompt:
                break

        block = "\n".join(lines)
        if len(block) > self._ltm_max_chars_in_prompt:
            block = block[: self._ltm_max_chars_in_prompt] + "…"
        return block

    def get_memory_summary(self) -> str:
        """获取记忆摘要"""
        if not self.memory:
            return "无记忆"

        summary = f"{self.agent_id} 最近 {len(self.memory)} 帧记忆:\n"
        for i, frame in enumerate(list(self.memory)[-3:]):
            summary += f"  [{i}] {frame.timestamp}: {list(frame.data.keys())}\n"
        return summary

    def clear_memory(self):
        """清除记忆"""
        self.memory.clear()
        logger.info(f"{self.agent_id} 记忆已清除")

    def call_vlm(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        trace_meta: Optional[Dict[str, Any]] = None,
        vlm_url: str = None,
        model_name: str = None,
        timeout: int = None,
        max_retries: int = None,
        max_tokens: int = None,
        temperature: float = None,
    ) -> Optional[str]:
        """
        调用VLM模型

        Args:
            prompt: 文本提示词
            images: 图片Data URL列表
            vlm_url: VLM API地址（None则使用配置）
            model_name: VLM模型名称（None则使用配置）
            timeout: 请求超时时间（None则使用配置）
            max_retries: 最大重试次数（None则使用配置）
            max_tokens: 最大token数（None则使用配置）
            temperature: 采样温度（None则使用配置）

        Returns:
            模型输出或None
        """
        # 使用配置中的默认值
        vlm_url = vlm_url or self.vlm_url
        model_name = model_name or self.model_name
        timeout = timeout or self.timeout
        max_retries = max_retries or self.max_retries
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        headers = {"Content-Type": "application/json", "Connection": "keep-alive"}

        # 构建消息内容
        content = [{"type": "text", "text": prompt}]
        if images:
            for img_url in images:
                content.append({"type": "image_url", "image_url": {"url": img_url}})

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # 超时：允许分别配置 connect/read（read 默认沿用 timeout）
        connect_timeout = self._vlm_config.get("api", {}).get("connect_timeout", 10)
        request_timeout = (
            float(connect_timeout),
            float(timeout),
        )

        # 重试机制 + 指数退避
        backoff = 0.5
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"{self.agent_id} VLM调用 (尝试 {attempt + 1}/{max_retries})"
                )
                t0 = time.time()
                post_fn = self._session.post
                try:
                    from unittest.mock import Mock

                    # tests 常 patch requests.post；此时用它以便拦截/断言
                    if isinstance(getattr(requests, "post", None), Mock):
                        post_fn = requests.post
                except Exception:
                    pass

                response = post_fn(
                    vlm_url, headers=headers, json=payload, timeout=request_timeout
                )
                elapsed = time.time() - t0

                if response.status_code == 200:
                    try:
                        result = response.json()
                        output = result["choices"][0]["message"]["content"]
                    except Exception as e:
                        logger.warning(f"⚠️ VLM响应解析失败: {e}")
                        output = None

                    if not output:
                        logger.warning("⚠️ VLM返回空内容")
                        continue

                    logger.info(f"✅ {self.agent_id} VLM响应成功 (耗时 {elapsed:.2f}s)")
                    self.success_count += 1

                    # 保存VLM输入输出到文件/trace
                    self._save_vlm_io(
                        prompt,
                        images,
                        output,
                        attempt + 1,
                        meta={
                            "vlm_url": vlm_url,
                            "model": model_name,
                            "timeout": {
                                "connect": float(connect_timeout),
                                "read": float(timeout),
                            },
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "http_status": int(response.status_code),
                            "elapsed_s": float(elapsed),
                            "trace_meta": trace_meta or {},
                        },
                    )

                    return output
                else:
                    logger.warning(
                        f"⚠️ VLM返回 {response.status_code} (耗时 {elapsed:.2f}s)"
                    )

                    self._save_vlm_io(
                        prompt,
                        images,
                        "",
                        attempt + 1,
                        meta={
                            "vlm_url": vlm_url,
                            "model": model_name,
                            "timeout": {
                                "connect": float(connect_timeout),
                                "read": float(timeout),
                            },
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "http_status": int(response.status_code),
                            "elapsed_s": float(elapsed),
                            "error": f"http_status_{response.status_code}",
                            "trace_meta": trace_meta or {},
                        },
                    )

            except requests.exceptions.Timeout:
                logger.warning(
                    f"⚠️ VLM请求超时 (connect={connect_timeout}s, read={timeout}s)"
                )
                self._save_vlm_io(
                    prompt,
                    images,
                    "",
                    attempt + 1,
                    meta={
                        "vlm_url": vlm_url,
                        "model": model_name,
                        "timeout": {
                            "connect": float(connect_timeout),
                            "read": float(timeout),
                        },
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "error": "timeout",
                        "trace_meta": trace_meta or {},
                    },
                )
            except Exception as e:
                logger.warning(f"⚠️ VLM调用异常: {e}")
                self._save_vlm_io(
                    prompt,
                    images,
                    "",
                    attempt + 1,
                    meta={
                        "vlm_url": vlm_url,
                        "model": model_name,
                        "timeout": {
                            "connect": float(connect_timeout),
                            "read": float(timeout),
                        },
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "error": str(e),
                        "trace_meta": trace_meta or {},
                    },
                )

            # 指数退避
            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2

        logger.error(f"❌ {self.agent_id} VLM调用在 {max_retries} 次尝试后失败")
        self.failure_count += 1
        return None

    def _image_to_data_url(self, image: np.ndarray, *, kind: str = "rgb") -> str:
        """将 numpy 图像转换为 data:image/jpeg;base64,...

        - kind='rgb': 常规观测图像
        - kind='floorplan': 平面图（通常可更小/更高压缩）
        """
        if image is None:
            return ""

        # 读取配置
        enabled = bool(self._image_cfg.get("enabled", True))
        max_size = self._image_cfg.get("max_size", [600, 600])
        quality = int(self._image_cfg.get("quality", 85))
        if kind == "floorplan":
            max_size = self._image_cfg.get("floorplan_max_size", max_size)
            quality = int(self._image_cfg.get("floorplan_quality", quality))

        # 兼容：enabled=false 表示尽量少改动尺寸，但仍需 JPEG 编码
        if not enabled:
            max_size = None

        # 仅对可复用的大图（如平面图）做缓存，避免重复 base64 编码
        cache_key = None
        if kind == "floorplan":
            try:
                cache_key = (
                    id(image),
                    tuple(image.shape),
                    str(image.dtype),
                    kind,
                    str(max_size),
                    quality,
                )
                if cache_key in self._image_cache:
                    return self._image_cache[cache_key]
            except Exception:
                cache_key = None

        img = image
        if not isinstance(img, np.ndarray):
            return ""

        # 优先用 OpenCV（更快）。没有 cv2 时回退到 Pillow。
        try:
            import cv2  # type: ignore

            # 统一通道：灰度/带 alpha
            if len(img.shape) == 2:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 4:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            else:
                img_rgb = img

            # OpenCV 编码期望 BGR；输入约定为 RGB
            if len(img_rgb.shape) == 3 and img_rgb.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_rgb

            # 缩放（保持长宽比）
            if max_size is not None:
                try:
                    max_h, max_w = int(max_size[0]), int(max_size[1])
                    h, w = img_bgr.shape[:2]
                    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
                    if scale < 1.0:
                        new_w = max(1, int(w * scale))
                        new_h = max(1, int(h * scale))
                        img_bgr = cv2.resize(
                            img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA
                        )
                except Exception:
                    pass

            encode_params = [
                int(cv2.IMWRITE_JPEG_QUALITY),
                int(np.clip(int(quality), 1, 100)),
            ]
            ok, buf = cv2.imencode(".jpg", img_bgr, encode_params)
            if not ok:
                return ""

            img_data = base64.b64encode(buf.tobytes()).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{img_data}"
        except Exception:
            try:
                from io import BytesIO

                from PIL import Image  # type: ignore

                arr = img
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)

                if len(arr.shape) == 2:
                    pil_img = Image.fromarray(arr, mode="L").convert("RGB")
                elif len(arr.shape) == 3 and arr.shape[2] == 4:
                    pil_img = Image.fromarray(arr, mode="RGBA").convert("RGB")
                else:
                    pil_img = Image.fromarray(arr, mode="RGB")

                if max_size is not None:
                    max_h, max_w = int(max_size[0]), int(max_size[1])
                    try:
                        resample = Image.Resampling.LANCZOS  # Pillow>=9
                    except Exception:
                        resample = Image.LANCZOS
                    pil_img.thumbnail((max_w, max_h), resample=resample)

                buf = BytesIO()
                pil_img.save(
                    buf,
                    format="JPEG",
                    quality=int(np.clip(int(quality), 1, 100)),
                    optimize=True,
                )
                img_data = base64.b64encode(buf.getvalue()).decode("utf-8")
                data_url = f"data:image/jpeg;base64,{img_data}"
            except Exception:
                return ""

        if cache_key is not None:
            # 简单容量控制
            if len(self._image_cache) >= self._image_cache_max_entries:
                self._image_cache.pop(next(iter(self._image_cache)))
            self._image_cache[cache_key] = data_url

        return data_url

    def _depth_to_vis_rgb(self, depth: np.ndarray) -> np.ndarray:
        """将深度图转换为可供VLM理解的可视化RGB图（灰度3通道）。

        - 输入: depth (H,W) 或 (H,W,1)，通常为 float32（米）
        - 输出: uint8 (H,W,3)
        - 可视化：默认“近亮远暗”（invert=true），更符合直觉
        """
        if depth is None or not isinstance(depth, np.ndarray) or depth.size == 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        d = depth
        try:
            if d.ndim == 3 and d.shape[-1] == 1:
                d = d[:, :, 0]
        except Exception:
            pass

        try:
            d = d.astype(np.float32, copy=False)
        except Exception:
            d = np.array(d, dtype=np.float32)

        depth_cfg = {}
        try:
            if isinstance(self._image_cfg, dict):
                depth_cfg = self._image_cfg.get("depth", {}) or {}
        except Exception:
            depth_cfg = {}

        try:
            max_depth_m = float(depth_cfg.get("max_depth_m", 10.0))
        except Exception:
            max_depth_m = 10.0
        max_depth_m = max(0.1, max_depth_m)

        try:
            invert = bool(depth_cfg.get("invert", True))
        except Exception:
            invert = True

        d2 = d.copy()
        try:
            invalid = ~np.isfinite(d2)
            invalid |= d2 <= 0
            d2[invalid] = max_depth_m
        except Exception:
            pass

        d2 = np.clip(d2, 0.0, max_depth_m)
        norm = d2 / max_depth_m
        if invert:
            norm = 1.0 - norm

        vis = (np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)
        return np.repeat(vis[:, :, None], 3, axis=2)

    def _save_vlm_io(
        self,
        prompt: str,
        images: Optional[List[str]],
        output: str,
        attempt: int,
        meta: Optional[Dict[str, Any]] = None,
    ):
        """
        保存VLM输入和输出到文件

        Args:
            prompt: 输入提示词
            images: 图像URL列表
            output: VLM输出
            attempt: 尝试次数
        """
        if not self.output_dir and not self._vlm_trace_path:
            return

        meta = meta or {}

        # 注意：为避免单文件过大，这里默认不写入完整 data-url base64 内容。
        images_meta: List[Dict[str, Any]] = []
        if images:
            for idx, img in enumerate(images):
                if not img:
                    continue
                images_meta.append(
                    {
                        "index": idx,
                        "data_url_len": len(img),
                        "mime": "image/jpeg"
                        if img.startswith("data:image/jpeg")
                        else "unknown",
                    }
                )

        record = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            "attempt": attempt,
            "input": {
                "prompt": prompt,
                "num_images": len(images) if images else 0,
                "images_meta": images_meta,
            },
            "output": {
                "text": output,
            },
            "meta": meta,
        }

        # 统一写入单文件 trace（默认: output_dir/vlm_io.json）；不再创建 vlm_inputs/vlm_outputs 目录
        trace_path: Optional[Path] = None
        if self._vlm_trace_path is not None:
            trace_path = self._vlm_trace_path
        elif self.output_dir is not None:
            trace_path = self.output_dir / "vlm_io.json"

        if trace_path is None:
            return

        try:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._append_json_record(trace_path, record)
        except Exception as e:
            logger.warning(f"⚠️ 写入 VLM trace 失败: {e}")

    @staticmethod
    def _append_json_record(path: Path, record: Dict[str, Any]) -> None:
        """向单个 JSON 文件追加一条记录。

        文件格式：{"records": [ ... ]}
        """
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        else:
            existing = {}

        records = existing.get("records")
        if not isinstance(records, list):
            records = []
        records.append(record)
        existing["records"] = records

        tmp_path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp_path.replace(path)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "agent_id": self.agent_id,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "memory_size": len(self.memory),
            "last_action": self.last_action,
        }


class Agent1(VLMAgent):
    """
    智能体1 - 动作控制
    功能：根据RGB和平面图输出单步动作
    """

    def __init__(
        self,
        memory_size: int = None,
        config_dir: str = "./configs",
        output_dir: str = None,
        vlm_trace_path: str = None,
    ):
        super().__init__("Agent1", memory_size, config_dir, output_dir, vlm_trace_path)
        self.action_history: List[str] = []
        try:
            self.action_steps = int(
                (self._agent_config.get("agent1", {}) or {}).get("action_steps", 5)
            )
        except Exception:
            self.action_steps = 5
        self.action_steps = max(1, self.action_steps)

    def generate_action_batch(
        self,
        rgb_image: np.ndarray,
        floorplan: np.ndarray,
        current_room: str,
        target_room: str,
        path_rooms: List[str] = None,
        context: Dict[str, Any] = None,
        *,
        rgb_images: List[np.ndarray] = None,
        depth_images: List[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        生成动作批次（N步，N 由配置 agents.agent1.action_steps 决定）

        Args:
            rgb_image: 前置RGB图像 (H, W, 3)
            floorplan: 楼层平面图 (H, W, 3)
            current_room: 当前房间ID
            target_room: 目标房间ID
            path_rooms: 规划的路径房间序列
            context: 额外上下文信息

        Returns:
            {
                "actions": List[str],  # 动作列表 ["move_forward", "turn_left", ...]
                "reached_goal": bool,  # 是否到达目标房间
                "reasoning": str  # VLM推理过程
            } 或 None
        """
        # 输入策略：使用“最后一次动作后的观测”单帧（t）。
        # 为兼容旧调用方：若传入多帧，默认只取最后一帧作为 t。
        if rgb_images is None:
            rgb_images = [rgb_image] if rgb_image is not None else []
        rgb_images = [
            img for img in rgb_images if isinstance(img, np.ndarray) and img.size > 0
        ]
        if not rgb_images and rgb_image is not None:
            rgb_images = [rgb_image]
        if len(rgb_images) > 1:
            rgb_images = [rgb_images[-1]]

        # 可选 Depth：同样只取最后一帧（t）
        if depth_images is not None:
            depth_images = [
                d for d in depth_images if isinstance(d, np.ndarray) and d.size > 0
            ]
            if len(depth_images) > 1:
                depth_images = [depth_images[-1]]

        # 转换图像为Data URL
        rgb_urls = [self._image_to_data_url(img, kind="rgb") for img in rgb_images]
        depth_urls: List[str] = []
        if depth_images:
            try:
                depth_vis = [self._depth_to_vis_rgb(d) for d in depth_images]
                depth_urls = [
                    self._image_to_data_url(img, kind="rgb") for img in depth_vis
                ]
            except Exception:
                depth_urls = []
        floorplan_url = self._image_to_data_url(floorplan, kind="floorplan")

        # 路径信息
        path_str = " → ".join(path_rooms) if path_rooms else "未规划"
        memory_summary = self.get_memory_summary()
        ltm_block = self.get_key_memory_prompt_block()
        route_guidance_block = self.get_route_guidance_prompt_block()
        episodic_block = self.get_episodic_prompt_block(target_room=target_room)

        ctx = context or {}
        collision_info = ctx.get("collision") or {}
        last_actions = ctx.get("last_actions") or []
        action_counts = ctx.get("action_counts") or {}
        next_room_hint = ctx.get("next_room_hint")
        path_index_hint = ctx.get("path_index_hint")
        remaining_path_rooms = ctx.get("remaining_path_rooms")
        target_floor_hint = ctx.get("target_floor_hint")
        stairs_ctx = ctx.get("stairs") or {}

        # 从 LTM 中取“最近一次 subgoal”，用于让模型保持短期一致性（减少来回改策略/转圈）
        last_subgoal_hint = None
        try:
            for e in reversed(getattr(self, "_ltm_entries", []) or []):
                if str(e.get("kind")) == "last_subgoal":
                    last_subgoal_hint = (e.get("data") or {}).get("subgoal")
                    if last_subgoal_hint:
                        last_subgoal_hint = str(last_subgoal_hint).strip()
                    break
        except Exception:
            last_subgoal_hint = None

        # 将“规划路径”写入长期记忆（pinned，避免每步都依赖外部变量）
        try:
            if path_rooms:
                self.add_key_memory(
                    kind="planned_path",
                    data={"path": " → ".join(path_rooms)},
                    importance=0.95,
                    pinned=True,
                    source="planner",
                )
        except Exception:
            pass
        collision_hint = ""
        try:
            if bool(collision_info.get("had_collision")):
                collision_hint = (
                    "\n    【上一步反馈】\n"
                    f"    - 疑似碰撞/卡住: action={collision_info.get('action')} "
                    f"dist_moved={collision_info.get('dist_moved')} "
                    f"consecutive={collision_info.get('consecutive')}\n"
                )
        except Exception:
            collision_hint = ""

        stairs_hint = ""
        try:
            if isinstance(stairs_ctx, dict) and stairs_ctx:
                stairs_hint = (
                    "\n    【楼梯阶段状态（系统提示）】\n"
                    f"    - in_stairwell: {bool(stairs_ctx.get('in_stairwell'))}\n"
                    f"    - floor_anchor(for floorplan): {stairs_ctx.get('floor_anchor')}\n"
                    f"    - floorplan_floor_used: {stairs_ctx.get('floorplan_floor_used')}\n"
                    f"    - next_room_hint: {stairs_ctx.get('next_room_hint')}\n"
                )
        except Exception:
            stairs_hint = ""

        last_actions_text = "无"
        try:
            if isinstance(last_actions, list) and last_actions:
                last_actions_text = ", ".join([str(a) for a in last_actions[-12:]])
        except Exception:
            last_actions_text = "无"

        action_counts_text = ""
        try:
            if isinstance(action_counts, dict) and action_counts:
                action_counts_text = (
                    f"move_forward={action_counts.get('move_forward', 0)}, "
                    f"move_backward={action_counts.get('move_backward', 0)}, "
                    f"turn_left={action_counts.get('turn_left', 0)}, "
                    f"turn_right={action_counts.get('turn_right', 0)}"
                )
        except Exception:
            action_counts_text = ""

        # 从配置注入“平面图图例/符号语义”，让 VLM 稳定解读地图标注
        floorplan_legend = ""
        try:
            floorplan_legend = (
                (self._prompts_config.get("system", {}) or {}).get(
                    "floorplan_legend", ""
                )
                or ""
            ).strip()
        except Exception:
            floorplan_legend = ""
        floorplan_legend_block = ""
        if floorplan_legend:
            try:
                indented_lines: List[str] = []
                for line in floorplan_legend.splitlines():
                    if line.strip():
                        indented_lines.append(f"    {line}")
                    else:
                        indented_lines.append("")
                floorplan_legend_block = "\n" + "\n".join(indented_lines) + "\n"
            except Exception:
                floorplan_legend_block = f"\n    {floorplan_legend}\n"

        # 进一步注入“协议化”提示：Map-first + 自检 + 置信度标定 + reasoning 模板
        protocol_text = ""
        confidence_text = ""
        reasoning_template_text = ""
        sim_scene_text = ""
        stage2_policy_text = ""
        memory_usage_text = ""
        anti_loop_text = ""
        try:
            sys_prompts = self._prompts_config.get("system", {}) or {}
            protocol_text = (
                sys_prompts.get("floorplan_navigation_protocol", "") or ""
            ).strip()
            confidence_text = (
                sys_prompts.get("confidence_calibration", "") or ""
            ).strip()
            reasoning_template_text = (
                sys_prompts.get("navigation_output_template", "") or ""
            ).strip()
            # 仅注入“短声明/核心策略”，避免把 prompts.system.navigation 整段重复塞进来造成冲突
            sim_scene_text = (
                sys_prompts.get("habitat_scene_declaration", "") or ""
            ).strip()
            stage2_policy_text = (
                sys_prompts.get("stage2_navigation_policy", "") or ""
            ).strip()
            memory_usage_text = (
                sys_prompts.get("memory_usage_policy", "") or ""
            ).strip()
            anti_loop_text = (sys_prompts.get("anti_loop_policy", "") or "").strip()
        except Exception:
            protocol_text = ""
            confidence_text = ""
            reasoning_template_text = ""
            sim_scene_text = ""
            stage2_policy_text = ""
            memory_usage_text = ""
            anti_loop_text = ""

        def _indent_block(text: str) -> str:
            if not text:
                return ""
            lines: List[str] = []
            for line in text.splitlines():
                if line.strip():
                    lines.append(f"    {line}")
                else:
                    lines.append("")
            return "\n" + "\n".join(lines) + "\n"

        protocol_block = _indent_block(protocol_text)
        confidence_block = _indent_block(confidence_text)
        reasoning_template_block = _indent_block(reasoning_template_text)
        # 注意：prompts.system.navigation 本身很长，但其中包含“Habitat虚拟仿真声明”。
        # 这里仅用于抽取其关键声明与约束（由配置维护）。
        sim_scene_block = _indent_block(sim_scene_text)
        stage2_policy_block = _indent_block(stage2_policy_text)
        memory_usage_block = _indent_block(memory_usage_text)
        anti_loop_block = _indent_block(anti_loop_text)

        # 额外进度信号：最近动作“转向 vs 位移”比例，用于触发更自然的“转向后前进”
        progress_hint = ""
        try:
            ra = list(last_actions or [])[-12:]
            turn_n = sum(1 for a in ra if a in ("turn_left", "turn_right"))
            move_n = sum(1 for a in ra if a in ("move_forward", "move_backward"))
            progress_hint = (
                "\n    【进度信号（系统统计）】\n"
                f"    - recent_turns(last12): {turn_n}\n"
                f"    - recent_moves(last12): {move_n}\n"
                f"    - last_subgoal_hint(LTM): {last_subgoal_hint}\n"
            )
        except Exception:
            progress_hint = ""

        stairs_policy_text = ""
        try:
            sys_prompts = self._prompts_config.get("system", {}) or {}
            stairs_policy_text = (
                sys_prompts.get("stair_traversal_policy", "") or ""
            ).strip()
        except Exception:
            stairs_policy_text = ""
        stairs_policy_block = _indent_block(stairs_policy_text)

        # 构建提示词（批量决策）
        prompt = f"""你是一个机器狗(Spot)导航控制智能体。

    你将看到图像输入（按顺序）：
    1) 前置RGB摄像头视图（时间 t；即“最后一次动作后的观测”）
    2) （可选）前置Depth视图（时间 t；灰度可视化：近亮远暗，用于辅助判断距离/障碍）
    3) 当前楼层的平面图（用于辅助定位）

{sim_scene_block}

{stage2_policy_block}

{memory_usage_block}

{anti_loop_block}

    【重要约束】
    - 不能假设“每一步就进入路径中的下一个房间”。当前房间与楼层必须基于视觉信息+平面图综合判断。
    - 只有当你非常确信已经到达目标房间 {target_room} 时，才可以结束导航。
    - 如果你不确定当前位置/楼层，请降低 confidence，并优先选择安全探索动作（转向）而不是盲目前进。
    - 如果上一步疑似碰撞/卡住，请避免重复同一个前进/后退动作，先转向重新观察。
    - 楼梯/跨楼层：机器狗可以走楼梯；只有走完一个完整楼梯间（两跑楼梯）并到达下一层落地/可进入走廊时，才更新 current_floor。

    【当前状态（仅作提示，可能不准确）】
    - 当前房间提示: {current_room}
    - 目标房间: {target_room}
    - 目标楼层提示(target_floor_hint): {target_floor_hint}
    - 规划路径（仅作参考）: {path_str}
    - 路径进度提示(path_index_hint): {path_index_hint}
    - 下一房间提示(next_room_hint): {next_room_hint}
    - 剩余路径房间(remaining_path_rooms): {remaining_path_rooms}
    {collision_hint}
    {stairs_hint}
    {progress_hint}

    【最近动作（用于避免震荡）】
    - last_actions: {last_actions_text}
    - action_counts: {action_counts_text}

{floorplan_legend_block}

{protocol_block}

{confidence_block}

{reasoning_template_block}

{stairs_policy_block}

    【如何充分利用平面图（请严格执行）】
    1) 先看 floorplan：识别你可能所在的“走廊/开阔区/房间”形状、门洞位置、拐角、T字路口等结构。
    2) 再看当前前置视图（以及可选Depth）：判断面前是否可通行、是否接近墙体/障碍、是否对齐门洞/走廊方向。
    3) 在 floorplan 上选一个“短期子目标”（subgoal）：这是你下一步必须到达的结构点（优先：门洞D、走廊拐角/路口、楼梯平台/进入楼梯间S）。
    4) 规划接下来 {self.action_steps} 步动作：优先“对齐走廊方向(转向)”再“稳定前进(多步 forward)”。
    5) 不确定位置时：先用 2~4 次小角度转向获取更多信息，再前进；不要盲目连续前进。

        【输出要求（用 reasoning 强制体现“平面图规划”）】
        - reasoning 必须是“单行字符串”，建议按模板用“;”或“|”分隔字段。
        - reasoning 必须包含：定位假设(房间/楼层)、地图证据(map_evidence)、t-1/t变化(vision_delta)、subgoal、动作与subgoal对应(plan)、风险自检(risk)。

    【稳定性约束（非常重要）】
    - 避免左右来回震荡：不要输出 turn_left,turn_right,turn_left... 这种序列。
    - 避免反复撞墙：如果连续碰撞(consecutive>=2)或视觉像“贴近墙/障碍”，先转向再前进。
    - 若 floorplan 与视觉矛盾，以 floorplan 的结构约束为准，选择能让你回到走廊/路口的动作。

    【机器狗规格】
    - 身高: 0.55米 | 宽度: 0.28米
    - 单步前进: 0.25米 | 单次转弯: 10°

    【可用动作】
    - move_forward: 前进0.25米
    - move_backward: 后退0.25米
    - turn_left: 左转10°
    - turn_right: 右转10°
    - stop: 停止（仅当 done=true 时允许）

    【你要输出的信息】
    1) 你判断的当前位置：current_room / current_floor
    2) 你对位置判断的置信度：confidence（0~1）
    3) 是否结束导航：done
    4) 你下一步必须到达的结构性子目标：subgoal（例如：穿过门洞D、到达走廊拐角/路口、进入楼梯间并到达平台）
    5) 若 done=false：输出接下来的{self.action_steps}步动作序列（actions 长度={self.action_steps}，且不得包含 stop）
    6) 若 done=true：actions 必须为 ["stop"]

    【输出格式】（严格 JSON；只输出 JSON，不要输出其它文字）
    {{
      "done": true/false,
      "reached_goal": true/false,
      "current_room": "...",
      "current_floor": 1,
      "confidence": 0.0,
            "subgoal": "...",
            "actions": ["move_forward", "turn_left", "move_backward", "..."],
            "reasoning": "单行字符串：建议按模板输出 loc/floor/map_evidence/vision/subgoal/plan/risk"
    }}

    【一致性要求】
    - reached_goal 必须与 done 保持一致（两者同真或同假）。
    - done=true => actions=["stop"]。
    - done=false => actions 长度必须为 {self.action_steps}，且不允许 stop。

    【历史记忆】
    {memory_summary}

    {ltm_block}

    {route_guidance_block}

    {episodic_block}
    """

        # 调用VLM：单帧 RGB(t) +（可选）Depth(t) + floorplan
        images = [*rgb_urls]
        if depth_urls:
            images.extend(depth_urls)
        images.append(floorplan_url)
        trace_meta = {
            "kind": "action_generation",
            "current_room_hint": current_room,
            "target_room": target_room,
            "planned_path_rooms": path_rooms or [],
            "context": ctx,
            "rgb_frames": len(rgb_urls),
            "depth_frames": len(depth_urls),
            "action_steps": int(self.action_steps),
        }
        response = self.call_vlm(prompt, images, trace_meta=trace_meta)

        if not response:
            logger.error("❌ Agent1 无法生成动作")
            return None

        # 解析响应 - JSON格式
        try:
            # 清理响应（去除可能的markdown标记）
            response_clean = response.strip()
            if response_clean.startswith("```"):
                response_clean = response_clean.split("\n", 1)[1]
            if response_clean.endswith("```"):
                response_clean = response_clean.rsplit("\n", 1)[0]
            response_clean = response_clean.strip()

            # 解析JSON
            data = json.loads(response_clean)

            reached_goal = bool(data.get("reached_goal", False))
            done = bool(data.get("done", False))
            actions = data.get("actions", [])
            if isinstance(actions, str):
                actions = [actions]

            reasoning = data.get("reasoning", "无推理")
            conf_val = None
            try:
                if data.get("confidence") is not None:
                    conf_val = float(data.get("confidence"))
            except Exception:
                conf_val = None

            # subgoal：优先使用结构化字段；否则回退到从 reasoning 提取
            subgoal_text = None
            try:
                if data.get("subgoal") is not None:
                    subgoal_text = str(data.get("subgoal")).strip()
            except Exception:
                subgoal_text = None
            if not subgoal_text:
                try:
                    m = re.search(r"subgoal\s*=\s*([^;|]+)", str(reasoning))
                    if m:
                        subgoal_text = m.group(1).strip()
                except Exception:
                    subgoal_text = None

            # 验证动作
            valid_actions = {
                "move_forward",
                "move_backward",
                "turn_left",
                "turn_right",
                "stop",
            }
            cleaned_actions = []
            for action in actions:
                action = action.strip().lower()
                if action not in valid_actions:
                    # 尝试标准化
                    if "forward" in action or "前进" in action:
                        action = "move_forward"
                    elif "back" in action or "后" in action:
                        action = "move_backward"
                    elif "left" in action or "左" in action:
                        action = "turn_left"
                    elif "right" in action or "右" in action:
                        action = "turn_right"
                    elif "stop" in action or "停" in action:
                        action = "stop"
                    else:
                        action = "move_forward"  # 默认前进
                cleaned_actions.append(action)

            # 严格一致性：done / reached_goal 同步；stop 只在结束时允许
            should_stop = bool(done or reached_goal)
            if should_stop:
                cleaned_actions = ["stop"]
                done = True
                reached_goal = True
            else:
                # 过滤 stop，并把序列补齐/截断到 action_steps
                cleaned_actions = [a for a in cleaned_actions if a != "stop"]

                # 选择一个“优先转向方向”，避免兜底逻辑长期固定 turn_left 导致原地打转。
                # 规则：根据最近动作里的左右转计数做平衡；若最近明显偏左，则优先向右，反之亦然。
                def _preferred_turn_direction(recent_actions: Any) -> str:
                    try:
                        ra = list(recent_actions or [])
                    except Exception:
                        ra = []
                    ra = ra[-8:]
                    left_n = sum(1 for a in ra if a == "turn_left")
                    right_n = sum(1 for a in ra if a == "turn_right")
                    if left_n - right_n >= 3:
                        return "turn_right"
                    if right_n - left_n >= 3:
                        return "turn_left"
                    # 若最后一步就是转向，则沿用最后方向（更稳定）
                    try:
                        if ra and ra[-1] in ("turn_left", "turn_right"):
                            return ra[-1]
                    except Exception:
                        pass
                    return "turn_left"

                preferred_turn = _preferred_turn_direction(
                    (ctx or {}).get("last_actions")
                )
                opposite_turn = (
                    "turn_right" if preferred_turn == "turn_left" else "turn_left"
                )

                # 低置信度/连续碰撞时：强制先探索（转向），提升后续定位质量
                try:
                    consecutive = int(collision_info.get("consecutive", 0) or 0)
                except Exception:
                    consecutive = 0
                had_collision = False
                try:
                    had_collision = bool(collision_info.get("had_collision"))
                except Exception:
                    had_collision = False

                # 低置信度/连续碰撞时：优先“扫描 + 轻微位移”以获取视差，避免一直单向原地转。
                if consecutive >= 2:
                    # 疑似卡住：先转向，再后退脱困，再尝试前进。
                    cleaned_actions = [
                        preferred_turn,
                        preferred_turn,
                        "move_backward",
                        opposite_turn,
                        "move_forward",
                    ]
                elif conf_val is not None and conf_val < 0.35:
                    if had_collision:
                        # 刚碰撞过时，先转向重新对齐，再尝试位移
                        cleaned_actions = [
                            preferred_turn,
                            preferred_turn,
                            preferred_turn,
                            opposite_turn,
                            preferred_turn,
                        ]
                    else:
                        cleaned_actions = [
                            preferred_turn,
                            preferred_turn,
                            "move_forward",
                            opposite_turn,
                            "move_forward",
                        ]

                # 反转圈循环（硬策略）：
                # 若最近动作几乎都是转向且没有有效位移/无碰撞，则强制加入位移动作，避免一直原地转。
                try:
                    ra = list((ctx or {}).get("last_actions") or [])[-12:]
                    turn_n = sum(1 for a in ra if a in ("turn_left", "turn_right"))
                    move_n = sum(
                        1 for a in ra if a in ("move_forward", "move_backward")
                    )
                    same_dir = False
                    if len(ra) >= 6:
                        tail = ra[-6:]
                        same_dir = all(a == "turn_left" for a in tail) or all(
                            a == "turn_right" for a in tail
                        )

                    cleaned_turn_only = bool(cleaned_actions) and all(
                        a in ("turn_left", "turn_right") for a in cleaned_actions
                    )

                    # 典型转圈：最近 12 步里转向>=9 且移动<=1，且当前计划也是纯转向
                    if (
                        consecutive == 0
                        and not had_collision
                        and turn_n >= 9
                        and move_n <= 1
                        and cleaned_turn_only
                    ):
                        cleaned_actions = [
                            opposite_turn,
                            "move_forward",
                            "move_forward",
                        ]
                    # 长期同向转（例如一直左转）：强制换向 + 位移
                    elif (
                        consecutive == 0
                        and not had_collision
                        and same_dir
                        and move_n == 0
                        and cleaned_turn_only
                    ):
                        cleaned_actions = [
                            opposite_turn,
                            "move_forward",
                            preferred_turn,
                        ]
                except Exception:
                    pass

                # 抗震荡：避免 turn_left/turn_right 交替抖动
                try:
                    for i in range(1, len(cleaned_actions)):
                        a0 = cleaned_actions[i - 1]
                        a1 = cleaned_actions[i]
                        if a0 == "turn_left" and a1 == "turn_right":
                            cleaned_actions[i] = "turn_left"
                        elif a0 == "turn_right" and a1 == "turn_left":
                            cleaned_actions[i] = "turn_right"
                except Exception:
                    pass

                # 若上一步疑似碰撞，优先避免首步继续前后移动
                try:
                    if bool(collision_info.get("had_collision")) and cleaned_actions:
                        if cleaned_actions[0] in ("move_forward", "move_backward"):
                            cleaned_actions[0] = preferred_turn
                except Exception:
                    pass

                while len(cleaned_actions) < self.action_steps:
                    cleaned_actions.append(preferred_turn)
                cleaned_actions = cleaned_actions[: self.action_steps]

            # 构建返回结果
            result = {
                "actions": cleaned_actions,
                "reached_goal": bool(reached_goal),
                "reasoning": reasoning,
                "current_room": data.get("current_room"),
                "current_floor": data.get("current_floor"),
                "confidence": data.get("confidence"),
                "subgoal": subgoal_text,
            }

            # 记录“本轮 VLM 决策经历”，为后续空间理解/避免循环提供上下文
            try:
                subgoal = subgoal_text

                self.record_experience(
                    kind="vlm_decision",
                    data={
                        "step": (ctx or {}).get("step"),
                        "current_room": data.get("current_room"),
                        "current_floor": data.get("current_floor"),
                        "confidence": conf_val,
                        "subgoal": subgoal,
                        "actions": cleaned_actions,
                        "in_stairwell": bool((stairs_ctx or {}).get("in_stairwell")),
                        "floorplan_floor_used": (stairs_ctx or {}).get(
                            "floorplan_floor_used"
                        ),
                        "collision": {
                            "had_collision": bool(collision_info.get("had_collision")),
                            "consecutive": collision_info.get("consecutive"),
                            "action": collision_info.get("action"),
                        },
                    },
                    importance=0.35,
                    source="vlm",
                )
            except Exception:
                pass

            # ──────────────────────────────────────────────────────────
            # 关键记忆抽取：
            # - 高置信定位 -> confirmed_location
            # - reasoning 中的 subgoal -> last_subgoal
            # - 连续碰撞/卡住 -> collision_hotspot
            # 注意：这些记忆会落盘（若配置/输出目录允许）。
            # ──────────────────────────────────────────────────────────
            try:
                vlm_room = (data.get("current_room") or "").strip()
                vlm_floor = data.get("current_floor")
                conf = conf_val
                if conf is None:
                    try:
                        conf = float(data.get("confidence"))
                    except Exception:
                        conf = None

                if (
                    vlm_room
                    and vlm_room.lower() != "unknown"
                    and conf is not None
                    and conf >= 0.85
                ):
                    self.add_key_memory(
                        kind="confirmed_location",
                        data={
                            "room": vlm_room,
                            "floor": vlm_floor,
                            "evidence": (reasoning or "")[:160],
                        },
                        importance=min(1.0, float(conf)),
                        pinned=False,
                        source="vlm",
                    )

                # 从 reasoning 里提取 subgoal（我们要求了模板 subgoal=...）
                try:
                    if subgoal_text:
                        self.add_key_memory(
                            kind="last_subgoal",
                            data={"subgoal": subgoal_text},
                            importance=0.6,
                            pinned=False,
                            source="vlm",
                        )
                except Exception:
                    pass

                try:
                    consecutive = int(collision_info.get("consecutive", 0) or 0)
                except Exception:
                    consecutive = 0
                if consecutive >= 2:
                    self.add_key_memory(
                        kind="collision_hotspot",
                        data={
                            "action": collision_info.get("action"),
                            "consecutive": consecutive,
                            "note": "疑似卡住：优先转向重新定位",
                        },
                        importance=0.75,
                        pinned=False,
                        source="env",
                    )
            except Exception:
                pass

            # 记录到历史
            self.action_history.extend(cleaned_actions)
            self.add_memory(
                {
                    "type": "action_batch_generated",
                    "actions": cleaned_actions,
                    "reached_goal": bool(reached_goal),
                    "current_room": current_room,
                    "target_room": target_room,
                    "reasoning": reasoning,
                    "vlm_current_room": data.get("current_room"),
                    "vlm_current_floor": data.get("current_floor"),
                    "vlm_confidence": data.get("confidence"),
                    "subgoal": subgoal_text,
                }
            )

            if bool(reached_goal or done):
                logger.info(f"🎯 Agent1: 已到达目标 {target_room}！")
            else:
                logger.info(
                    f"✅ Agent1 生成动作序列: {cleaned_actions} | 推理: {reasoning}"
                )

            return result

        except json.JSONDecodeError as e:
            logger.error(f"❌ Agent1 JSON解析失败: {e}")
            logger.error(f"   原始响应: {response[:200]}")
            return None
        except Exception as e:
            logger.error(f"❌ Agent1 响应处理失败: {e}")
            return None

    def generate_next_action(
        self,
        rgb_image: np.ndarray,
        floorplan: np.ndarray,
        current_room: str,
        target_room: str,
        path_rooms: List[str] = None,
        context: Dict[str, Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        生成下一步动作（单步）

        Args:
            rgb_image: 前置RGB图像 (H, W, 3)
            floorplan: 楼层平面图 (H, W, 3)
            current_room: 当前房间ID
            target_room: 目标房间ID
            path_rooms: 规划的路径房间序列
            context: 额外上下文信息

        Returns:
            {
                "action": str,  # "move_forward"/"turn_left"/"turn_right"/"stop"
                "reached_goal": bool,  # 是否到达目标房间
                "reasoning": str  # VLM推理过程
            } 或 None
        """
        # 转换图像为Data URL
        rgb_url = self._image_to_data_url(rgb_image, kind="rgb")
        floorplan_url = self._image_to_data_url(floorplan, kind="floorplan")

        # 路径信息
        path_str = " → ".join(path_rooms) if path_rooms else "未规划"
        memory_summary = self.get_memory_summary()
        ltm_block = self.get_key_memory_prompt_block()

        # 将“规划路径”写入长期记忆（pinned）
        try:
            if path_rooms:
                self.add_key_memory(
                    kind="planned_path",
                    data={"path": " → ".join(path_rooms)},
                    importance=0.95,
                    pinned=True,
                    source="planner",
                )
        except Exception:
            pass

        # 构建提示词（单步决策 + 目标判断）
        prompt = f"""你是一个机器狗(Spot)导航控制智能体。

【当前状态】
- 当前房间: {current_room}
- 目标房间: {target_room}
- 规划路径: {path_str}

【机器狗规格】
- 身高: 0.55米 | 宽度: 0.28米
- 单步前进: 0.25米 | 单次转弯: 10°

【任务】
分析前置RGB摄像头视图和楼层平面图，判断：
1. **是否已到达目标房间** {target_room}？
2. 如果未到达，规划**下一步最优动作**

【可用动作】
- move_forward: 前进0.25米
- turn_left: 左转10°
- turn_right: 右转10°
- stop: 停止（仅当到达目标时）

【输出格式】(严格JSON格式)
{{
  "reached_goal": true/false,
  "action": "move_forward",
  "reasoning": "简短推理过程"
}}

如果已到达目标房间，输出:
{{
  "reached_goal": true,
  "action": "stop",
  "reasoning": "已到达目标房间{target_room}"
}}

【历史记忆】
{memory_summary}

{ltm_block}

请直接输出JSON，无其他内容。
"""

        # 调用VLM
        images = [rgb_url, floorplan_url]
        response = self.call_vlm(prompt, images)

        if not response:
            logger.error("❌ Agent1 无法生成动作")
            return None

        # 解析响应 - JSON格式
        try:
            # 清理响应（去除可能的markdown标记）
            response_clean = response.strip()
            if response_clean.startswith("```"):
                response_clean = response_clean.split("\n", 1)[1]
            if response_clean.endswith("```"):
                response_clean = response_clean.rsplit("\n", 1)[0]
            response_clean = response_clean.strip()

            # 解析JSON
            data = json.loads(response_clean)

            reached_goal = data.get("reached_goal", False)
            action = data.get("action", "move_forward").strip().lower()
            reasoning = data.get("reasoning", "无推理")

            # 验证动作
            valid_actions = {"move_forward", "turn_left", "turn_right", "stop"}
            if action not in valid_actions:
                # 尝试标准化
                if "forward" in action or "前进" in action:
                    action = "move_forward"
                elif "left" in action or "左" in action:
                    action = "turn_left"
                elif "right" in action or "右" in action:
                    action = "turn_right"
                elif "stop" in action or "停" in action:
                    action = "stop"
                else:
                    action = "move_forward"  # 默认前进

            # 构建返回结果
            result = {
                "action": action,
                "reached_goal": reached_goal,
                "reasoning": reasoning,
            }

            # 单步模式的关键记忆抽取（更保守）：
            # - reached_goal=true 记为高重要度事件
            # - reasoning 里若包含 subgoal=... 也记录
            try:
                if bool(reached_goal):
                    self.add_key_memory(
                        kind="reached_goal",
                        data={
                            "target_room": target_room,
                            "note": (reasoning or "")[:160],
                        },
                        importance=1.0,
                        pinned=True,
                        source="vlm",
                    )

                try:
                    m = re.search(r"subgoal\s*=\s*([^;|]+)", str(reasoning))
                    if m:
                        subgoal = m.group(1).strip()
                        if subgoal:
                            self.add_key_memory(
                                kind="last_subgoal",
                                data={"subgoal": subgoal},
                                importance=0.6,
                                pinned=False,
                                source="vlm",
                            )
                except Exception:
                    pass

                ctx = context or {}
                collision_info = ctx.get("collision") or {}
                try:
                    consecutive = int(collision_info.get("consecutive", 0) or 0)
                except Exception:
                    consecutive = 0
                if consecutive >= 2:
                    self.add_key_memory(
                        kind="collision_hotspot",
                        data={
                            "action": collision_info.get("action"),
                            "consecutive": consecutive,
                            "note": "单步模式疑似卡住：优先转向",
                        },
                        importance=0.75,
                        pinned=False,
                        source="env",
                    )
            except Exception:
                pass

            # 记录到历史
            self.action_history.append(action)
            self.add_memory(
                {
                    "type": "action_generated",
                    "action": action,
                    "reached_goal": reached_goal,
                    "current_room": current_room,
                    "target_room": target_room,
                    "reasoning": reasoning,
                }
            )

            if reached_goal:
                logger.info(f"🎯 Agent1: 已到达目标 {target_room}！")
            else:
                logger.info(f"✅ Agent1 下一步动作: {action} | 推理: {reasoning}")

            return result

        except json.JSONDecodeError as e:
            logger.error(f"❌ Agent1 JSON解析失败: {e}")
            logger.error(f"   原始响应: {response[:200]}")
            return None
        except Exception as e:
            logger.error(f"❌ Agent1 响应处理失败: {e}")
            return None

    def select_best_path(
        self,
        candidate_paths: List[Tuple[List[str], List[str], int]],
        floor_maps: Dict[int, np.ndarray],
        start_room: str,
        end_room: str,
    ) -> int:
        """
        从3条候选路径中选择最佳路径（阶段1）

        Args:
            candidate_paths: [(rooms, doors, steps), ...] 候选路径列表
            floor_maps: {floor: image} 所有楼层的平面图
            start_room: 起点房间
            end_room: 终点房间

        Returns:
            选中的路径索引 (0-2)
        """
        # 准备平面图图像
        floor_image_urls = []
        for floor_num in sorted(floor_maps.keys()):
            img = floor_maps[floor_num]
            floor_image_urls.append(self._image_to_data_url(img, kind="floorplan"))

        # 构建候选路径描述
        path_descriptions = []
        for i, (rooms, doors, steps) in enumerate(candidate_paths, 1):
            desc = f"""路径{i}: {" → ".join(rooms)}
  - 总步数: {steps}
  - 经过门: {", ".join(doors)}
  - 楼层变化: {self._describe_floor_changes(rooms)}"""
            path_descriptions.append(desc)

        paths_text = "\n\n".join(path_descriptions)

        # 构建提示词
        floorplan_legend = ""
        protocol_text = ""
        try:
            sys_prompts = self._prompts_config.get("system", {}) or {}
            floorplan_legend = (sys_prompts.get("floorplan_legend", "") or "").strip()
            protocol_text = (
                sys_prompts.get("floorplan_navigation_protocol", "") or ""
            ).strip()
        except Exception:
            floorplan_legend = ""
            protocol_text = ""

        legend_block = ""
        if floorplan_legend:
            legend_block = "\n" + floorplan_legend + "\n"
        protocol_block = ""
        if protocol_text:
            protocol_block = "\n" + protocol_text + "\n"

        route_guidance_req = ""
        try:
            sys_prompts = self._prompts_config.get("system", {}) or {}
            route_guidance_req = (
                sys_prompts.get("route_guidance_requirements", "") or ""
            ).strip()
        except Exception:
            route_guidance_req = ""

        route_req_block = ""
        if route_guidance_req:
            route_req_block = "\n" + route_guidance_req + "\n"

        prompt = f"""你是一个机器狗路径规划智能体。

【任务】
从3条候选路径中选择最优路径，用于从 {start_room} 导航到 {end_room}。

{legend_block}
{protocol_block}
{route_req_block}

【候选路径】
{paths_text}

【选择标准】
1. 优先选择步数最少的路径
2. 考虑门的宽度（优先宽门）
3. 避免频繁跨楼层
4. 参考楼层平面图的可通行性

【输出格式】
严格输出 JSON（只输出 JSON，无其它文字）：
{{
    "choice": 1,
    "route_description": "..."
}}
"""

        # 调用VLM
        response = self.call_vlm(prompt, floor_image_urls)

        if not response:
            logger.warning("⚠️ Agent1 路径选择失败，默认选择路径1")
            choice_idx = 0
            route_text = self._build_route_guidance_text(
                candidate_paths[choice_idx][0], start_room, end_room
            )
            self._persist_route_guidance(
                rooms=candidate_paths[choice_idx][0],
                start_room=start_room,
                end_room=end_room,
                choice=choice_idx,
                route_description=route_text,
            )
            return choice_idx

        # 解析响应
        try:
            response_clean = (response or "").strip()
            # 容忍模型包裹 ```json
            if response_clean.startswith("```"):
                response_clean = response_clean.split("\n", 1)[1]
            if response_clean.endswith("```"):
                response_clean = response_clean.rsplit("\n", 1)[0]
            response_clean = response_clean.strip()

            choice_idx = 0
            route_desc = ""

            # 优先解析 JSON
            try:
                obj = json.loads(response_clean)
                c = obj.get("choice")
                if c is not None:
                    choice_idx = int(c) - 1
                route_desc = str(obj.get("route_description", "") or "").strip()
            except Exception:
                choice_idx = None

            # JSON 失败则退回到“提取数字”模式
            if choice_idx is None:
                import re

                match = re.search(r"[123]", response_clean)
                if match:
                    choice_idx = int(match.group()) - 1
                else:
                    choice_idx = 0

            if not (0 <= int(choice_idx) < len(candidate_paths)):
                choice_idx = 0

            if not route_desc:
                route_desc = self._build_route_guidance_text(
                    candidate_paths[choice_idx][0], start_room, end_room
                )

            self._persist_route_guidance(
                rooms=candidate_paths[choice_idx][0],
                start_room=start_room,
                end_room=end_room,
                choice=int(choice_idx),
                route_description=route_desc,
            )

            logger.info(f"✅ Agent1 选择路径{int(choice_idx) + 1}")
            return int(choice_idx)

        except Exception as e:
            logger.error(f"❌ Agent1 路径选择异常: {e}")
            choice_idx = 0
            route_text = self._build_route_guidance_text(
                candidate_paths[choice_idx][0], start_room, end_room
            )
            self._persist_route_guidance(
                rooms=candidate_paths[choice_idx][0],
                start_room=start_room,
                end_room=end_room,
                choice=choice_idx,
                route_description=route_text,
            )
            return choice_idx

    def _persist_route_guidance(
        self,
        *,
        rooms: List[str],
        start_room: str,
        end_room: str,
        choice: int,
        route_description: str,
    ) -> None:
        """把阶段1生成的整体指路写入 LTM 并（若有输出目录）落盘。"""
        try:
            self.add_key_memory(
                kind="route_guidance",
                data={
                    "start": start_room,
                    "end": end_room,
                    "choice": int(choice) + 1,
                    "path": " → ".join(list(rooms or [])),
                    "text": str(route_description or "")[:4000],
                },
                importance=1.0,
                pinned=True,
                source="planner",
            )
        except Exception:
            pass

        try:
            if self.output_dir is None:
                return
            # run.py 会创建 paths 子目录；若没有也创建
            pdir = self.output_dir / "paths"
            pdir.mkdir(parents=True, exist_ok=True)
            out_path = pdir / "route_guidance.txt"
            out_path.write_text(str(route_description or ""), encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def _build_route_guidance_text(
        rooms: List[str], start_room: str, end_room: str
    ) -> str:
        """在无 VLM 描述时，生成一段可用的“指路式”整体导航描述（不依赖几何，只依赖路径序列与编号规则）。"""
        if not rooms:
            return f"从 {start_room} 前往 {end_room}：未获得有效路径。"

        def kind(label: str) -> str:
            s = str(label)
            if s.startswith("Start"):
                return "outdoor"
            if s.startswith("S"):
                return "stair"
            if s.startswith("H"):
                return "hallway"
            if s.startswith("R"):
                return "room"
            return "node"

        def floor_of(label: str) -> Optional[int]:
            m = re.search(r"(\d)", str(label))
            if not m:
                return None
            try:
                return int(m.group(1))
            except Exception:
                return None

        steps = []
        steps.append(f"起点：从 {start_room} 出发，目标 {end_room}。")
        steps.append(
            "按平面图依次经过以下节点（以门洞D为可通行入口，W/C为不可穿过障碍）："
        )
        steps.append("路径：" + " → ".join([str(r) for r in rooms]))

        # 提示楼层/楼梯规则
        floors = [floor_of(r) for r in rooms if floor_of(r) is not None]
        if floors:
            steps.append(
                f"楼层提示：路径涉及楼层 {sorted(set(floors))}；跨层仅在楼梯间 Sxx 内完成两跑后才更新楼层。"
            )

        # 生成更“指路”的分段描述
        for i in range(len(rooms) - 1):
            a, b = rooms[i], rooms[i + 1]
            ka, kb = kind(a), kind(b)
            fa, fb = floor_of(a), floor_of(b)
            if kb == "stair":
                steps.append(
                    f"第{i + 1}段：从 {a} 朝楼梯间 {b} 方向移动，进入楼梯间后保持沿楼梯方向前进，尽量少左右震荡。"
                )
            elif ka == "stair" and kb in ("hallway", "room"):
                if fa is not None and fb is not None and fa != fb:
                    steps.append(
                        f"第{i + 1}段：在楼梯间完成两跑并到达 {fb}F 落地后，从 {a} 出楼梯进入 {b}（确认有走廊/门洞D再更新楼层）。"
                    )
                else:
                    steps.append(
                        f"第{i + 1}段：从楼梯间 {a} 出口进入 {b}，用平面图确认你已离开 S 区域。"
                    )
            else:
                steps.append(
                    f"第{i + 1}段：从 {a} 朝 {b} 前进；优先对齐走廊/门洞方向（小角度转向），再稳定 move_forward 前进。"
                )

        steps.append(
            "执行策略：每次决策先 Map-First 定位（看平面图的H/R/S/D结构），再用 t-1/t 视野变化确认转向或前进是否生效；若连续碰撞，先转向+后退脱困再前进。"
        )
        return "\n".join(steps)

    @staticmethod
    def _describe_floor_changes(rooms: List[str]) -> str:
        """描述路径中的楼层变化"""
        from src.scene_graph import SceneGraph

        floors = [SceneGraph.extract_floor(r) for r in rooms]
        unique_floors = sorted(set(floors))

        if len(unique_floors) == 1:
            return f"{unique_floors[0]}F (单楼层)"
        else:
            return (
                f"{unique_floors[0]}F → {unique_floors[-1]}F (跨{len(unique_floors)}层)"
            )

    def update_position(self, new_position: np.ndarray, action: str):
        """更新位置记忆"""
        self.last_action = action
        self.add_memory(
            {
                "type": "position_update",
                "position": new_position.tolist(),
                "action": action,
            }
        )

    # NOTE: 旧的 @staticmethod _image_to_data_url 已被移动到 VLMAgent._image_to_data_url
    # 以便统一读取配置与复用 HTTP session。


class Agent2(VLMAgent):
    """
    智能体2 - 路径规划与重规划
    功能：路径规划、通行性判断、重规划
    """

    def __init__(
        self,
        memory_size: int = None,
        config_dir: str = "./configs",
        output_dir: str = None,
        vlm_trace_path: str = None,
    ):
        super().__init__("Agent2", memory_size, config_dir, output_dir, vlm_trace_path)

    def replan_path(
        self,
        current_room: str,
        target_room: str,
        blocked_edge: Tuple[str, str],
        scene_graph: Any,
    ) -> Optional[List[str]]:
        """
        重规划路径（当遇到障碍时）

        Args:
            current_room: 当前房间
            target_room: 目标房间
            blocked_edge: (from_room, to_room) 被阻塞的边
            scene_graph: 场景图对象

        Returns:
            新的路径房间列表，如果无法规划则返回None
        """
        logger.info(
            f"🔄 Agent2: 正在重规划路径 (避开 {blocked_edge[0]} -> {blocked_edge[1]})..."
        )

        # 临时移除边
        u, v = blocked_edge
        removed = False
        weight = 0

        try:
            if scene_graph.graph.has_edge(u, v):
                weight = scene_graph.graph[u][v]["weight"]
                scene_graph.graph.remove_edge(u, v)
                removed = True

            # 重新计算最短路径
            import networkx as nx

            try:
                path = nx.shortest_path(
                    scene_graph.graph,
                    source=current_room,
                    target=target_room,
                    weight="weight",
                )
                logger.info(f"✅ Agent2: 重规划成功: {' → '.join(path)}")
                return path
            except nx.NetworkXNoPath:
                logger.error("❌ Agent2: 无法找到替代路径")
                return None

        finally:
            # 恢复边
            if removed:
                scene_graph.graph.add_edge(u, v, weight=weight)

    def check_passability(
        self,
        rgb_image: np.ndarray,
        current_room: str,
        next_room: str,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        判断是否可通行

        Args:
            rgb_image: RGB图像
            current_room: 当前房间
            next_room: 下一个房间
            context: 上下文

        Returns:
            {"passable": bool, "confidence": float, "reason": str}
        """
        rgb_url = self._image_to_data_url(rgb_image, kind="rgb")

        prompt = f"""你是一个环境感知智能体。
当前位置: {current_room}
尝试前往: {next_room}

请根据RGB图像判断从{current_room}到{next_room}是否可以通行。
考虑: 障碍物、门是否打开、空间是否足够等

请以JSON格式输出:
{{"passable": true/false, "confidence": 0.0-1.0, "reason": "具体原因"}}

上下文信息:
{json.dumps(context or {}, ensure_ascii=False, indent=2)}
"""

        response = self.call_vlm(prompt, [rgb_url])

        if not response:
            logger.warning("⚠️ Agent2 无法判断通行性，默认不通行")
            return {"passable": False, "confidence": 0.0, "reason": "VLM调用失败"}

        try:
            data = json.loads(response)
            result = {
                "passable": data.get("passable", False),
                "confidence": float(data.get("confidence", 0.0)),
                "reason": data.get("reason", "未知原因"),
            }

            self.add_memory(
                {
                    "type": "passability_check",
                    "current_room": current_room,
                    "next_room": next_room,
                    "result": result,
                }
            )

            if not result["passable"]:
                self.blocked_locations.append(next_room)

            logger.info(
                f"✅ Agent2 通行性判断: {next_room} -> {result['passable']} ({result['confidence']:.2f})"
            )
            return result

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"❌ Agent2 响应解析失败: {e}")
            return {"passable": False, "confidence": 0.0, "reason": f"解析错误: {e}"}

    def mark_blocked(self, current_room: str, next_room: str, reason: str):
        """
        标记房间为不通行（主动记录）

        Args:
            current_room: 当前房间
            next_room: 被阻挡的房间
            reason: 阻挡原因
        """
        if next_room not in self.blocked_locations:
            self.blocked_locations.append(next_room)

        self.add_memory(
            {
                "type": "blocked_marked",
                "current_room": current_room,
                "blocked_room": next_room,
                "reason": reason,
            }
        )

        logger.info(f"🚫 Agent2 标记 {next_room} 为不通行，原因: {reason}")

    def replan_path(
        self,
        scene_graph: "SceneGraph",
        current_room: str,
        target_room: str,
        blocked_room: Optional[str] = None,
    ) -> Optional[List[str]]:
        """
        重新规划路径（排除被阻挡的房间）

        Args:
            scene_graph: 场景图对象
            current_room: 当前房间（新起点）
            target_room: 目标房间
            blocked_room: 新发现不通行的房间

        Returns:
            新的房间序列或None
        """
        # 添加新阻挡房间到列表
        if blocked_room and blocked_room not in self.blocked_locations:
            self.blocked_locations.append(blocked_room)
            logger.info(f"⚠️ Agent2 新增阻挡房间: {blocked_room}")

        # 使用A*算法重新规划，传入所有阻挡房间
        blocked_set = set(self.blocked_locations)
        logger.info(f"🔍 Agent2 重规划: 避开 {len(blocked_set)} 个阻挡房间")

        try:
            # 调用场景图生成候选路径，传入blocked_rooms参数
            paths = scene_graph.find_k_shortest_paths(
                current_room, target_room, k=3, blocked_rooms=blocked_set
            )

            if not paths:
                logger.error(f"❌ Agent2 从 {current_room} 到 {target_room} 无可用路径")
                return None

            # 选择第一条路径（A*已经避开了阻挡房间）
            new_path, _, steps = paths[0]

            self.add_memory(
                {
                    "type": "path_replanned",
                    "current_room": current_room,
                    "target_room": target_room,
                    "new_path": new_path,
                    "avoided_rooms": list(blocked_set),
                    "steps": steps,
                }
            )

            logger.info(f"✅ Agent2 路径重规划成功: {' → '.join(new_path)} ({steps}步)")
            logger.info(f"   已避开: {', '.join(blocked_set)}")

            return new_path

        except Exception as e:
            logger.error(f"❌ Agent2 重规划异常: {e}")
            import traceback

            traceback.print_exc()
            return None
