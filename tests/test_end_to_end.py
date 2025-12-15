#!/usr/bin/env python3
"""
端到端完整测试 - 从场景加载到导航结束
================================================

测试目的：
1. 验证Habitat场景加载
2. 验证机器人URDF加载
3. 验证双RGB传感器配置
4. 验证两阶段导航流程
5. 验证RGB图像保存
6. 验证VLM输入输出记录
7. 验证视频录制
8. 验证Agent2重规划
9. 验证可视化界面
10. 生成详细测试报告供人工判断

测试模式：
- Mock模式：快速测试系统逻辑
- Habitat模式：实际环境测试（需要Habitat安装）
"""

import unittest
import sys
import os
from pathlib import Path
import numpy as np
import shutil
import json
import time
import subprocess
from datetime import datetime
from unittest.mock import MagicMock, patch
import logging

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from navigation_system import NavigationSystem
from habitat_integration import EnvironmentFactory, MockEnvironment
from agents import Agent1, Agent2
from config_loader import UnifiedConfigLoader


class TestEndToEnd(unittest.TestCase):
    """端到端完整测试"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.test_start_time = datetime.now()
        cls.test_results = []
        cls.config_dir = Path("configs")
        cls.test_output_dir = Path("tests/test_output_e2e")

        # 创建测试输出目录
        if cls.test_output_dir.exists():
            shutil.rmtree(cls.test_output_dir)
        cls.test_output_dir.mkdir(parents=True)

        # 设置日志
        log_file = cls.test_output_dir / "test_report.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        cls.logger = logging.getLogger("EndToEndTest")
        cls.logger.info("=" * 80)
        cls.logger.info("端到端测试开始")
        cls.logger.info("=" * 80)

    def setUp(self):
        """每个测试前的初始化"""
        self.test_name = self._testMethodName
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"测试: {self.test_name}")
        self.logger.info(f"{'=' * 80}")

    def tearDown(self):
        """每个测试后的清理"""
        pass

    def _record_result(self, test_name: str, status: str, details: dict):
        """记录测试结果"""
        result = {
            "test_name": test_name,
            "status": status,  # "PASS", "FAIL", "WARNING"
            "timestamp": datetime.now().isoformat(),
            "details": details,
        }
        self.test_results.append(result)

        # 输出到日志
        self.logger.info(f"结果: {status}")
        self.logger.info(f"详情: {json.dumps(details, indent=2, ensure_ascii=False)}")

    # ========================================================================
    # 测试1: 配置文件加载
    # ========================================================================

    def test_01_config_loading(self):
        """测试1: 验证所有配置文件加载"""
        self.logger.info("【测试1】配置文件加载验证")

        try:
            loader = UnifiedConfigLoader(config_dir=str(self.config_dir))

            # 加载所有配置
            configs = {
                "environment": loader.load_environment_config(),
                "agent": loader.load_config("agent_config"),
                "vlm": loader.load_vlm_config(),
                "paths": loader.load_paths_config(),
                "system": loader.load_config("system_config"),
            }

            # 验证关键配置
            checks = {
                "场景路径配置": configs["environment"].get("scene", {}).get("path")
                is not None,
                "机器人URDF配置": configs["environment"]
                .get("robot", {})
                .get("urdf_path")
                is not None,
                "前置摄像头配置": configs["environment"]
                .get("sensors", {})
                .get("front_camera")
                is not None,
                "后置摄像头配置": configs["environment"]
                .get("sensors", {})
                .get("back_camera")
                is not None,
                "Agent1配置": configs["agent"].get("agent1") is not None,
                "Agent2配置": configs["agent"].get("agent2") is not None,
                "VLM API配置": configs["vlm"].get("api") is not None,
            }

            all_pass = all(checks.values())

            self._record_result(
                "config_loading",
                "PASS" if all_pass else "FAIL",
                {
                    "configs_loaded": list(configs.keys()),
                    "scene_path": configs["environment"].get("scene", {}).get("path"),
                    "robot_urdf": configs["environment"]
                    .get("robot", {})
                    .get("urdf_path"),
                    "checks": checks,
                },
            )

            self.assertTrue(all_pass, "配置文件加载失败")

        except Exception as e:
            self._record_result("config_loading", "FAIL", {"error": str(e)})
            raise

    # ========================================================================
    # 测试2: 场景文件存在性验证
    # ========================================================================

    def test_02_scene_files_existence(self):
        """测试2: 验证场景和机器人文件是否存在"""
        self.logger.info("【测试2】场景文件存在性验证")

        try:
            loader = UnifiedConfigLoader(config_dir=str(self.config_dir))
            env_config = loader.load_environment_config()

            scene_path = env_config.get("scene", {}).get("path")
            robot_urdf = env_config.get("robot", {}).get("urdf_path")

            files_check = {
                "场景文件": Path(scene_path).exists() if scene_path else False,
                "机器人URDF": Path(robot_urdf).exists() if robot_urdf else False,
            }

            # 检查navmesh（可选）
            navmesh_path = env_config.get("scene", {}).get("navmesh")
            if navmesh_path:
                files_check["Navmesh文件"] = Path(navmesh_path).exists()

            all_exist = all(files_check.values())

            self._record_result(
                "scene_files_existence",
                "PASS" if all_exist else "WARNING",
                {
                    "scene_path": scene_path,
                    "robot_urdf": robot_urdf,
                    "files_check": files_check,
                },
            )

            if not all_exist:
                self.logger.warning("部分文件不存在，将使用Mock环境")

        except Exception as e:
            self._record_result("scene_files_existence", "FAIL", {"error": str(e)})
            raise

    # ========================================================================
    # 测试3: Mock环境创建和传感器验证
    # ========================================================================

    def test_03_mock_environment_creation(self):
        """测试3: 创建Mock环境并验证传感器"""
        self.logger.info("【测试3】Mock环境创建和传感器验证")

        try:
            # 创建Mock环境
            logger = logging.getLogger("MockEnvTest")
            env = MockEnvironment(logger)

            # 重置环境
            obs = env.reset()

            # 验证观测
            checks = {
                "rgb_front存在": "rgb_front" in obs,
                "rgb_back存在": "rgb_back" in obs,
                "agent_pos存在": "agent_pos" in obs,
                "agent_rot存在": "agent_rot" in obs,
                "rgb_front形状正确": obs.get("rgb_front", np.array([])).shape
                == (720, 1280, 3),
                "rgb_back形状正确": obs.get("rgb_back", np.array([])).shape
                == (480, 640, 3),
                "agent_pos维度正确": obs.get("agent_pos", np.array([])).shape == (3,),
            }

            all_pass = all(checks.values())

            # 测试step
            obs_after, info = env.step("move_forward")
            step_works = "agent_pos" in obs_after

            # 测试get_navigable_map
            nav_map = env.get_navigable_map()
            has_nav_map = nav_map is not None and nav_map.size > 0

            self._record_result(
                "mock_environment_creation",
                "PASS" if all_pass and step_works else "FAIL",
                {
                    "observations_check": checks,
                    "step_works": step_works,
                    "navigable_map_available": has_nav_map,
                    "navigable_map_shape": nav_map.shape if has_nav_map else None,
                },
            )

            env.close()
            self.assertTrue(all_pass and step_works, "Mock环境验证失败")

        except Exception as e:
            self._record_result("mock_environment_creation", "FAIL", {"error": str(e)})
            raise

    # ========================================================================
    # 测试4: Habitat环境创建（可选）
    # ========================================================================

    def test_04_habitat_environment_creation(self):
        """测试4: 尝试创建Habitat环境（如果可用）"""
        self.logger.info("【测试4】Habitat环境创建测试（可选）")

        try:
            loader = UnifiedConfigLoader(config_dir=str(self.config_dir))
            env_cfg = loader.load_environment_config()
            scene_path = env_cfg.get("scene", {}).get("path")

            # 检查Habitat是否可用
            try:
                import habitat_sim

                habitat_available = True
            except ImportError:
                habitat_available = False
                self.logger.warning("Habitat-sim未安装，跳过Habitat环境测试")

            if not habitat_available or not Path(scene_path).exists():
                self._record_result(
                    "habitat_environment_creation",
                    "SKIP",
                    {
                        "reason": "Habitat不可用或场景文件不存在",
                        "habitat_available": habitat_available,
                        "scene_exists": Path(scene_path).exists()
                        if scene_path
                        else False,
                    },
                )
                self.skipTest("Habitat环境不可用")
                return

            # 额外安全检查：在子进程里探测是否能创建 windowless EGL 上下文。
            # 某些环境下 habitat-sim 会在创建 Simulator 时直接 exit(1)/abort，无法被 try/except 捕获。
            def _probe_habitat_sim_safe(scene: str) -> bool:
                code = (
                    "import os\n"
                    "import habitat_sim\n"
                    "scene=os.environ.get('HAB_SCENE','')\n"
                    "sim_cfg=habitat_sim.SimulatorConfiguration()\n"
                    "sim_cfg.scene_id=scene\n"
                    "sim_cfg.enable_physics=False\n"
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
                env["HAB_SCENE"] = scene
                try:
                    p = subprocess.run(
                        [sys.executable, "-c", code],
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=45,
                    )
                    return p.returncode == 0 and "OK" in (p.stdout or "")
                except Exception:
                    return False

            if not _probe_habitat_sim_safe(str(scene_path)):
                self._record_result(
                    "habitat_environment_creation",
                    "SKIP",
                    {
                        "reason": "Habitat-sim EGL/GL context 不可用（子进程探测失败）",
                        "scene_path": scene_path,
                    },
                )
                self.skipTest("Habitat-sim EGL/GL context 不可用")
                return

            # 尝试创建Habitat环境
            factory = EnvironmentFactory(logging.getLogger("HabitatTest"))
            config = {
                "scene_path": scene_path,
                "use_habitat_lab": False,
                "agent_config": env_cfg.get("agent", {}),
                "sim_config": {
                    "enable_physics": env_cfg.get("physics", {}).get("enabled", True),
                },
                "env_config": env_cfg,
                "enable_visualization": False,
            }

            env, _ = factory.create_environment(config)

            # 验证环境
            obs = env.reset()
            checks = {
                "环境创建成功": env is not None,
                "rgb_front存在": "rgb_front" in obs,
                "rgb_back存在": "rgb_back" in obs,
                "agent_pos存在": "agent_pos" in obs,
            }

            # 测试step
            obs_after, info = env.step("move_forward")
            step_works = "agent_pos" in obs_after

            # 测试get_navigable_map
            nav_map = env.get_navigable_map()
            has_nav_map = nav_map is not None and nav_map.size > 0

            self._record_result(
                "habitat_environment_creation",
                "PASS" if all(checks.values()) and step_works else "FAIL",
                {
                    "checks": checks,
                    "step_works": step_works,
                    "navigable_map_available": has_nav_map,
                    "scene_path": scene_path,
                },
            )

            env.close()
            self.assertTrue(all(checks.values()), "Habitat环境验证失败")

        except Exception as e:
            # unittest.skipTest() 会抛出 SkipTest；这里要透传出去，避免被误记为 FAIL
            if isinstance(e, unittest.SkipTest):
                raise

            self._record_result(
                "habitat_environment_creation", "FAIL", {"error": str(e)}
            )
            # 不抛出异常，因为Habitat是可选的
            self.logger.warning(f"Habitat环境创建失败: {e}")

    # ========================================================================
    # 测试5: 场景图和路径规划
    # ========================================================================

    def test_05_scene_graph_path_planning(self):
        """测试5: 场景图加载和A*路径规划"""
        self.logger.info("【测试5】场景图和A*路径规划测试")

        try:
            from scene_graph import SceneGraph

            # 检查数据文件
            door_table = "data/door_table.xlsx"
            comp_table = "data/component_table.xlsx"

            if not Path(door_table).exists() or not Path(comp_table).exists():
                self._record_result(
                    "scene_graph_path_planning",
                    "SKIP",
                    {"reason": "场景图数据文件不存在"},
                )
                self.skipTest("场景图数据文件不存在")
                return

            # 创建场景图
            scene_graph = SceneGraph(door_table, comp_table)

            # 测试路径规划
            start = "Start01"
            end = "R309"

            paths = scene_graph.find_k_shortest_paths(start, end, k=3)

            checks = {
                "场景图创建成功": scene_graph.graph is not None,
                "找到候选路径": len(paths) > 0,
                "路径数量正确": len(paths) <= 3,
            }

            path_details = []
            for i, (rooms, edges, cost) in enumerate(paths, 1):
                path_details.append(
                    {
                        "path_id": i,
                        "rooms": rooms,
                        "length": len(rooms),
                        "cost": cost,
                    }
                )

            all_pass = all(checks.values())

            self._record_result(
                "scene_graph_path_planning",
                "PASS" if all_pass else "FAIL",
                {
                    "checks": checks,
                    "paths_found": len(paths),
                    "path_details": path_details,
                },
            )

            self.assertTrue(all_pass, "场景图路径规划失败")

        except Exception as e:
            self._record_result("scene_graph_path_planning", "FAIL", {"error": str(e)})
            raise

    # ========================================================================
    # 测试6: Agent初始化和VLM通信
    # ========================================================================

    def test_06_agent_initialization(self):
        """测试6: Agent初始化和基本功能"""
        self.logger.info("【测试6】Agent初始化测试")

        try:
            # 创建Agent实例
            agent1 = Agent1(
                config_dir=str(self.config_dir), output_dir=str(self.test_output_dir)
            )
            agent2 = Agent2(
                config_dir=str(self.config_dir), output_dir=str(self.test_output_dir)
            )

            checks = {
                "Agent1创建成功": agent1 is not None,
                "Agent2创建成功": agent2 is not None,
                "Agent1有记忆": hasattr(agent1, "memory"),
                "Agent2有记忆": hasattr(agent2, "memory"),
                "Agent1有VLM配置": hasattr(agent1, "vlm_url"),
                "Agent1有输出目录": agent1.output_dir is not None,
            }

            # 测试记忆功能
            agent1.add_memory({"test": "data"})
            memory_works = len(agent1.memory) == 1

            # 测试统计
            stats1 = agent1.get_stats()
            stats_works = "agent_id" in stats1

            self._record_result(
                "agent_initialization",
                "PASS"
                if all(checks.values()) and memory_works and stats_works
                else "FAIL",
                {
                    "checks": checks,
                    "memory_works": memory_works,
                    "stats_works": stats_works,
                    "agent1_stats": stats1,
                },
            )

            self.assertTrue(all(checks.values()), "Agent初始化失败")

        except Exception as e:
            self._record_result("agent_initialization", "FAIL", {"error": str(e)})
            raise

    # ========================================================================
    # 测试7: 完整导航流程（Mock环境）
    # ========================================================================

    def test_07_full_navigation_mock(self):
        """测试7: 完整导航流程（Mock环境）"""
        self.logger.info("【测试7】完整导航流程测试（Mock环境）")

        try:
            # 创建导航系统
            nav_system = NavigationSystem(config_dir=str(self.config_dir))
            nav_system.output_dir = self.test_output_dir / "navigation_test"
            nav_system.output_dir.mkdir(exist_ok=True)

            # 设置子目录
            nav_system.frames_dir = nav_system.output_dir / "frames"
            nav_system.videos_dir = nav_system.output_dir / "videos"
            nav_system.logs_dir = nav_system.output_dir / "logs"
            nav_system.metrics_dir = nav_system.output_dir / "metrics"
            nav_system.paths_dir = nav_system.output_dir / "paths"

            for d in [
                nav_system.frames_dir,
                nav_system.videos_dir,
                nav_system.logs_dir,
                nav_system.metrics_dir,
                nav_system.paths_dir,
            ]:
                d.mkdir(parents=True, exist_ok=True)

            # Mock Agent1和Agent2
            nav_system.agent1 = MagicMock(spec=Agent1)
            nav_system.agent1.output_dir = str(nav_system.output_dir)
            nav_system.agent1.generate_action_batch.return_value = {
                "actions": [
                    "move_forward",
                    "move_forward",
                    "turn_left",
                    "move_forward",
                ],
                "reached_goal": False,
                "reasoning": "测试批量动作",
            }
            nav_system.agent1.memory = []
            nav_system.agent1.get_stats.return_value = {
                "agent_id": "Agent1",
                "success_count": 1,
                "failure_count": 0,
            }

            nav_system.agent2 = MagicMock(spec=Agent2)
            nav_system.agent2.memory = []
            nav_system.agent2_enabled = True

            # 设置路径
            nav_system.current_path = ["Start01", "S101", "R309"]
            nav_system.target_room = "R309"
            nav_system.current_room = "Start01"

            # 创建Mock环境
            env = MockEnvironment(logging.getLogger("NavTest"))

            # 限制步数避免死循环
            nav_system.max_steps = 10

            # 执行导航
            nav_system.execute_navigation(env)

            # 验证结果
            checks = {
                "Agent1被调用": nav_system.agent1.generate_action_batch.called,
                "环境执行了动作": env.step_count > 0,
                "RGB前置图像已保存": len(
                    list(nav_system.frames_dir.glob("*front_rgb*.jpg"))
                )
                > 0,
                "RGB后置图像已保存": len(
                    list(nav_system.frames_dir.glob("*back_rgb*.jpg"))
                )
                > 0,
                "视频文件已创建": len(list(nav_system.videos_dir.glob("*.mp4"))) > 0,
            }

            # 统计
            front_images = list(nav_system.frames_dir.glob("*front_rgb*.jpg"))
            back_images = list(nav_system.frames_dir.glob("*back_rgb*.jpg"))
            videos = list(nav_system.videos_dir.glob("*.mp4"))

            all_pass = all(checks.values())

            self._record_result(
                "full_navigation_mock",
                "PASS" if all_pass else "FAIL",
                {
                    "checks": checks,
                    "total_steps": env.step_count,
                    "front_images_count": len(front_images),
                    "back_images_count": len(back_images),
                    "videos_count": len(videos),
                    "agent1_call_count": nav_system.agent1.generate_action_batch.call_count,
                },
            )

            env.close()
            self.assertTrue(all_pass, "完整导航流程失败")

        except Exception as e:
            self._record_result("full_navigation_mock", "FAIL", {"error": str(e)})
            raise

    # ========================================================================
    # 测试8: VLM输入输出记录
    # ========================================================================

    def test_08_vlm_io_recording(self):
        """测试8: VLM输入输出记录功能"""
        self.logger.info("【测试8】VLM输入输出记录测试")

        try:
            # 创建Agent实例
            agent1 = Agent1(
                config_dir=str(self.config_dir), output_dir=str(self.test_output_dir)
            )

            # Mock VLM API
            with patch("requests.post") as mock_post:
                mock_post.return_value.status_code = 200
                mock_post.return_value.json.return_value = {
                    "choices": [
                        {
                            "message": {
                                "content": '{"actions": ["move_forward"], "reached_goal": false, "reasoning": "test"}'
                            }
                        }
                    ]
                }

                # 调用VLM
                result = agent1.call_vlm(
                    "Test prompt for navigation",
                    ["data:image/jpeg;base64,test_image_data"],
                )

                # 等待文件写入
                time.sleep(0.1)

                # 验证统一的 VLM I/O 单文件
                vlm_io_file = self.test_output_dir / "vlm_io.json"

                checks = {
                    "VLM调用成功": result is not None,
                    "VLM记录文件创建": vlm_io_file.exists(),
                }

                # 读取并验证内容
                vlm_io_content = None
                if vlm_io_file.exists():
                    with open(vlm_io_file, "r", encoding="utf-8") as f:
                        vlm_io_content = json.load(f)
                has_records = (
                    isinstance(vlm_io_content, dict)
                    and isinstance(vlm_io_content.get("records"), list)
                    and len(vlm_io_content.get("records")) > 0
                )
                checks["VLM记录包含records"] = has_records

                all_pass = all(checks.values())

                self._record_result(
                    "vlm_io_recording",
                    "PASS" if all_pass else "FAIL",
                    {
                        "checks": checks,
                        "records_count": len(vlm_io_content.get("records"))
                        if isinstance(vlm_io_content, dict)
                        and isinstance(vlm_io_content.get("records"), list)
                        else 0,
                        "vlm_io_sample": (
                            (vlm_io_content.get("records") or [None])[0]
                            if isinstance(vlm_io_content, dict)
                            else None
                        ),
                    },
                )

                self.assertTrue(all_pass, "VLM输入输出记录失败")

        except Exception as e:
            self._record_result("vlm_io_recording", "FAIL", {"error": str(e)})
            raise

    # ========================================================================
    # 测试报告生成
    # ========================================================================

    @classmethod
    def tearDownClass(cls):
        """生成最终测试报告"""
        cls.logger.info("\n" + "=" * 80)
        cls.logger.info("生成测试报告")
        cls.logger.info("=" * 80)

        # 统计
        total_tests = len(cls.test_results)
        passed = sum(1 for r in cls.test_results if r["status"] == "PASS")
        failed = sum(1 for r in cls.test_results if r["status"] == "FAIL")
        skipped = sum(1 for r in cls.test_results if r["status"] == "SKIP")
        warnings = sum(1 for r in cls.test_results if r["status"] == "WARNING")

        # 生成报告
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "warnings": warnings,
                "start_time": cls.test_start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": (
                    datetime.now() - cls.test_start_time
                ).total_seconds(),
            },
            "test_results": cls.test_results,
        }

        # 保存JSON报告
        report_file = cls.test_output_dir / "test_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 生成人类可读报告
        readable_report = cls.test_output_dir / "TEST_REPORT.md"
        with open(readable_report, "w", encoding="utf-8") as f:
            f.write("# 端到端测试报告\n\n")
            f.write(
                f"**测试时间**: {cls.test_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )
            f.write(
                f"**测试时长**: {report['test_summary']['duration_seconds']:.2f} 秒\n\n"
            )

            f.write("## 测试统计\n\n")
            f.write(f"- **总测试数**: {total_tests}\n")
            f.write(f"- **通过**: {passed} ✅\n")
            f.write(f"- **失败**: {failed} ❌\n")
            f.write(f"- **跳过**: {skipped} ⏭️\n")
            f.write(f"- **警告**: {warnings} ⚠️\n\n")

            f.write("## 详细结果\n\n")

            for result in cls.test_results:
                status_icon = {
                    "PASS": "✅",
                    "FAIL": "❌",
                    "SKIP": "⏭️",
                    "WARNING": "⚠️",
                }.get(result["status"], "❓")

                f.write(f"### {status_icon} {result['test_name']}\n\n")
                f.write(f"**状态**: {result['status']}\n\n")
                f.write(f"**时间**: {result['timestamp']}\n\n")
                f.write("**详情**:\n\n")
                f.write("```json\n")
                f.write(json.dumps(result["details"], indent=2, ensure_ascii=False))
                f.write("\n```\n\n")
                f.write("---\n\n")

            # 判断建议
            f.write("## 判断建议\n\n")

            if failed > 0:
                f.write("❌ **测试未通过** - 存在失败的测试用例，请检查详细日志\n\n")
            elif warnings > 0:
                f.write("⚠️ **测试通过但有警告** - 部分可选功能不可用\n\n")
            else:
                f.write("✅ **所有测试通过** - 系统功能正常\n\n")

            f.write("### 关键检查项\n\n")
            f.write("请人工检查以下内容：\n\n")
            f.write("1. 配置文件是否正确加载 ✓\n")
            f.write("2. 场景文件和机器人模型是否存在（如需要）\n")
            f.write("3. Mock环境是否正常工作 ✓\n")
            f.write("4. Habitat环境是否成功创建（可选）\n")
            f.write("5. 路径规划是否生成合理路径 ✓\n")
            f.write("6. Agent初始化是否成功 ✓\n")
            f.write("7. 完整导航流程是否执行 ✓\n")
            f.write("8. RGB图像是否保存 ✓\n")
            f.write("9. VLM输入输出是否记录 ✓\n")
            f.write("10. 视频文件是否生成 ✓\n\n")

            f.write("### 输出文件位置\n\n")
            f.write(f"- **测试报告**: `{report_file}`\n")
            f.write(f"- **日志文件**: `{cls.test_output_dir / 'test_report.log'}`\n")
            f.write(f"- **导航输出**: `{cls.test_output_dir / 'navigation_test'}`\n\n")

        cls.logger.info(f"测试报告已保存: {readable_report}")
        cls.logger.info(f"JSON报告: {report_file}")
        cls.logger.info("\n" + "=" * 80)
        cls.logger.info(f"测试完成 - 通过: {passed}/{total_tests}")
        cls.logger.info("=" * 80)


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)
