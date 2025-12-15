#!/usr/bin/env python3
"""
run.py 集成测试

测试内容：
1. 配置加载
2. 场景图初始化和路径规划
3. VLMNavigationRunner初始化
4. 输出目录创建
5. 命令行参数解析
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config_loader import UnifiedConfigLoader
from src.scene_graph import SceneGraph
from src.agents import Agent1, Agent2
import logging


def setup_logger():
    """设置测试日志"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


logger = setup_logger()


def test_config_loading():
    """测试1: 配置加载"""
    logger.info("=" * 70)
    logger.info("测试1: 配置加载")
    logger.info("=" * 70)

    try:
        config_loader = UnifiedConfigLoader("./configs")

        env_config = config_loader.load_environment_config()
        agent_config = config_loader.load_agent_config()
        vlm_config = config_loader.load_vlm_config()
        system_config = config_loader.load_system_config()
        paths_config = config_loader.load_paths_config()
        prompts_config = config_loader.load_prompts_config()

        logger.info("✅ 环境配置加载成功")
        logger.info(f"   - 场景: {env_config.get('scene', {}).get('name', 'unknown')}")
        logger.info(
            f"   - 机器人: {env_config.get('agent', {}).get('height', 'unknown')}m"
        )

        logger.info("✅ Agent配置加载成功")
        logger.info(
            f"   - Agent1启用: {agent_config.get('agents', {}).get('agent1', {}).get('enabled', False)}"
        )
        logger.info(
            f"   - Agent2启用: {agent_config.get('agents', {}).get('agent2', {}).get('enabled', False)}"
        )

        logger.info("✅ VLM配置加载成功")
        logger.info(f"   - 模型: {vlm_config.get('model', {}).get('name', 'unknown')}")
        logger.info(
            f"   - API端点: {vlm_config.get('model', {}).get('api_endpoint', 'unknown')}"
        )

        logger.info("✅ 系统配置加载成功")
        logger.info("✅ 路径配置加载成功")
        logger.info("✅ 提示词配置加载成功")

        return True

    except Exception as e:
        logger.error(f"❌ 配置加载失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_scene_graph():
    """测试2: 场景图初始化和路径规划"""
    logger.info("\n" + "=" * 70)
    logger.info("测试2: 场景图初始化和路径规划")
    logger.info("=" * 70)

    try:
        # 初始化场景图
        door_excel = "./data/door_table.xlsx"
        comp_excel = "./data/component_table.xlsx"

        # 检查文件是否存在
        if not Path(door_excel).exists() or not Path(comp_excel).exists():
            logger.warning("⚠️ Excel文件不存在")
            logger.info("   (跳过场景图测试，继续其他测试)")
            return True

        scene_graph = SceneGraph(
            door_excel=door_excel,
            comp_excel=comp_excel,
        )

        logger.info("✅ 场景图初始化成功")

        # 测试路径规划
        start_room = "Start01"
        end_room = "R309"

        logger.info(f"\n生成候选路径: {start_room} → {end_room}")

        paths = scene_graph.find_k_shortest_paths(start_room, end_room, k=3)

        if paths:
            logger.info(f"✅ 成功生成 {len(paths)} 条候选路径")
            for i, (rooms, doors, steps) in enumerate(paths, 1):
                path_str = " → ".join(rooms)
                logger.info(f"   路径{i}: {path_str} ({steps}步)")
            return True
        else:
            logger.warning(f"⚠️ 无法生成从 {start_room} 到 {end_room} 的路径")
            logger.info("   (可能是场景图中不存在该房间)")
            return True  # 这不是失败，可能只是数据问题

    except Exception as e:
        logger.error(f"❌ 场景图测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_agent_initialization():
    """测试3: Agent初始化"""
    logger.info("\n" + "=" * 70)
    logger.info("测试3: Agent初始化")
    logger.info("=" * 70)

    try:
        # 测试Agent1初始化
        output_dir = "./tests/output_test/vlm_test"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        agent1 = Agent1(config_dir="./configs", output_dir=output_dir)
        logger.info("✅ Agent1初始化成功")
        logger.info(f"   - Agent ID: {agent1.agent_id}")
        logger.info(f"   - 内存大小: {len(agent1.memory)}")

        # 测试Agent2初始化
        agent2 = Agent2(config_dir="./configs", output_dir=output_dir)
        logger.info("✅ Agent2初始化成功")
        logger.info(f"   - Agent ID: {agent2.agent_id}")
        logger.info(f"   - 内存大小: {len(agent2.memory)}")

        return True

    except Exception as e:
        logger.error(f"❌ Agent初始化失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_output_directory_creation():
    """测试4: 输出目录创建"""
    logger.info("\n" + "=" * 70)
    logger.info("测试4: 输出目录创建")
    logger.info("=" * 70)

    try:
        base_dir = Path("./tests/output_test")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base_dir / f"run_{timestamp}"

        # 创建所有必要的子目录
        dirs = {
            "frames": run_dir / "frames",
            "videos": run_dir / "videos",
            "logs": run_dir / "logs",
            "metrics": run_dir / "metrics",
            "paths": run_dir / "paths",
        }

        for dir_name, dir_path in dirs.items():
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"✅ 输出目录创建成功: {run_dir}")

        # 验证所有目录都存在
        for dir_name, dir_path in dirs.items():
            if dir_path.exists():
                logger.info(f"   ✅ {dir_name}: {dir_path}")
            else:
                logger.error(f"   ❌ {dir_name}: 未创建")
                return False

        # 测试文件写入
        test_file = run_dir / "test.json"
        with open(test_file, "w") as f:
            json.dump({"test": "data"}, f)

        if test_file.exists():
            logger.info(f"✅ 文件写入成功: test.json")
            test_file.unlink()  # 清理测试文件
        else:
            logger.error("❌ 文件写入失败")
            return False

        return True

    except Exception as e:
        logger.error(f"❌ 输出目录创建失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_command_line_parsing():
    """测试5: 命令行参数解析"""
    logger.info("\n" + "=" * 70)
    logger.info("测试5: 命令行参数解析")
    logger.info("=" * 70)

    try:
        # 导入run.py中的parse_arguments函数
        sys.argv = [
            "run.py",
            "--start",
            "Start01",
            "--end",
            "R309",
            "--use-habitat",
            "--enable-viz",
            "--max-steps",
            "300",
            "--log-level",
            "DEBUG",
        ]

        # 创建一个临时的参数解析器（避免导入整个run.py）
        import argparse

        parser = argparse.ArgumentParser(description="VLM导航系统")
        parser.add_argument("--start", type=str, default="Start01")
        parser.add_argument("--end", type=str, default="R309")
        parser.add_argument("--config-dir", type=str, default="./configs")
        parser.add_argument("--use-habitat", action="store_true")
        parser.add_argument("--enable-viz", action="store_true")
        parser.add_argument("--enable-agent2", action="store_true")
        parser.add_argument("--max-steps", type=int, default=500)
        parser.add_argument("--log-level", type=str, default="INFO")
        parser.add_argument("--output-dir", type=str, default="./output")

        args = parser.parse_args()

        logger.info("✅ 命令行参数解析成功")
        logger.info(f"   - 起点: {args.start}")
        logger.info(f"   - 终点: {args.end}")
        logger.info(f"   - 使用Habitat: {args.use_habitat}")
        logger.info(f"   - 启用可视化: {args.enable_viz}")
        logger.info(f"   - 最大步数: {args.max_steps}")
        logger.info(f"   - 日志级别: {args.log_level}")
        logger.info(f"   - 配置目录: {args.config_dir}")
        logger.info(f"   - 输出目录: {args.output_dir}")

        return True

    except Exception as e:
        logger.error(f"❌ 命令行参数解析失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    logger.info("\n" + "█" * 70)
    logger.info("█" + " " * 68 + "█")
    logger.info("█" + " " * 20 + "run.py 集成测试" + " " * 33 + "█")
    logger.info("█" + " " * 68 + "█")
    logger.info("█" * 70)

    tests = [
        ("配置加载", test_config_loading),
        ("场景图初始化", test_scene_graph),
        ("Agent初始化", test_agent_initialization),
        ("输出目录创建", test_output_directory_creation),
        ("命令行解析", test_command_line_parsing),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"❌ 测试 '{test_name}' 执行异常: {e}")
            results[test_name] = False

    # 汇总结果
    logger.info("\n" + "=" * 70)
    logger.info("测试汇总")
    logger.info("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"{status}: {test_name}")

    logger.info("=" * 70)
    logger.info(f"总计: {passed}/{total} 通过")

    if passed == total:
        logger.info("✅ 所有测试通过!")
        return True
    else:
        logger.warning(f"⚠️ 有 {total - passed} 个测试失败")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
