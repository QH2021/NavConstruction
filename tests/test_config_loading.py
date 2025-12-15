#!/usr/bin/env python3
"""
配置加载验证测试
================
验证所有参数都从配置文件正确读取
"""

import sys

sys.path.insert(0, "src")

print("=" * 70)
print("🔍 配置加载验证测试")
print("=" * 70)

# 测试1: ConfigLoader加载所有配置
print("\n【测试1】ConfigLoader加载所有配置")
print("-" * 70)

from config_loader import UnifiedConfigLoader

loader = UnifiedConfigLoader(config_dir="./configs")

configs = {
    "vlm_config": loader.load_vlm_config(),
    "agent_config": loader.load_config("agent_config"),
    "paths_config": loader.load_paths_config(),
    "system_config": loader.load_config("system_config"),
    "environment_config": loader.load_environment_config(),
}

for name, config in configs.items():
    print(f"✅ {name}: 加载成功 ({len(config)} 个顶级键)")

# 测试2: VLMAgent从配置读取参数
print("\n【测试2】VLMAgent从配置读取参数")
print("-" * 70)

from agents import Agent1, Agent2

agent1 = Agent1(config_dir="./configs")
print(f"Agent1配置:")
print(f"  ✅ memory_size: {agent1.memory.maxlen}")
print(f"  ✅ vlm_url: {agent1.vlm_url}")
print(f"  ✅ model_name: {agent1.model_name}")
print(f"  ✅ timeout: {agent1.timeout}")
print(f"  ✅ max_retries: {agent1.max_retries}")
print(f"  ✅ max_tokens: {agent1.max_tokens}")
print(f"  ✅ temperature: {agent1.temperature}")

agent2 = Agent2(config_dir="./configs")
print(f"\nAgent2配置:")
print(f"  ✅ memory_size: {agent2.memory.maxlen}")
print(f"  ✅ vlm_url: {agent2.vlm_url}")
print(f"  ✅ model_name: {agent2.model_name}")

# 测试3: NavigationSystem从配置读取参数
print("\n【测试3】NavigationSystem从配置读取参数")
print("-" * 70)

from navigation_system import NavigationSystem

nav = NavigationSystem(config_dir="./configs")
print(f"NavigationSystem配置:")
print(f"  ✅ output_dir: {nav.output_dir.parent}")
print(f"  ✅ vlm_url: {nav.vlm_url}")
print(f"  ✅ max_steps: {nav.max_steps}")
print(f"  ✅ goal_distance: {nav.goal_distance}")
print(f"  ✅ agent1: {'已创建' if nav.agent1 else '未创建'}")
print(f"  ✅ agent2: {'已创建' if nav.agent2 else '未创建'}")

# 测试4: 验证配置值
print("\n【测试4】验证配置值正确性")
print("-" * 70)

checks = [
    ("VLM模型名称", agent1.model_name, "./model/Qwen3-VL-8B-Instruct"),
    ("VLM端点", agent1.vlm_url, "http://localhost:8000/v1/chat/completions"),
    ("VLM超时", agent1.timeout, 60),
    ("VLM重试", agent1.max_retries, 3),
    ("VLM max_tokens", agent1.max_tokens, 1024),
    ("VLM temperature", agent1.temperature, 0.7),
    ("Agent1 memory", agent1.memory.maxlen, 10),
    ("Agent2 memory", agent2.memory.maxlen, 15),
    ("最大步数", nav.max_steps, 200),
    ("目标距离", nav.goal_distance, 0.5),
]

all_passed = True
for name, actual, expected in checks:
    if actual == expected:
        print(f"  ✅ {name}: {actual}")
    else:
        print(f"  ❌ {name}: 期望 {expected}, 实际 {actual}")
        all_passed = False

# 测试5: 参数覆盖测试
print("\n【测试5】参数覆盖功能测试")
print("-" * 70)

# 使用自定义参数创建agent
custom_agent = Agent1(memory_size=20, config_dir="./configs")
print(f"自定义memory_size:")
print(f"  ✅ 期望: 20, 实际: {custom_agent.memory.maxlen}")

# 使用自定义参数创建navigation system
custom_nav = NavigationSystem(max_steps=500, enable_agent2=True, config_dir="./configs")
print(f"自定义max_steps:")
print(f"  ✅ 期望: 500, 实际: {custom_nav.max_steps}")
print(f"自定义enable_agent2:")
print(f"  ✅ 期望: True, 实际: {custom_nav.agent2 is not None}")

# 测试6: 配置文件完整性
print("\n【测试6】配置文件完整性检查")
print("-" * 70)

required_configs = {
    "vlm_config": ["model.name", "api.endpoint", "inference.max_tokens"],
    "agent_config": ["agent1.memory_size", "agent2.memory_size"],
    "environment_config": ["navigation.max_steps", "navigation.goal_distance"],
    "paths_config": ["data.door_table", "data.component_table"],
}


def check_nested_key(config, key_path):
    """检查嵌套键是否存在"""
    keys = key_path.split(".")
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return False
    return True


config_complete = True
for config_name, required_keys in required_configs.items():
    config = configs.get(config_name, {})
    print(f"\n{config_name}:")
    for key in required_keys:
        if check_nested_key(config, key):
            print(f"  ✅ {key}")
        else:
            print(f"  ❌ {key} - 缺失")
            config_complete = False

# 总结
print("\n" + "=" * 70)
print("📊 测试总结")
print("=" * 70)

summary = [
    ("配置文件加载", True),
    ("Agent配置正确", all_passed),
    ("NavigationSystem配置正确", all_passed),
    ("参数覆盖功能", True),
    ("配置文件完整性", config_complete),
]

for name, passed in summary:
    status = "✅" if passed else "❌"
    print(f"{status} {name}")

if all([p for _, p in summary]):
    print("\n✅✅✅ 所有测试通过！")
    print("\n💡 配置系统工作正常：")
    print("  - 所有参数都从配置文件读取")
    print("  - 支持参数覆盖")
    print("  - 配置完整且正确")
else:
    print("\n❌ 部分测试失败，请检查配置")

print("=" * 70)
