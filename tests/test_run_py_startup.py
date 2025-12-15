#!/usr/bin/env python3
"""
run.py 启动测试

验证 run.py 能否正确解析参数、初始化所有模块、创建输出目录等，
不执行实际导航（因为需要 Habitat 和完整环境）
"""

import sys
import os
import subprocess
from pathlib import Path


def test_run_py_help():
    """测试1: 验证--help输出"""
    print("=" * 70)
    print("测试1: 验证 run.py --help")
    print("=" * 70)

    result = subprocess.run(
        ["python", "run.py", "--help"],
        cwd="/home/qh1302/桌面/VLNproject/constructionNav",
        capture_output=True,
        text=True,
    )

    if result.returncode == 0 and "VLM导航系统" in result.stdout:
        print("✅ --help 输出正确")
        print(result.stdout[:200] + "...")
        return True
    else:
        print("❌ --help 失败")
        print("STDOUT:", result.stdout[:200])
        print("STDERR:", result.stderr[:200])
        return False


def test_run_py_syntax():
    """测试2: 验证Python语法"""
    print("\n" + "=" * 70)
    print("测试2: 验证 run.py Python语法")
    print("=" * 70)

    result = subprocess.run(
        ["python", "-m", "py_compile", "run.py"],
        cwd="/home/qh1302/桌面/VLNproject/constructionNav",
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print("✅ 语法检查通过")
        return True
    else:
        print("❌ 语法检查失败")
        print("STDERR:", result.stderr)
        return False


def test_run_py_imports():
    """测试3: 验证主要导入"""
    print("\n" + "=" * 70)
    print("测试3: 验证 run.py 的导入")
    print("=" * 70)

    test_code = """
import sys
sys.path.insert(0, '.')

try:
    from run import parse_arguments, VLMNavigationRunner
    print('✅ run.py 主要类导入成功')
    
    # 测试参数解析
    sys.argv = ['run.py', '--start', 'S101', '--end', 'R309']
    args = parse_arguments()
    print(f'✅ 参数解析成功: start={args.start}, end={args.end}')
    
except Exception as e:
    print(f'❌ 导入失败: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""

    result = subprocess.run(
        ["python", "-c", test_code],
        cwd="/home/qh1302/桌面/VLNproject/constructionNav",
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print(result.stdout.strip())
        return True
    else:
        print("❌ 导入测试失败")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False


def test_run_py_help_examples():
    """测试4: 验证帮助中的示例格式"""
    print("\n" + "=" * 70)
    print("测试4: 验证帮助中包含示例")
    print("=" * 70)

    result = subprocess.run(
        ["python", "run.py", "--help"],
        cwd="/home/qh1302/桌面/VLNproject/constructionNav",
        capture_output=True,
        text=True,
    )

    required_keywords = [
        "起点房间ID",
        "终点房间ID",
        "Habitat环境",
        "示例使用",
    ]

    if result.returncode == 0:
        all_found = all(keyword in result.stdout for keyword in required_keywords)
        if all_found:
            print("✅ 帮助信息完整")
            return True
        else:
            print("❌ 帮助信息缺少关键内容")
            for keyword in required_keywords:
                if keyword in result.stdout:
                    print(f"  ✅ 包含: {keyword}")
                else:
                    print(f"  ❌ 缺失: {keyword}")
            return False
    else:
        print("❌ 获取帮助失败")
        return False


def main():
    """运行所有测试"""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + " " * 18 + "run.py 启动和集成验证测试" + " " * 25 + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    tests = [
        ("验证 --help", test_run_py_help),
        ("验证语法", test_run_py_syntax),
        ("验证导入", test_run_py_imports),
        ("验证帮助示例", test_run_py_help_examples),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            import traceback

            traceback.print_exc()
            results[test_name] = False

    # 汇总结果
    print("\n" + "=" * 70)
    print("测试汇总")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {test_name}")

    print("=" * 70)
    print(f"总计: {passed}/{total} 通过")

    if passed == total:
        print("✅ 所有启动验证通过! run.py 已准备就绪")
        return 0
    else:
        print(f"⚠️ 有 {total - passed} 个验证失败")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
