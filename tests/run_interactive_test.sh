#!/bin/bash
# Habitat交互式测试快速启动脚本

echo "========================================="
echo "Habitat 环境交互式测试"
echo "========================================="
echo ""
echo "准备启动可视化测试..."
echo "控制说明:"
echo "  W/↑ : 前进    S/↓ : 后退"
echo "  A/← : 左转    D/→ : 右转"
echo "  R   : 重置    Q   : 退出"
echo "========================================="
echo ""

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit 1

# 检查Habitat是否安装
if ! python3 -c "import habitat_sim" 2>/dev/null; then
    echo "❌ 错误: Habitat-sim 未安装"
    echo ""
    echo "请先安装 Habitat-sim:"
    echo "  conda install habitat-sim -c conda-forge -c aihabitat"
    exit 1
fi

echo "✅ Habitat-sim 已安装"
echo ""

# 运行交互式测试
python3 tests/test_habitat_interactive.py

echo ""
echo "测试结束"
