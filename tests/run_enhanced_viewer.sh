#!/bin/bash
# Habitat增强交互式查看器启动脚本
# 功能: NavMesh计算、机器狗跟随、可视化控制

echo "============================================================"
echo "Habitat增强交互式查看器"
echo "============================================================"
echo ""
echo "功能特性:"
echo "  ✓ NavMesh可通行区域计算与可视化"
echo "  ✓ 机器狗模型实时跟随Agent"
echo "  ✓ WASD键盘控制"
echo "  ✓ 实时统计信息显示"
echo ""
echo "控制说明:"
echo "  W       : 前进"
echo "  S       : 后退"
echo "  A       : 左转"
echo "  D       : 右转"
echo "  N       : 切换NavMesh可视化"
echo "  Shift+N : 重新计算NavMesh"
echo "  Alt+N   : 随机采样新位置"
echo "  ESC     : 退出"
echo ""
echo "============================================================"
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到python3"
    exit 1
fi

# 检查Habitat-sim
python3 -c "import habitat_sim" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 错误: 未安装habitat-sim"
    echo "   请先安装: conda install habitat-sim -c conda-forge -c aihabitat"
    exit 1
fi

# 检查配置文件
if [ ! -d "configs" ]; then
    echo "❌ 错误: 未找到configs目录"
    echo "   请确保在项目根目录运行此脚本"
    exit 1
fi

# 运行增强查看器
echo "🚀 启动增强查看器..."
echo ""

python3 tests/test_habitat_interactive_enhanced.py

echo ""
echo "============================================================"
echo "查看器已关闭"
echo "============================================================"
