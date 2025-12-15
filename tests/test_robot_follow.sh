#!/bin/bash
# 机器狗跟随功能测试脚本

echo "============================================================"
echo "机器狗跟随功能测试"
echo "============================================================"
echo ""
echo "此脚本将验证以下修复:"
echo "  ✓ 机器狗朝向随Agent转向同步"
echo "  ✓ 机器狗可以跟随到二楼/三楼"
echo ""
echo "测试步骤:"
echo "  1. 启动增强查看器"
echo "  2. 按 W 前进几步"
echo "  3. 按 A 左转 - 观察日志中的 Robot rot 变化"
echo "  4. 按 D 右转 - 观察日志中的 Robot rot 变化"
echo "  5. 按 Alt+N 随机采样位置（可能在高层）"
echo "  6. 查看日志中 Agent 和 Robot 的 y 坐标是否相同"
echo ""
echo "============================================================"
echo ""

# 检查文件是否存在
if [ ! -f "tests/test_habitat_interactive_enhanced.py" ]; then
    echo "❌ 错误: 未找到 test_habitat_interactive_enhanced.py"
    exit 1
fi

echo "🚀 启动增强查看器..."
echo "   (日志中会显示 Agent 和 Robot 的位置/旋转)"
echo ""

# 运行测试
python3 tests/test_habitat_interactive_enhanced.py

echo ""
echo "============================================================"
echo "测试完成"
echo "============================================================"
echo ""
echo "验证要点:"
echo "  1. 日志中 'TURN LEFT/RIGHT' 后应显示 'Robot rot: ...'"
echo "  2. 日志中 'FORWARD/BACKWARD' 后应显示 'Robot: (x, y, z)'"
echo "  3. Agent 和 Robot 的坐标应完全相同"
echo "  4. 在高楼层（y > 2.0）时，Robot 也应该在相同高度"
echo ""
