# VLM 导航系统（Habitat / Mock）

基于 Habitat-sim（可选）+ OpenAI 兼容 VLM 接口的视觉导航系统：支持两帧 RGB 输入、可选同位 Depth 输入与楼层平面图输入；在无可用 EGL/GPU 时可自动降级到 Mock 环境以保证流程与测试可运行。

## 📋 项目概述

本项目主要入口为 [run.py](run.py)。核心特性：

- **统一配置**：以 [configs/unified_config.yaml](configs/unified_config.yaml) 为唯一权威配置入口
- **可选 Habitat-sim**：可用时使用真实仿真；不可用时自动回退 Mock（避免进程级崩溃）
- **两帧前置 RGB**：每次决策输入 $(t-1,t)$ 两帧，修复“只给最后一帧”的问题
- **可选两帧同位 Depth**：启用后将 $(t-1,t)$ 两帧 Depth（可视化成灰度）与 RGB 一起喂给 VLM
- **VLM OpenAI 兼容接口**：以 `chat/completions` 形式发送 text + image_url(data URL)
- **输出留痕**：运行目录内保存 VLM 输入/输出、帧、路径、指标等

## 🚀 快速开始

### 1) 安装依赖

```bash
pip install -r requirements.txt

# 可选：安装 habitat-sim（按你的机器/驱动选择安装方式）
# conda install habitat-sim -c conda-forge -c aihabitat
```

注意：尽量避免同时安装 `conda opencv` 与 `pip opencv-python`（容易产生 `cv2` 冲突）。

### 2) 运行（Mock / Habitat）

```bash
# 纯 Mock（不依赖 habitat-sim / EGL），用于快速跑通流程
python run.py --no-habitat --start S101 --end R309

# 使用 Habitat（默认创建 windowless 环境；若 EGL/GPU 不可用会自动降级 Mock）
python run.py --use-habitat --start S101 --end R309
```

更多参数见：`python run.py --help`。

## 🎯 配置文件说明

### configs/unified_config.yaml（推荐，单一配置入口）

项目以 [configs/unified_config.yaml](configs/unified_config.yaml) 为唯一权威配置来源。

为兼容旧代码/旧测试，`src/config_loader.py` 会在缺少独立的 `environment_config.yaml / agent_config.yaml / vlm_config.yaml / system_config.yaml / paths_config.yaml / prompts_config.yaml` 时，自动从 `unified_config.yaml` 派生对应配置结构。

下面仅展示关键片段（完整字段见 [configs/unified_config.yaml](configs/unified_config.yaml)）：

```yaml
environment:
  scene:
    path: './data/scene_datasets/habitat-test-scenes/3dExport1212f.glb'
  agent:
    height: 0.55
    radius: 0.28
  sensors:
    front_camera:
      enabled: true
      uuid: 'front_rgb'
      resolution: [720, 1280]
      hfov: 90.0
      position: [0.0, 0.55, -0.6]

    # ✅ 新增：前置深度摄像头（与前置RGB同位）
    front_depth:
      enabled: false
      uuid: 'front_depth'
      resolution: [720, 1280]
      hfov: 90.0
      position: [0.0, 0.55, -0.6]

    back_camera:
      enabled: true
      uuid: 'top_down_view'
      resolution: [480, 640]
      hfov: 90.0
      position: [0.0, 1.6, 1.0]
      orientation: [-0.6283185307, 0.0, 0.0]
  actions:
    move_forward: { amount: 0.25 }
    turn_left: { amount: 10.0 }
    turn_right: { amount: 10.0 }

vlm:
  api:
    endpoint: 'http://localhost:8000/v1/chat/completions'
    timeout: 60
    connect_timeout: 10
  inference:
    max_tokens: 1024
    temperature: 0.7
  image_processing:
    enabled: true
    max_size: [512, 512]
    quality: 70
    floorplan_max_size: [512, 512]
    floorplan_quality: 70

    # Depth 可视化（仅在启用 front_depth 时使用）
    depth:
      max_depth_m: 10.0
      invert: true
```

## 🧠 VLM 图像输入（两帧 + 可选 Depth）

系统在每次 VLM 决策时，使用“最近两帧”的前置观测 $(t-1, t)$，而不是只给最后一帧。

- 默认（Depth 关闭）：RGB$(t-1)$、RGB$(t)$、floorplan
- Depth 开启：RGB$(t-1)$、RGB$(t)$、Depth$(t-1)$、Depth$(t)$、floorplan

Depth 会在发送前被转换为灰度可视化图（默认“近亮远暗”），帮助模型判断障碍距离、门洞/走廊的可通行性。

## 📁 输出目录

每次运行会在 `output/run_YYYYMMDD_HHMMSS/` 下生成独立目录，常见内容包括：

- `frames/`：逐步保存的观测帧
- `paths/`：路径/轨迹相关文件
- `metrics/`：评估与统计
- `vlm_io.json`：VLM 输入/输出留痕（含图片 meta、模型回复等）

## ✅ 测试

推荐在安装了 `habitat_sim` / `cv2` 的环境中运行（例如 conda env `habitat`）：

```bash
conda run -n habitat --no-capture-output pytest
```

说明：

- [tests/test_end_to_end.py](tests/test_end_to_end.py) 的 Habitat 创建测试会在子进程中探测 EGL/渲染可用性；不可用时会跳过，避免 pytest 进程被 kill。
- [tests/test_habitat_interactive_enhanced.py](tests/test_habitat_interactive_enhanced.py) 是交互式 viewer 脚本，自动化 pytest 默认跳过；如需在 pytest 中启用，可设置 `RUN_HABITAT_INTERACTIVE_ENHANCED=1`。

## 📁 项目结构

```
constructionNav/
├── run.py                          # ⭐ 主运行入口
├── requirements.txt                 # Python 依赖
├── README.md                        # 本文档
│
├── configs/                         # 配置文件目录
│   └── unified_config.yaml         # ⭐ 单一配置入口（其余 *_config.yaml 可选）
│
├── src/                             # 核心代码
│   ├── config_loader.py            # 统一配置加载器
│   ├── agents.py                   # Agent1 和 Agent2 实现
│   ├── habitat_integration.py      # Habitat 集成层
│   ├── navigation_system.py        # 导航系统主控制器
│   ├── scene_graph.py              # 场景图和路径规划
│   ├── video_recorder.py           # 视频录制
│   └── __init__.py                 # 模块初始化
│
├── tests/                           # 测试目录
│   ├── test_habitat_interactive_enhanced.py  # 交互式 viewer（pytest 默认跳过）
│   ├── test_config_loading.py      # 配置加载测试
│   ├── test_end_to_end.py          # 端到端测试
│   ├── test_vlm_connection.py      # VLM 连接测试
│   └── test_habitat_interactive.py.bak
│
├── data/                            # 数据目录
│   ├── robots/                      # 机器人 URDF 模型
│   ├── scene_datasets/              # Habitat 场景数据
│   ├── map/                         # 楼层平面图
│   ├── door_table.xlsx              # 门的元数据
│   ├── component_table.xlsx         # 组件清单
│   └── ...                          # 其他数据文件
│
└── output/                          # 输出目录（运行时生成）
    └── run_YYYYMMDD_HHMMSS/        # 每次运行的输出
        ├── frames/
        ├── videos/
        ├── logs/
        ├── paths/
        └── metrics/
```

## 🔧 技术细节

### NavMesh 计算原理

NavMesh（导航网格）是一个表示可通行区域的多边形网格。系统会：

1. 根据场景几何信息计算 NavMesh
2. 使用 Agent 的高度和半径参数
3. 自动避免静态障碍物
4. 支持实时可视化和重新计算

### 机器狗跟随机制

机器狗（通过 URDF 模型加载）会：

1. 实时跟踪 Agent 的位置（+ 0.6m 高度偏移）
2. 同步 Agent 的身体旋转
3. 自动对齐初始朝向（支持手动调整）
4. 通过四元数插值实现平滑运动

### 四元数操作

系统使用标准的四元数 [w, x, y, z] 表示旋转，支持多种格式转换：
- NumPy 数组
- Magnum Quaternion
- Python quaternion 模块
- 迭代序列

## 🐛 常见问题

### Q1: 报错 `WindowlessContext: Unable to create windowless context`？

这是典型的 EGL/驱动/GPU 选择问题。

- 只想跑通流程/开发逻辑：直接用 `python run.py --no-habitat ...`（Mock 环境）
- 希望使用 Habitat：需要机器具备可用的 EGL + GPU 渲染环境；必要时尝试设置 `HABITAT_SIM_GPU_DEVICE_ID`（不同机器可用值不同）
- 自动化测试：会在 EGL/渲染不可用时跳过相关“可选 Habitat 创建测试”，避免 pytest 进程崩溃

### Q2: `cv2` 导入失败？

确保当前运行的 Python 环境里安装了 `opencv-python`，并避免与 `conda opencv` 双装冲突。

### Q3: 如何启用 Depth 输入？

在 [configs/unified_config.yaml](configs/unified_config.yaml) 中设置：

- `environment.sensors.front_depth.enabled: true`

### Q4: 如何修改场景？

编辑 [configs/unified_config.yaml](configs/unified_config.yaml) 中的 `environment.scene.path` 字段，指向不同的 `.glb` 场景文件。

### Q5: 交互式增强 Viewer 如何运行？

该脚本位于 [tests/test_habitat_interactive_enhanced.py](tests/test_habitat_interactive_enhanced.py)，建议作为脚本单独运行：

```bash
python tests/test_habitat_interactive_enhanced.py
```

## 📝 开发规范

- 所有参数通过配置文件管理，避免硬编码
- 使用类型注解（Type Hints）
- 遵循 PEP 8 代码风格
- 实现清晰的日志输出
- 提供异常处理和错误恢复机制

## 📄 许可证

本项目仅供学术研究使用。

## 🙏 致谢

- Habitat-sim 团队（Facebook AI Research）
- Qwen3-VL 模型（阿里巴巴）
