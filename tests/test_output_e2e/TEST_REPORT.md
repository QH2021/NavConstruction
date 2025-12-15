# 端到端测试报告

**测试时间**: 2025-12-15 17:26:18

**测试时长**: 1.41 秒

## 测试统计

- **总测试数**: 8
- **通过**: 7 ✅
- **失败**: 0 ❌
- **跳过**: 1 ⏭️
- **警告**: 0 ⚠️

## 详细结果

### ✅ config_loading

**状态**: PASS

**时间**: 2025-12-15T17:26:18.090382

**详情**:

```json
{
  "configs_loaded": [
    "environment",
    "agent",
    "vlm",
    "paths",
    "system"
  ],
  "scene_path": "./data/scene_datasets/habitat-test-scenes/3dExport1212f.glb",
  "robot_urdf": "./data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf",
  "checks": {
    "场景路径配置": true,
    "机器人URDF配置": true,
    "前置摄像头配置": true,
    "后置摄像头配置": true,
    "Agent1配置": true,
    "Agent2配置": true,
    "VLM API配置": true
  }
}
```

---

### ✅ scene_files_existence

**状态**: PASS

**时间**: 2025-12-15T17:26:18.107272

**详情**:

```json
{
  "scene_path": "./data/scene_datasets/habitat-test-scenes/3dExport1212f.glb",
  "robot_urdf": "./data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf",
  "files_check": {
    "场景文件": true,
    "机器人URDF": true
  }
}
```

---

### ✅ mock_environment_creation

**状态**: PASS

**时间**: 2025-12-15T17:26:18.113813

**详情**:

```json
{
  "observations_check": {
    "rgb_front存在": true,
    "rgb_back存在": true,
    "agent_pos存在": true,
    "agent_rot存在": true,
    "rgb_front形状正确": true,
    "rgb_back形状正确": true,
    "agent_pos维度正确": true
  },
  "step_works": true,
  "navigable_map_available": true,
  "navigable_map_shape": [
    200,
    200
  ]
}
```

---

### ⏭️ habitat_environment_creation

**状态**: SKIP

**时间**: 2025-12-15T17:26:18.902161

**详情**:

```json
{
  "reason": "Habitat-sim EGL/GL context 不可用（子进程探测失败）",
  "scene_path": "./data/scene_datasets/habitat-test-scenes/3dExport1212f.glb"
}
```

---

### ✅ scene_graph_path_planning

**状态**: PASS

**时间**: 2025-12-15T17:26:18.969997

**详情**:

```json
{
  "checks": {
    "场景图创建成功": true,
    "找到候选路径": true,
    "路径数量正确": true
  },
  "paths_found": 3,
  "path_details": [
    {
      "path_id": 1,
      "rooms": [
        "Start01",
        "S101",
        "S201",
        "S301",
        "H303",
        "H302",
        "H305",
        "H306",
        "R309"
      ],
      "length": 9,
      "cost": 8
    },
    {
      "path_id": 2,
      "rooms": [
        "Start01",
        "S101",
        "H103",
        "H102",
        "H105",
        "S102",
        "S202",
        "S302",
        "H305",
        "H306",
        "R309"
      ],
      "length": 11,
      "cost": 10
    },
    {
      "path_id": 3,
      "rooms": [
        "Start01",
        "S101",
        "S201",
        "H203",
        "H202",
        "H205",
        "S202",
        "S302",
        "H305",
        "H306",
        "R309"
      ],
      "length": 11,
      "cost": 10
    }
  ]
}
```

---

### ✅ agent_initialization

**状态**: PASS

**时间**: 2025-12-15T17:26:19.004840

**详情**:

```json
{
  "checks": {
    "Agent1创建成功": true,
    "Agent2创建成功": true,
    "Agent1有记忆": true,
    "Agent2有记忆": true,
    "Agent1有VLM配置": true,
    "Agent1有输出目录": true
  },
  "memory_works": true,
  "stats_works": true,
  "agent1_stats": {
    "agent_id": "Agent1",
    "success_count": 0,
    "failure_count": 0,
    "memory_size": 1,
    "last_action": null
  }
}
```

---

### ✅ full_navigation_mock

**状态**: PASS

**时间**: 2025-12-15T17:26:19.306198

**详情**:

```json
{
  "checks": {
    "Agent1被调用": true,
    "环境执行了动作": true,
    "RGB前置图像已保存": true,
    "RGB后置图像已保存": true,
    "视频文件已创建": true
  },
  "total_steps": 10,
  "front_images_count": 3,
  "back_images_count": 10,
  "videos_count": 1,
  "agent1_call_count": 3
}
```

---

### ✅ vlm_io_recording

**状态**: PASS

**时间**: 2025-12-15T17:26:19.422835

**详情**:

```json
{
  "checks": {
    "VLM调用成功": true,
    "VLM记录文件创建": true,
    "VLM记录包含records": true
  },
  "records_count": 1,
  "vlm_io_sample": {
    "timestamp": "2025-12-15T17:26:19.321837",
    "agent_id": "Agent1",
    "attempt": 1,
    "input": {
      "prompt": "Test prompt for navigation",
      "num_images": 1,
      "images_meta": [
        {
          "index": 0,
          "data_url_len": 38,
          "mime": "image/jpeg"
        }
      ]
    },
    "output": {
      "text": "{\"actions\": [\"move_forward\"], \"reached_goal\": false, \"reasoning\": \"test\"}"
    },
    "meta": {
      "vlm_url": "http://localhost:8000/v1/chat/completions",
      "model": "./model/Qwen3-VL-8B-Instruct",
      "timeout": {
        "connect": 10.0,
        "read": 60.0
      },
      "max_tokens": 1024,
      "temperature": 0.7,
      "http_status": 200,
      "elapsed_s": 2.8848648071289062e-05,
      "trace_meta": {}
    }
  }
}
```

---

## 判断建议

✅ **所有测试通过** - 系统功能正常

### 关键检查项

请人工检查以下内容：

1. 配置文件是否正确加载 ✓
2. 场景文件和机器人模型是否存在（如需要）
3. Mock环境是否正常工作 ✓
4. Habitat环境是否成功创建（可选）
5. 路径规划是否生成合理路径 ✓
6. Agent初始化是否成功 ✓
7. 完整导航流程是否执行 ✓
8. RGB图像是否保存 ✓
9. VLM输入输出是否记录 ✓
10. 视频文件是否生成 ✓

### 输出文件位置

- **测试报告**: `tests/test_output_e2e/test_report.json`
- **日志文件**: `tests/test_output_e2e/test_report.log`
- **导航输出**: `tests/test_output_e2e/navigation_test`

