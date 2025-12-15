#!/usr/bin/env python3
"""VLM 服务连接测试（可选）

说明：
- 原脚本会在 import 时直接发 HTTP 请求，导致 `pytest` 在未启动 VLM 服务的环境中失败。
- 现在改为 pytest 测试函数：若无法连接，则跳过（skip），以保证离线/CI 环境稳定。
"""

from pathlib import Path

import pytest
import requests
import yaml


def test_vlm_connection_optional():
    config_file = Path("configs/unified_config.yaml")
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    model_name = (config.get("model", {}) or {}).get("name")
    api_endpoint = (config.get("api", {}) or {}).get("endpoint")
    timeout = float((config.get("api", {}) or {}).get("timeout", 3))

    if not model_name or not api_endpoint:
        pytest.skip("VLM 配置不完整（可选测试）")

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "你好，请回答：1+1=?"}],
            }
        ],
        "max_tokens": 16,
        "temperature": 0.1,
    }

    try:
        response = requests.post(
            api_endpoint,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
    except requests.exceptions.ConnectionError:
        pytest.skip(f"VLM 服务未启动或不可达: {api_endpoint}")
    except requests.exceptions.Timeout:
        pytest.skip(f"VLM 请求超时(>{timeout}s): {api_endpoint}")

    if response.status_code in (404, 503):
        pytest.skip(f"VLM 服务不可用 (HTTP {response.status_code})")

    assert response.status_code == 200

    data = response.json()
    assert "choices" in data
