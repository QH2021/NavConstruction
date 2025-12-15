"""
==================================================================================
配置加载器 (Config Loader)
==================================================================================

该模块提供统一的配置加载、验证和管理接口，支持：
- 从 YAML/JSON 文件加载配置
- 配置验证和错误处理
- 环境变量覆盖
- 默认值回退
- 配置合并和继承
- 类型转换和验证

作者: 构造导航系统
日期: 2024
版本: 2.0.0
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import hashlib
import copy


# ============================================================================
# 日志配置
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# 类型定义
# ============================================================================

ConfigType = Union[Dict[str, Any], str, int, float, bool, List, None]


# ============================================================================
# 异常类
# ============================================================================


class ConfigException(Exception):
    """配置异常基类"""

    pass


class ConfigLoadException(ConfigException):
    """配置加载异常"""

    pass


class ConfigValidationException(ConfigException):
    """配置验证异常"""

    pass


class ConfigMissingException(ConfigException):
    """配置缺失异常"""

    pass


# ============================================================================
# 枚举
# ============================================================================


class ConfigSource(Enum):
    """配置来源"""

    FILE = "file"
    ENV = "env"
    DEFAULT = "default"
    OVERRIDE = "override"


class ConfigFormat(Enum):
    """配置格式"""

    YAML = "yaml"
    JSON = "json"
    ENV = "env"


# ============================================================================
# 数据类
# ============================================================================


@dataclass
class ConfigMetadata:
    """配置元数据"""

    # 配置文件路径
    file_path: Optional[str] = None
    # 加载时间戳
    loaded_at: Optional[str] = None
    # 配置哈希值
    config_hash: Optional[str] = None
    # 配置来源
    source: ConfigSource = ConfigSource.FILE
    # 是否已验证
    validated: bool = False
    # 验证错误
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class ConfigLoader:
    """配置加载器配置"""

    # 配置目录
    config_dir: str = "./configs"
    # 是否启用缓存
    enable_cache: bool = True
    # 是否启用环境变量覆盖
    enable_env_override: bool = True
    # 是否严格验证
    strict_validation: bool = False
    # 默认配置源优先级
    source_priority: List[ConfigSource] = field(
        default_factory=lambda: [
            ConfigSource.OVERRIDE,
            ConfigSource.ENV,
            ConfigSource.FILE,
            ConfigSource.DEFAULT,
        ]
    )


# ============================================================================
# 配置加载器主类
# ============================================================================


class UnifiedConfigLoader:
    """
    统一配置加载器

    功能：
    - 支持多种配置源（YAML、JSON、环境变量、默认值）
    - 自动配置验证和类型转换
    - 配置缓存和性能优化
    - 配置继承和合并
    """

    def __init__(
        self,
        config_dir: str = "./configs",
        enable_cache: bool = True,
        strict_validation: bool = False,
    ):
        """
        初始化配置加载器

        参数：
            config_dir: 配置文件目录
            enable_cache: 是否启用缓存
            strict_validation: 是否启用严格验证
        """
        self.config_dir = Path(config_dir)
        self.enable_cache = enable_cache
        self.strict_validation = strict_validation

        # 配置缓存
        self._cache: Dict[str, Dict[str, Any]] = {}
        # 配置哈希缓存
        self._hash_cache: Dict[str, str] = {}
        # 已加载的配置文件
        self._loaded_files: Dict[str, ConfigMetadata] = {}

        # 验证配置目录
        self._validate_config_dir()

        logger.info(f"配置加载器初始化: config_dir={config_dir}")

    def _validate_config_dir(self) -> None:
        """验证配置目录"""
        if not self.config_dir.exists():
            logger.warning(f"配置目录不存在: {self.config_dir}")
            # 创建目录
            try:
                self.config_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"创建配置目录: {self.config_dir}")
            except Exception as e:
                logger.error(f"创建配置目录失败: {e}")

    def _load_unified_raw(self) -> Optional[Dict[str, Any]]:
        """直接读取 unified_config.yaml（不走 load_config 以避免递归）。

        Returns:
            dict 或 None（不存在/解析失败）
        """
        unified_path = self.config_dir / "unified_config.yaml"
        if not unified_path.exists():
            return None
        try:
            cfg = self._load_yaml(unified_path)
            return cfg if isinstance(cfg, dict) else None
        except Exception as e:
            logger.warning(f"读取 unified_config.yaml 失败: {e}")
            return None

    def _derive_legacy_config_from_unified(
        self, config_name: str, unified: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """从 unified_config 派生旧式 *_config 结构，以保持向后兼容。

        典型映射：
        - environment_config -> unified.environment (+ 合并 unified.navigation.navigation_loop 到 environment.navigation)
        - agent_config -> unified.agents
        - vlm_config -> unified.vlm
        - system_config -> unified.system
        - paths_config -> unified.paths
        - prompts_config -> unified.prompts
        - navigation_config -> unified.navigation
        """
        if not isinstance(unified, dict):
            return None

        # 允许传入不带后缀的名字
        key = config_name
        if key.endswith(".yaml"):
            key = key[: -len(".yaml")]
        if key.endswith(".json"):
            key = key[: -len(".json")]

        mapping = {
            "environment_config": "environment",
            "agent_config": "agents",
            "vlm_config": "vlm",
            "system_config": "system",
            "paths_config": "paths",
            "prompts_config": "prompts",
            "navigation_config": "navigation",
        }

        if key not in mapping:
            return None

        section_name = mapping[key]
        section = unified.get(section_name, {})
        if not isinstance(section, dict):
            return None

        derived = copy.deepcopy(section)

        # environment_config 需要兼容旧测试里 environment_config.navigation.*
        if key == "environment_config":
            nav_loop = (unified.get("navigation", {}) or {}).get("navigation_loop", {})
            if isinstance(nav_loop, dict) and nav_loop:
                derived.setdefault("navigation", {})
                if isinstance(derived["navigation"], dict):
                    if "max_steps" not in derived["navigation"]:
                        derived["navigation"]["max_steps"] = nav_loop.get(
                            "max_steps", 200
                        )
                    # 兼容旧字段名 goal_distance
                    if "goal_distance" not in derived["navigation"]:
                        derived["navigation"]["goal_distance"] = nav_loop.get(
                            "goal_distance_threshold",
                            nav_loop.get("goal_distance", 0.5),
                        )

        return derived

    def load_config(
        self,
        config_name: str,
        config_format: Optional[ConfigFormat] = None,
        defaults: Optional[Dict[str, Any]] = None,
        required_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        加载配置文件

        参数：
            config_name: 配置文件名（不含扩展名）
            config_format: 配置格式（自动检测）
            defaults: 默认值字典
            required_keys: 必需的键列表

        返回：
            配置字典

        异常：
            ConfigLoadException: 加载失败
            ConfigValidationException: 验证失败
        """
        # 从缓存中获取
        if self.enable_cache and config_name in self._cache:
            logger.debug(f"从缓存获取配置: {config_name}")
            return copy.deepcopy(self._cache[config_name])

        # 尝试加载配置文件
        config = None
        file_path = None

        # 1. 尝试 YAML 格式
        yaml_path = self.config_dir / f"{config_name}.yaml"
        if yaml_path.exists():
            try:
                config = self._load_yaml(yaml_path)
                file_path = yaml_path
                logger.info(f"加载 YAML 配置: {yaml_path}")
            except Exception as e:
                logger.error(f"加载 YAML 配置失败: {e}")

        # 2. 尝试 JSON 格式
        if config is None:
            json_path = self.config_dir / f"{config_name}.json"
            if json_path.exists():
                try:
                    config = self._load_json(json_path)
                    file_path = json_path
                    logger.info(f"加载 JSON 配置: {json_path}")
                except Exception as e:
                    logger.error(f"加载 JSON 配置失败: {e}")

        # 3. 如果没有找到文件，使用默认值
        if config is None:
            if defaults is not None:
                config = copy.deepcopy(defaults)
                logger.warning(f"配置文件未找到，使用默认值: {config_name}")
            else:
                # 3.1 向后兼容：如果缺少独立 *_config.yaml，则从 unified_config.yaml 派生
                unified = None
                if config_name != "unified_config":
                    unified = self._load_unified_raw()
                if unified is not None:
                    derived = self._derive_legacy_config_from_unified(
                        config_name, unified
                    )
                    if derived is not None:
                        config = derived
                        file_path = self.config_dir / "unified_config.yaml"
                        logger.info(
                            f"未找到 {config_name}.yaml/json，已从 unified_config.yaml 派生配置"
                        )

                if config is None:
                    raise ConfigLoadException(
                        f"找不到配置文件: {config_name}, 搜索位置: {yaml_path}, {json_path}"
                    )

        # 4. 应用环境变量覆盖
        config = self._apply_env_overrides(config, config_name)

        # 5. 验证配置
        if required_keys:
            self._validate_required_keys(config, required_keys, config_name)

        # 6. 缓存配置
        if self.enable_cache:
            self._cache[config_name] = copy.deepcopy(config)
            logger.debug(f"配置已缓存: {config_name}")

        # 7. 记录元数据
        metadata = ConfigMetadata(
            file_path=str(file_path) if file_path else None,
            config_hash=self._compute_hash(config),
            validated=True,
        )
        self._loaded_files[config_name] = metadata

        return config

    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """加载 YAML 文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                if config is None:
                    config = {}
                return config
        except yaml.YAMLError as e:
            raise ConfigLoadException(f"YAML 解析错误 ({file_path}): {e}")
        except Exception as e:
            raise ConfigLoadException(f"加载 YAML 失败 ({file_path}): {e}")

    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """加载 JSON 文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigLoadException(f"JSON 解析错误 ({file_path}): {e}")
        except Exception as e:
            raise ConfigLoadException(f"加载 JSON 失败 ({file_path}): {e}")

    def _apply_env_overrides(
        self, config: Dict[str, Any], config_name: str
    ) -> Dict[str, Any]:
        """
        应用环境变量覆盖

        环境变量命名约定: CONFIG_NAME_KEY_SUBKEY=value
        例如: CONFIG_VLM_API_TIMEOUT=120
        """
        prefix = f"CONFIG_{config_name.upper()}_"

        for env_var, value in os.environ.items():
            if env_var.startswith(prefix):
                # 提取键路径
                key_path = env_var[len(prefix) :].lower().split("_")

                # 设置值
                current = config
                for key in key_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]

                current[key_path[-1]] = self._parse_env_value(value)
                logger.info(f"环境变量覆盖: {env_var}={value}")

        return config

    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """解析环境变量值"""
        # 尝试解析为布尔值
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False

        # 尝试解析为数字
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # 返回字符串
        return value

    def _validate_required_keys(
        self, config: Dict[str, Any], required_keys: List[str], config_name: str
    ) -> None:
        """验证必需的键"""
        missing_keys = []

        for key in required_keys:
            # 支持嵌套键: "section.subsection.key"
            keys = key.split(".")
            current = config

            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    missing_keys.append(key)
                    break

        if missing_keys:
            error_msg = f"配置 {config_name} 缺少必需的键: {missing_keys}"
            if self.strict_validation:
                raise ConfigValidationException(error_msg)
            else:
                logger.warning(error_msg)

    @staticmethod
    def _compute_hash(config: Dict[str, Any]) -> str:
        """计算配置的哈希值"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def load_system_config(self) -> Dict[str, Any]:
        """加载系统配置"""
        return self.load_config(
            "system_config",
            required_keys=["logging.level", "output.base_dir"],
        )

    def load_agent_config(self) -> Dict[str, Any]:
        """加载 Agent 配置"""
        return self.load_config(
            "agent_config", required_keys=["agent1.enabled", "agent2.enabled"]
        )

    def load_environment_config(self) -> Dict[str, Any]:
        """加载环境配置"""
        return self.load_config(
            "environment_config",
            required_keys=[
                "scene.path",
                "sensors.front_camera.enabled",
                "actions.move_forward.amount",
            ],
        )

    def load_vlm_config(self) -> Dict[str, Any]:
        """加载 VLM 配置"""
        return self.load_config(
            "vlm_config", required_keys=["model.name", "inference.max_tokens"]
        )

    def load_paths_config(self) -> Dict[str, Any]:
        """加载路径配置"""
        config = self.load_config(
            "paths_config", required_keys=["project.root", "data.scene_data_dir"]
        )

        # 对路径配置做额外验证：检查引用的文件/目录是否存在，并根据策略创建缺失目录
        try:
            self.validate_paths(config, create_missing=True, strategy="create")
        except Exception as e:
            logger.warning(f"路径验证或创建时出现问题: {e}")
            if self.strict_validation:
                raise

        return config

    def load_prompts_config(self) -> Dict[str, Any]:
        """加载提示配置"""
        return self.load_config("prompts_config")

    def load_unified_config(self) -> Dict[str, Any]:
        """
        加载统一配置文件

        返回：
            包含所有配置部分的字典：
            - environment
            - agents
            - vlm
            - system
            - paths
            - navigation
            - prompts
        """
        config = self.load_config(
            "unified_config",
            required_keys=[
                "environment.scene.path",
                "environment.agent.height",
                "vlm.model.name",
                "system.logging.level",
            ],
        )

        logger.info("✅ 统一配置文件加载成功")
        return config

    def validate_paths(
        self,
        paths_config: Dict[str, Any],
        create_missing: bool = True,
        strategy: str = "create",
    ) -> Dict[str, bool]:
        """
        验证并（可选）创建 `paths_config` 中声明的路径。

        参数:
            paths_config: 已加载的 `paths_config` 字典（可嵌套）
            create_missing: 当路径缺失时是否尝试创建（仅对目录生效）
            strategy: 行为策略：'create'|'warn'|'fail'

        返回:
            dict: 将路径键（被扁平化为点分路径）映射到布尔值（存在/可用）

        行为说明:
            - 遍历嵌套字典，遇到字符串类型的叶子节点则视为路径并检查其是否存在。
            - 如果路径缺失并且 `create_missing` 为 True，且该路径看起来像目录（没有文件扩展名或以 '/' 结尾或键名含 'dir'/'root'），
              尝试创建该目录。
            - 如果 `strategy=='fail'` 且发现缺失路径，则会抛出 `ConfigValidationException`（除非已创建）。
        """

        results: Dict[str, bool] = {}

        def recurse(prefix: str, obj: Any):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_prefix = f"{prefix}.{k}" if prefix else k
                    recurse(new_prefix, v)
            else:
                # 仅处理字符串叶子节点（可能是文件或目录路径）
                if isinstance(obj, str):
                    key = prefix
                    path_str = obj
                    try:
                        p = Path(path_str)
                    except Exception:
                        results[key] = False
                        return

                    if p.exists():
                        results[key] = True
                        return

                    # 判断是否应该尝试创建（目录判断）
                    looks_like_dir = (
                        path_str.endswith(os.sep)
                        or key.lower().endswith("dir")
                        or key.lower().endswith("root")
                        or p.suffix == ""
                    )

                    if create_missing and looks_like_dir:
                        try:
                            p.mkdir(parents=True, exist_ok=True)
                            results[key] = True
                            logger.info(f"已创建缺失目录: {p}")
                        except Exception as e:
                            results[key] = False
                            logger.error(f"创建目录失败: {p} -> {e}")
                            if strategy == "fail" or self.strict_validation:
                                raise ConfigValidationException(
                                    f"创建目录失败: {p} -> {e}"
                                )
                    else:
                        # 不创建，记录为缺失
                        results[key] = False
                        msg = f"路径缺失: {key} -> {path_str}"
                        if strategy == "fail" or self.strict_validation:
                            logger.error(msg)
                            raise ConfigValidationException(msg)
                        else:
                            logger.warning(msg)

        recurse("", paths_config)

        return results

    def load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """加载所有配置"""
        configs = {}

        try:
            configs["system"] = self.load_system_config()
        except Exception as e:
            logger.warning(f"加载系统配置失败: {e}")

        try:
            configs["agent"] = self.load_agent_config()
        except Exception as e:
            logger.warning(f"加载 Agent 配置失败: {e}")

        try:
            configs["environment"] = self.load_environment_config()
        except Exception as e:
            logger.warning(f"加载环境配置失败: {e}")

        try:
            configs["vlm"] = self.load_vlm_config()
        except Exception as e:
            logger.warning(f"加载 VLM 配置失败: {e}")

        try:
            configs["paths"] = self.load_paths_config()
        except Exception as e:
            logger.warning(f"加载路径配置失败: {e}")

        try:
            configs["prompts"] = self.load_prompts_config()
        except Exception as e:
            logger.warning(f"加载提示配置失败: {e}")

        return configs

    def get_config_value(
        self, config_name: str, key_path: str, default: Any = None
    ) -> Any:
        """
        获取嵌套配置值

        参数：
            config_name: 配置文件名
            key_path: 键路径 (支持 "section.subsection.key")
            default: 默认值

        返回：
            配置值或默认值
        """
        try:
            config = self.load_config(config_name)
            keys = key_path.split(".")
            current = config

            for key in keys:
                current = current[key]

            return current
        except (KeyError, TypeError):
            return default

    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        return {
            "cached_configs": list(self._cache.keys()),
            "cache_size": len(self._cache),
            "loaded_files": {
                name: asdict(metadata) for name, metadata in self._loaded_files.items()
            },
        }

    def clear_cache(self) -> None:
        """清除缓存"""
        self._cache.clear()
        self._hash_cache.clear()
        logger.info("配置缓存已清除")

    def validate_all_configs(self) -> Dict[str, bool]:
        """验证所有配置"""
        results = {}

        config_files = [
            "system_config",
            "agent_config",
            "environment_config",
            "vlm_config",
            "paths_config",
        ]

        for config_name in config_files:
            try:
                self.load_config(config_name)
                results[config_name] = True
                logger.info(f"✓ 配置验证通过: {config_name}")
            except Exception as e:
                results[config_name] = False
                logger.error(f"✗ 配置验证失败: {config_name} - {e}")

        return results

    def export_config(
        self,
        config_name: str,
        export_path: str,
        format: ConfigFormat = ConfigFormat.YAML,
    ) -> None:
        """
        导出配置文件

        参数：
            config_name: 配置名称
            export_path: 导出路径
            format: 导出格式
        """
        config = self.load_config(config_name)

        try:
            if format == ConfigFormat.YAML:
                with open(export_path, "w", encoding="utf-8") as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            elif format == ConfigFormat.JSON:
                with open(export_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)

            logger.info(f"配置已导出: {export_path}")
        except Exception as e:
            logger.error(f"配置导出失败: {e}")
            raise


# ============================================================================
# 全局配置加载器单例
# ============================================================================

_global_loader: Optional[UnifiedConfigLoader] = None


def get_config_loader(
    config_dir: str = "./configs", enable_cache: bool = True
) -> UnifiedConfigLoader:
    """
    获取全局配置加载器

    参数：
        config_dir: 配置目录
        enable_cache: 是否启用缓存

    返回：
        配置加载器实例
    """
    global _global_loader

    if _global_loader is None:
        _global_loader = UnifiedConfigLoader(
            config_dir=config_dir, enable_cache=enable_cache
        )

    return _global_loader


# ============================================================================
# 便利函数
# ============================================================================


def load_config(config_name: str) -> Dict[str, Any]:
    """加载配置"""
    return get_config_loader().load_config(config_name)


def get_config_value(config_name: str, key_path: str, default: Any = None) -> Any:
    """获取配置值"""
    return get_config_loader().get_config_value(config_name, key_path, default)


def validate_configs() -> Dict[str, bool]:
    """验证所有配置"""
    return get_config_loader().validate_all_configs()


if __name__ == "__main__":
    # 测试配置加载器
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    loader = get_config_loader("./configs")

    # 验证所有配置
    logger = logging.getLogger("ConfigLoaderMain")
    logger.info("验证配置文件...")
    results = loader.validate_all_configs()

    logger.info("配置验证结果:")
    for config_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info("  %s: %s", status, config_name)

    # 加载所有配置
    logger.info("加载所有配置...")
    all_configs = loader.load_all_configs()

    logger.info("已加载配置: %s", list(all_configs.keys()))
    logger.info("缓存信息: %s", loader.get_cache_info())
