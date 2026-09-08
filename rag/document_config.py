"""
文档配置模块 - 读取 data/Config/document_config.json
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# 配置文件路径（相对项目根目录）
CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "Config" / "document_config.json"


@dataclass
class DocumentConfig:
    """与 document_config.json 中单个元素对应的数据类"""
    id: int
    name: str
    is_simplified: int  # 0: 繁体, 1: 简体
    high_freq_entities: List[str] = field(default_factory=list) # 高频的实体

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentConfig":
        """从单个 JSON 字典生成 DocumentConfig 实例"""
        return cls(
            id=int(data["id"]),
            name=str(data["name"]),
            is_simplified=int(data["is_simplified"]),
            high_freq_entities=list(data.get("high_freq_entities", [])),
        )

    @classmethod
    def load_all(cls, path: Path | str = CONFIG_PATH) -> List["DocumentConfig"]:
        """读取 JSON 文件，生成对应的 DocumentConfig 实例列表"""
        with open(path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)
        return [cls.from_dict(item) for item in raw_list]

    @classmethod
    def get_by_name(cls, name: str, path: Path | str = CONFIG_PATH) -> "DocumentConfig | None":
        """根据书名 name 查找对应的 DocumentConfig，找不到返回 None"""
        target = name.strip()
        for cfg in cls.load_all(path):
            if cfg.name.strip() == target:
                return cfg
        return None
