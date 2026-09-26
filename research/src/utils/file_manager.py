#!/usr/bin/env python3
"""
文件管理工具模块
自动管理文件保存位置和命名规范
"""
from pathlib import Path
from datetime import datetime


class ProjectPaths:
    """项目路径管理"""

    def __init__(self, base_dir=None):
        if base_dir is None:
            # 默认：从当前文件位置向上3层到项目根目录
            self.base_dir = Path(__file__).parents[2]
        else:
            self.base_dir = Path(base_dir)

    @property
    def src(self):
        return self.base_dir / "src"

    @property
    def data_raw(self):
        return self.base_dir / "data" / "raw"

    @property
    def data_processed(self):
        return self.base_dir / "data" / "processed"

    @property
    def models(self):
        return self.base_dir / "models"

    @property
    def experiments(self):
        return self.base_dir / "experiments"

    @property
    def analysis(self):
        return self.base_dir / "analysis"

    @property
    def threshold_evaluation(self):
        return self.base_dir / "analysis" / "threshold_evaluation"

    @property
    def docs(self):
        return self.base_dir / "docs"


def get_filename(description, extension, version=None):
    """
    生成带日期的文件名

    Args:
        description: 文件描述（如 "clip_evaluation"）
        extension: 文件扩展名（如 "xlsx", "png", "csv"）
        version: 可选版本号（如 "2", "v2"）

    Returns:
        文件名（如 "clip_evaluation_20260706.xlsx"）
    """
    date_str = datetime.now().strftime("%Y%m%d")

    if version:
        version_str = str(version).lower()
        if not version_str.startswith("v"):
            version_str = f"v{version_str}"
        return f"{description}_{date_str}_{version_str}.{extension}"

    return f"{description}_{date_str}.{extension}"


def save_dataframe(df, description, directory, extension="xlsx", version=None):
    """
    保存 DataFrame 到指定目录

    Args:
        df: pandas DataFrame
        description: 文件描述
        directory: 目标目录 (Path对象)
        extension: 文件格式 ("xlsx" 或 "csv")
        version: 可选版本号

    Returns:
        保存的文件路径
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    filename = get_filename(description, extension, version)
    filepath = directory / filename

    if extension == "xlsx":
        df.to_excel(filepath, index=False)
    elif extension == "csv":
        df.to_csv(filepath, index=False)
    else:
        raise ValueError(f"不支持的格式: {extension}")

    print(f"✅ 保存到: {filepath}")
    return filepath


def save_figure(fig, description, directory, extension="png", version=None, dpi=300, **kwargs):
    """
    保存 matplotlib 图表

    Args:
        fig: matplotlib Figure 对象
        description: 文件描述
        directory: 目标目录 (Path对象)
        extension: 文件格式 ("png", "jpg", "pdf")
        version: 可选版本号
        dpi: 图片质量
        **kwargs: 传递给 fig.savefig 的其他参数

    Returns:
        保存的文件路径
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    filename = get_filename(description, extension, version)
    filepath = directory / filename

    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", **kwargs)
    print(f"✅ 保存到: {filepath}")
    return filepath


def save_json(data, description, directory, version=None):
    """
    保存 JSON 数据

    Args:
        data: 可序列化的 Python 对象
        description: 文件描述
        directory: 目标目录 (Path对象)
        version: 可选版本号

    Returns:
        保存的文件路径
    """
    import json

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    filename = get_filename(description, "json", version)
    filepath = directory / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ 保存到: {filepath}")
    return filepath
