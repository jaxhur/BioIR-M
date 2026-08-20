"""安全解压并校验 BioIR-M 所需的 LOL-v1 与 LOL-v2 数据集。

该脚本只依赖 Python 标准库。它将 README 指定的两个 ZIP 归档解压到项目当前
配置需要的 ``datasets/LOL-v1`` 与 ``datasets/LOL-v2``，并按相对路径验证每个
LQ/GT 目录中的图像能否一一配对。
"""

from __future__ import annotations

import argparse
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetLayout:
    """描述一套 LOL 数据的目标目录与全部 LQ/GT 配对目录。

    Args:
        name: 数据集名称，用于输出提示。
        target_directory: 数据根目录下解压后的标准目录名。
        pairs: 相对于标准目录的 ``(LQ, GT)`` 目录元组。
    """

    name: str
    target_directory: str
    pairs: tuple[tuple[Path, Path], ...]


LOL_LAYOUTS = (
    DatasetLayout(
        name="LOL-v1",
        target_directory="LOL-v1",
        pairs=(
            (Path("our485/low"), Path("our485/high")),
            (Path("eval15/low"), Path("eval15/high")),
        ),
    ),
    DatasetLayout(
        name="LOL-v2",
        target_directory="LOL-v2",
        pairs=(
            (Path("Synthetic/Train/Low"), Path("Synthetic/Train/Normal")),
            (Path("Synthetic/Test/Low"), Path("Synthetic/Test/Normal")),
            (Path("Real_captured/Train/Low"), Path("Real_captured/Train/Normal")),
            (Path("Real_captured/Test/Low"), Path("Real_captured/Test/Normal")),
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    """解析数据集解压与校验参数。

    Returns:
        包含数据根目录、ZIP 归档和安全替换选项的参数对象。
    """

    parser = argparse.ArgumentParser(
        description="安全解压并严格校验 BioIR-M 所需的 LOL-v1/LOL-v2 数据集。")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets"),
        help="标准 LOL 数据根目录，默认项目根目录下的 datasets。",
    )
    parser.add_argument(
        "--lol-v1-archive",
        type=Path,
        help="README 中 LOL-v1 Google Drive 归档的本地 ZIP 路径。",
    )
    parser.add_argument(
        "--lol-v2-archive",
        type=Path,
        help="README 中 LOL-v2 Google Drive 归档的本地 ZIP 路径。",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="只校验现有目录和 LQ/GT 相对路径配对，不解压任何归档。",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="仅在标准数据目录已存在但校验失败时删除并替换；默认绝不删除。",
    )
    return parser.parse_args()


def discover_images(directory: Path) -> dict[str, Path]:
    """递归建立大小写无关的图像相对路径索引。

    Args:
        directory: 当前 LQ 或 GT 图像根目录。

    Returns:
        键为规范化相对路径，值为真实图像路径的字典。

    Raises:
        FileNotFoundError: 目录不存在时抛出。
        ValueError: 目录中没有图像或出现重复键时抛出。
    """

    if not directory.is_dir():
        raise FileNotFoundError(f"图像目录不存在：{directory}")
    index: dict[str, Path] = {}
    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        relative_path = path.relative_to(directory)
        key = relative_path.as_posix().casefold()
        if key in index:
            raise ValueError(f"发现重复相对路径：{index[key]} 与 {path}")
        index[key] = path
    if not index:
        raise ValueError(f"目录中没有支持的图像：{directory}")
    return index


def validate_pair(lq_directory: Path, gt_directory: Path) -> int:
    """严格验证一组 LQ/GT 目录按相对路径完全配对。

    Args:
        lq_directory: 低照度图像根目录。
        gt_directory: 正常曝光 GT 图像根目录。

    Returns:
        成功配对的图像数量。

    Raises:
        ValueError: 缺失 LQ、GT 或数量不匹配时抛出。
    """

    lq_index = discover_images(lq_directory)
    gt_index = discover_images(gt_directory)
    lq_keys = set(lq_index)
    gt_keys = set(gt_index)
    missing_gt = sorted(lq_keys - gt_keys)
    missing_lq = sorted(gt_keys - lq_keys)
    if missing_gt or missing_lq:
        details: list[str] = []
        if missing_gt:
            details.append(f"缺少 GT：{missing_gt[:5]}")
        if missing_lq:
            details.append(f"缺少 LQ：{missing_lq[:5]}")
        raise ValueError(
            f"配对失败：{lq_directory} <-> {gt_directory}；" + "；".join(details))
    return len(lq_index)


def validate_layout(data_root: Path, layout: DatasetLayout) -> list[tuple[Path, int]]:
    """验证一套数据集的所有训练/测试 LQ/GT 目录。

    Args:
        data_root: 数据集总根目录。
        layout: 当前 LOL 数据集目录定义。

    Returns:
        每个 LQ 相对目录及其配对图像数量。
    """

    dataset_root = data_root / layout.target_directory
    results: list[tuple[Path, int]] = []
    for lq_relative, gt_relative in layout.pairs:
        count = validate_pair(dataset_root / lq_relative, dataset_root / gt_relative)
        results.append((lq_relative, count))
    return results


def validate_all_layouts(data_root: Path) -> None:
    """验证 LOL-v1 和 LOL-v2 的全部目录，并打印每个 split 的图像数量。

    Args:
        data_root: 数据集总根目录。
    """

    for layout in LOL_LAYOUTS:
        print(f"[校验] {layout.name}")
        for lq_relative, count in validate_layout(data_root, layout):
            print(f"  {lq_relative.as_posix()}: {count} 对")


def _is_zip_member_safe(destination: Path, member: zipfile.ZipInfo) -> bool:
    """判断 ZIP 成员路径与类型是否安全，防止路径穿越和符号链接。

    Args:
        destination: 当前解压目录。
        member: 待检查的 ZIP 成员。

    Returns:
        成员可安全解压时返回 ``True``。
    """

    member_path = (destination / member.filename).resolve()
    try:
        member_path.relative_to(destination.resolve())
    except ValueError:
        return False
    mode = member.external_attr >> 16
    return not stat.S_ISLNK(mode)


def extract_zip_safely(archive_path: Path, destination: Path) -> None:
    """将 ZIP 归档安全解压至空 staging 目录。

    Args:
        archive_path: 输入 ZIP 文件。
        destination: staging 解压目录。

    Raises:
        ValueError: ZIP 中包含路径穿越或符号链接时抛出。
    """

    with zipfile.ZipFile(archive_path) as archive:
        unsafe_members = [
            member.filename for member in archive.infolist()
            if not _is_zip_member_safe(destination, member)
        ]
        if unsafe_members:
            raise ValueError(f"ZIP 包含不安全路径：{unsafe_members[:3]}")
        archive.extractall(destination)


def find_dataset_root(staging_root: Path, layout: DatasetLayout) -> Path:
    """在解压 staging 目录中定位满足项目标准结构的实际数据根目录。

    README 的归档可能包含 ``LOL-v1-Fr``、``LOL-v2-Fr`` 或额外的顶层目录；该函数
    不依赖归档名称，只根据项目实际需要的所有 LQ/GT 子目录定位根目录。

    Args:
        staging_root: 当前归档的临时解压目录。
        layout: 需要满足的 LOL 目录定义。

    Returns:
        包含全部所需子目录的数据集根目录。

    Raises:
        FileNotFoundError: 无法在归档中找到符合项目要求的目录时抛出。
    """

    candidates: Iterable[Path] = [staging_root, *staging_root.rglob("*")]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if all((candidate / relative).is_dir()
               for pair in layout.pairs for relative in pair):
            return candidate
    raise FileNotFoundError(
        f"归档中未找到 {layout.name} 所需的完整目录结构。")


def remove_existing_target(target: Path) -> None:
    """删除已确认无效且用户明确允许替换的数据目录。

    Args:
        target: 要删除的项目标准数据目录。
    """

    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()


def install_archive(data_root: Path, archive_path: Path, layout: DatasetLayout,
                    replace_existing: bool) -> None:
    """解压一个 LOL ZIP，并规范化为项目配置使用的标准目录名。

    Args:
        data_root: 数据集总根目录。
        archive_path: 待解压的 ZIP 文件。
        layout: 当前数据集目录定义。
        replace_existing: 是否允许删除校验失败的现有目标目录。

    Raises:
        FileExistsError: 现有目标目录无效且未显式允许替换时抛出。
    """

    data_root.mkdir(parents=True, exist_ok=True)
    target = data_root / layout.target_directory
    if target.exists():
        try:
            validate_layout(data_root, layout)
            print(f"[跳过] {layout.name} 已存在且校验通过：{target}")
            return
        except (FileNotFoundError, ValueError) as error:
            if not replace_existing:
                raise FileExistsError(
                    f"{target} 已存在但校验失败：{error}\n"
                    "请先人工处理该目录，或确认后使用 --replace-existing。") from error
            print(f"[替换] 删除校验失败的目录：{target}")
            remove_existing_target(target)

    with tempfile.TemporaryDirectory(prefix=f".{layout.target_directory}.extract.",
                                     dir=data_root) as staging_name:
        staging_root = Path(staging_name)
        print(f"[解压] {archive_path.name} -> 临时目录")
        extract_zip_safely(archive_path, staging_root)
        source_root = find_dataset_root(staging_root, layout)
        shutil.move(str(source_root), str(target))
    validate_layout(data_root, layout)
    print(f"[完成] {layout.name} 已准备至：{target}")


def main() -> None:
    """执行归档解压或现有 LOL 数据目录的严格校验。"""

    args = parse_args()
    data_root = args.data_root.resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    if args.verify_only:
        validate_all_layouts(data_root)
        return
    if args.lol_v1_archive is None or args.lol_v2_archive is None:
        raise ValueError("解压数据集时必须同时提供 --lol-v1-archive 和 --lol-v2-archive。")

    archives = (args.lol_v1_archive.resolve(), args.lol_v2_archive.resolve())
    for archive in archives:
        if not archive.is_file():
            raise FileNotFoundError(f"数据集归档不存在：{archive}")
    install_archive(data_root, archives[0], LOL_LAYOUTS[0], args.replace_existing)
    install_archive(data_root, archives[1], LOL_LAYOUTS[1], args.replace_existing)
    validate_all_layouts(data_root)


if __name__ == "__main__":
    main()
