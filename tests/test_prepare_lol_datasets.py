"""验证 LOL 数据集安全解压、目录归一化与严格配对。"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIRECTORY = PROJECT_ROOT / "scripts"
TEST_TEMP_ROOT = Path(
    os.environ.get("BIOIR_TEST_TMPDIR", PROJECT_ROOT / ".test-tmp"))
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from prepare_lol_datasets import (  # noqa: E402
    LOL_LAYOUTS,
    install_archive,
    validate_all_layouts,
)


class PrepareLOLDatasetsTest(unittest.TestCase):
    """覆盖归档根目录识别、标准目录写入与相对路径配对失败。"""

    @staticmethod
    def _temporary_directory(prefix: str) -> tempfile.TemporaryDirectory[str]:
        """在可配置、可清理的测试根目录创建临时目录。"""

        TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(prefix=prefix, dir=TEST_TEMP_ROOT)

    @staticmethod
    def _write_pair(root: Path, lq_relative: Path, gt_relative: Path,
                    nested_name: str = "scene/example.png") -> None:
        """在指定 LQ/GT 目录下写入同名占位图像文件。"""

        for relative in (lq_relative / nested_name, gt_relative / nested_name):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"placeholder")

    def _create_archive(self, archive_path: Path, archive_root_name: str,
                        layout_index: int) -> None:
        """构造带 README 风格顶层目录的最小可校验 ZIP 归档。"""

        layout = LOL_LAYOUTS[layout_index]
        with self._temporary_directory("test-lol-source-") as source_name:
            source_root = Path(source_name) / archive_root_name
            for lq_relative, gt_relative in layout.pairs:
                self._write_pair(source_root, lq_relative, gt_relative)
            with zipfile.ZipFile(archive_path, "w") as archive:
                for file_path in source_root.rglob("*"):
                    if file_path.is_file():
                        archive.write(file_path, file_path.relative_to(source_root.parent))

    def test_archives_are_normalized_to_project_directory_names(self):
        """LOL-v1-Fr/LOL-v2-Fr 归档应转为配置所需的 LOL-v1/LOL-v2。"""

        with self._temporary_directory("test-lol-workspace-") as workspace_name:
            workspace = Path(workspace_name)
            data_root = workspace / "datasets"
            v1_archive = workspace / "LOL-v1.zip"
            v2_archive = workspace / "LOL-v2-renamed.zip"
            self._create_archive(v1_archive, "LOL-v1-Fr", layout_index=0)
            self._create_archive(v2_archive, "LOL-v2-Fr", layout_index=1)

            install_archive(data_root, v1_archive, LOL_LAYOUTS[0], False)
            install_archive(data_root, v2_archive, LOL_LAYOUTS[1], False)
            self.assertTrue((data_root / "LOL-v1/our485/low").is_dir())
            self.assertTrue((data_root / "LOL-v2/Synthetic/Train/Low").is_dir())
            validate_all_layouts(data_root)

    def test_pair_mismatch_is_rejected(self):
        """GT 缺失同名相对路径时必须拒绝通过，不能静默继续。"""

        with self._temporary_directory("test-lol-workspace-") as workspace_name:
            data_root = Path(workspace_name) / "datasets"
            dataset_root = data_root / "LOL-v1"
            self._write_pair(dataset_root, Path("our485/low"), Path("our485/high"))
            self._write_pair(dataset_root, Path("eval15/low"), Path("eval15/high"))
            (dataset_root / "eval15/high/scene/example.png").unlink()
            with self.assertRaises(ValueError):
                validate_all_layouts(data_root)


if __name__ == "__main__":
    unittest.main()
