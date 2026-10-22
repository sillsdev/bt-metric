"""Downloads the public human-judgment datasets used to meta-evaluate the metric."""

import argparse
import logging
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExternalDataset:
    name: str
    url: str
    file_name: str
    description: str

    def download(self, data_dir: Path, force: bool) -> Path:
        # Each dataset gets its own directory, so files from different sources can never collide.
        target_dir = data_dir / self.name
        path = target_dir / self.file_name
        if path.is_file() and not force:
            LOGGER.info(f"{self.name}: already downloaded to {path}")
            return path
        target_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"{self.name}: downloading {self.url}")
        partial_path = path.with_name(path.name + ".partial")
        with urllib.request.urlopen(self.url) as response, partial_path.open("wb") as file:
            shutil.copyfileobj(response, file)
        partial_path.replace(path)
        if tarfile.is_tarfile(path):
            with tarfile.open(path) as archive:
                # The "data" filter rejects absolute paths, links out of the directory and special files.
                archive.extractall(target_dir, filter="data")
        return path


class DatasetCatalog:
    def __init__(self) -> None:
        self._datasets: List[ExternalDataset] = [
            ExternalDataset(
                name="wmt-mqm",
                url="https://huggingface.co/datasets/RicardoRei/wmt-mqm-human-evaluation/resolve/main/train.csv",
                file_name="wmt-mqm-human-evaluation.csv",
                description="WMT 2020-2022 MQM scores with source, MT output, reference and system name",
            ),
            ExternalDataset(
                name="wmt-da",
                url="https://huggingface.co/datasets/RicardoRei/wmt-da-human-evaluation/resolve/main/train.csv",
                file_name="wmt-da-human-evaluation.csv",
                description="WMT 2017-2022 direct assessment scores, including lower-resource language pairs",
            ),
            ExternalDataset(
                name="afrimte",
                url="https://github.com/masakhane-io/africomet/raw/main/wmt2024_challenge_set.tgz",
                file_name="wmt2024_challenge_set.tgz",
                description="AfriMTE-ade-devtest-v2: 13 African language pairs, one MT system each, with human scores",
            ),
        ]

    def names(self) -> List[str]:
        return [dataset.name for dataset in self._datasets]

    def download(self, names: List[str], data_dir: Path, force: bool) -> None:
        for dataset in self._datasets:
            if dataset.name in names:
                dataset.download(data_dir, force)


def main() -> None:
    catalog = DatasetCatalog()
    parser = argparse.ArgumentParser(description="Download public human-judgment datasets for meta-evaluation")
    parser.add_argument("names", nargs="*", choices=catalog.names(), help="Datasets to download (default: all)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/external"))
    parser.add_argument("--force", action="store_true", help="Download again even if the file exists")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    catalog.download(args.names or catalog.names(), args.data_dir, args.force)


if __name__ == "__main__":
    main()
