import kagglehub
import shutil
from pathlib import Path

cache_path = kagglehub.dataset_download(
    "umerhaddii/ai-governance-documents-data"
)

destination = Path("./data/documents")

shutil.copytree(
    cache_path,
    destination,
    dirs_exist_ok=True
)

print(f"Copied to: {destination.resolve()}")