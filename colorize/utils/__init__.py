from pathlib import Path

project_root = Path(__file__).parent.parent.parent
input_dir = project_root / "data" / "raw" / "unsplash"
output_dir = project_root / "data" / "processed" / "unsplash"

__all__ = [
    'project_root', 'input_dir', 'output_dir'
]
