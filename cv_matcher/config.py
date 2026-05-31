import os
from pathlib import Path

_root = Path(__file__).resolve().parent
_data_path = _root / "data"
_out_path = _root / "output"

_tgt_file = _data_path / "target_role.txt"
_lbl_file = _data_path / "ground_truth.csv"
_res_dir = _data_path / "resumes"

_exts = {".txt", ".pdf", ".docx"}
_seed = 21

os.makedirs(_out_path, exist_ok=True)