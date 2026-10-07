"""Package source, PDF, and evaluation records without large model weights."""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent


def main():
    source_names = ["README.md", "requirements.txt", ".gitignore", "experiment.py", "download_resources.py",
                    "build_report.py", "predict.py", "verify_outputs.py", "make_submission.py",
                    "run_all.ps1", "实验报告.pdf", "tests/test_experiment.py"]
    files = [ROOT / name for name in source_names]
    files.extend(p for p in sorted((ROOT / "outputs").rglob("*"))
                 if p.is_file() and "models" not in p.relative_to(ROOT / "outputs").parts
                 and p.suffix in {".json", ".csv", ".log"})
    if any(not path.exists() for path in files):
        raise FileNotFoundError("A required source or report is missing.")
    verified = json.loads((ROOT / "outputs/verification.json").read_text(encoding="utf-8"))
    if len(verified["models"]) != 7 or not all(v["passed"] for v in verified["models"].values()):
        raise ValueError("All seven persisted models must pass verification before packaging.")
    manifest = {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    target = ROOT / "作业一_提交包.zip"
    with ZipFile(target, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, arcname=path.relative_to(ROOT))
        archive.writestr("SHA256SUMS.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise IOError("ZIP integrity verification failed.")
    print(f"Saved {target.name}: {len(files)+1} files, {target.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
