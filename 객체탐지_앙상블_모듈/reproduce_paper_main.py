from __future__ import annotations

"""
논문 실험 재현용 단일 실행 스크립트.

역할:
1) 데이터/예측 파일 존재 여부 검증
2) 레거시 실험 파이프라인(main.py)을 그대로 호출
3) 결과 파일 위치 안내
"""

import importlib.util
import sys
from pathlib import Path


def _load_legacy_main(repo_root: Path):
    main_path = repo_root / "main.py"
    if not main_path.exists():
        raise FileNotFoundError(f"main.py not found: {main_path}")

    spec = importlib.util.spec_from_file_location("legacy_main_module", str(main_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load main.py module spec.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        raise AttributeError("main.py does not expose `main()`.")
    return module.main


def _validate_required_files(repo_root: Path):
    required = [
        repo_root / "data" / "annotations" / "test_annotations.json",
        repo_root / "runs" / "single_preds" / "pred_DeformableDETR_hf.json",
        repo_root / "runs" / "single_preds" / "pred_FRCNN_tv.json",
        repo_root / "runs" / "single_preds" / "pred_FCOS_tv.json",
        repo_root / "runs" / "single_preds" / "pred_YOLOv8_test.json",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        msg = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(
            "재현에 필요한 입력 파일이 없습니다. 아래 파일을 채워주세요:\n" + msg
        )


def main():
    module_dir = Path(__file__).resolve().parent
    repo_root = module_dir.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    print(f"[INFO] Repo root: {repo_root}")
    _validate_required_files(repo_root)

    legacy_main = _load_legacy_main(repo_root)
    print("[INFO] Running paper reproduction pipeline via main.py ...")
    legacy_main()

    out_dir = repo_root / "outputs" / "stats"
    print("[DONE] 실험 재현 실행이 완료되었습니다.")
    print(f"[DONE] 결과 확인 경로: {out_dir}")
    print("       - table4_preensemble_indicators.csv")
    print("       - table5_actual_ensemble_gains.csv")
    print("       - table6_correlations.csv")
    print("       - table7_selection_performance.csv")


if __name__ == "__main__":
    main()
