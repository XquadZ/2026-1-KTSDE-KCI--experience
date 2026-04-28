from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis import correlation_report, ranking_table
from score import add_score_columns


def main() -> None:
    parser = argparse.ArgumentParser(description="논문 앙상블 점수 S 파이프라인")
    parser.add_argument("--input", required=True, help="입력 CSV 경로")
    parser.add_argument("--outdir", required=True, help="결과 저장 폴더")
    args = parser.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    required = ["Combo", "Gain_miss", "Dis", "UFP", "Comp", "TFC"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    scored = add_score_columns(df)
    ranked = ranking_table(scored, score_col="S", combo_col="Combo")
    corr = correlation_report(scored, score_col="S", gain_ap_col="GainAP", gain_ar_col="GainAR")

    scored.to_csv(outdir / "scored_table.csv", index=False, encoding="utf-8-sig")
    ranked.to_csv(outdir / "ranked_by_s.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(outdir / "correlations.csv", index=False, encoding="utf-8-sig")

    print(f"[완료] scored_table.csv -> {outdir}")
    print(f"[완료] ranked_by_s.csv -> {outdir}")
    print(f"[완료] correlations.csv -> {outdir}")


if __name__ == "__main__":
    main()
