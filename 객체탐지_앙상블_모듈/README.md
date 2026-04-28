# 객체 탐지 앙상블 실험 모듈

이 폴더는 논문 **"오류 상관 분석을 이용한 효율적 객체 탐지 앙상블 선택 전략"**의 실험 코드를
재사용 가능한 모듈 형태로 정리한 버전입니다.

## 구성

- `metrics.py`
  - 논문 핵심 지표 계산 함수
  - `disagreement`, `joint_miss`, `gain_miss`, `ufp`, `tfc`, `comp`
- `score.py`
  - Min-Max 정규화 + 제안 점수 `S` 계산
  - `S = n(Gain_miss) + n(Dis) + n(Comp) - n(UFP) - n(TFC)`
- `analysis.py`
  - Spearman/Pearson 상관 분석
  - 조합 랭킹 생성
- `run_pipeline.py`
  - CSV 입력을 받아 `S` 산출 및 랭킹/상관 분석까지 한 번에 실행하는 엔트리 포인트

## 입력 데이터 형식

`run_pipeline.py`는 아래 컬럼이 포함된 CSV를 기대합니다.

- `Combo` (조합명)
- `Gain_miss`
- `Dis`
- `UFP`
- `Comp`
- `TFC`
- (선택) `GainAP`, `GainAR` (상관 분석용)

## 실행 예시

```bash
python run_pipeline.py --input "table4_preensemble_indicators.csv" --outdir "./outputs"
```

## 원본 근거

다음 실험 노트북/산출물의 공통 로직을 모듈화했습니다.

- `오류 상관 분석을 이용한 효율적 객체 탐지 앙상블 선택 전략  .ipynb`
- `논문수정작업-Valid.ipynb`
- `논문수정작업-Test.ipynb`
- `preensemble_outputs/table4_preensemble_indicators.csv`
- `preensemble_outputs/table5_actual_ensemble_gains.csv`
- `preensemble_outputs/table6_correlations.csv`
