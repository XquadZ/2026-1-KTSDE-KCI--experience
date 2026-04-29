# 2026-1-KTSDE-KCI--experience

객체 탐지 앙상블 선택 논문  
**"오류 상관 분석을 이용한 효율적 객체 탐지 앙상블 선택 전략"**의 실험 코드를 정리한 저장소입니다.

이 레포는 두 가지 실행 경로를 제공합니다.

- `main.py` 기반의 기존 통합 실험
- `객체탐지_앙상블_모듈/` 기반의 모듈형 파이프라인

## 프로젝트 구조

- `main.py`
  - 단일 모델 예측 기반 pre-ensemble 지표 계산
  - 실제 앙상블 결과 검증(COCOeval) 포함
- `객체탐지_앙상블_모듈/`
  - `metrics.py`: 핵심 지표(`disagreement`, `joint_miss`, `gain_miss`, `ufp`, `tfc`, `comp`)
  - `score.py`: Min-Max 정규화 + 제안 점수 `S` 계산
  - `analysis.py`: Spearman/Pearson 상관 분석, 랭킹 테이블
  - `run_pipeline.py`: CSV 입력 -> 점수 계산/랭킹/상관분석 일괄 실행
  - `reproduce_paper_main.py`: 논문 실험 재현용 진입 스크립트
- `src/`
  - 추론/융합/전략 평가 보조 코드

## 제안 점수 식

논문 제안 점수는 아래와 같습니다.

`S = n(Gain_miss) + n(Dis) + n(Comp) - n(UFP) - n(TFC)`

- `n(.)`: Min-Max 정규화

## 설치

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

## 빠른 실행

### 1) 모듈형 파이프라인 실행

```bash
python "객체탐지_앙상블_모듈/run_pipeline.py" --input "path/to/table4_preensemble_indicators.csv" --outdir "./outputs"
```

입력 CSV 권장 컬럼:

- `Combo`
- `Gain_miss`
- `Dis`
- `UFP`
- `Comp`
- `TFC`
- (선택) `GainAP`, `GainAR`

### 2) 논문 재현 엔트리 실행

```bash
python "객체탐지_앙상블_모듈/reproduce_paper_main.py"
```

### 3) 기존 통합 실험 실행

```bash
python main.py
```

## 산출물

파이프라인 실행 시 일반적으로 다음 결과가 생성됩니다.

- 조합별 점수/순위 테이블
- 상관분석 결과(Spearman, Pearson)
- 실험 요약 CSV

## 참고

- 현재 모듈형 코드는 `객체탐지_앙상블_모듈/`에 있습니다.
- 한글 경로 사용이 불편하면 동일 내용을 영문 폴더(`object_detection_ensemble_module/`)로 복사해 운영하는 것을 권장합니다.
