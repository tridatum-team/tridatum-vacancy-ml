# Vacant House Prediction Pipeline

집계구(OA) 단위 빈집 발생 확률을 예측하는 ML 파이프라인

## Overview

국토교통부 빈집 실태조사 데이터와 다양한 외부 데이터를 통합하여 집계구 단위의 빈집 발생 확률을 예측합니다.

### Pipeline Flow

```
Step 1: 데이터 통합
    └─ Oracle DB 메인 데이터 + 전기 데이터 + 외부 집계구 데이터
    └─ PU Learning으로 빈집 라벨 보정

Step 2: 파생변수 생성
    └─ 집계구 단위 집계 (빈집비율, 건물수, 노후도 등)
    └─ 공간 피처 (시설 거리, 인프라 접근성)

Step 3: 클러스터링
    └─ K-Means 클러스터링 (지역 특성별 그룹화)
    └─ 클러스터 프로파일 생성

Step 4: 모델 학습
    └─ RandomForest, XGBoost, LightGBM
    └─ 클러스터별 개별 모델 학습
    └─ Feature Importance 분석

Step 5: 앙상블
    └─ Soft Voting 앙상블
    └─ 클러스터별 최적 가중치 적용

Step 6: 예측
    └─ 5개년 빈집 발생 확률 예측
    └─ 결과 DB 저장
```

## Project Structure

```
├── binzib_dag.py          # Airflow DAG 정의
├── main.py                # CLI 실행 스크립트
├── configs/
│   └── config.yaml        # 설정 파일 (DB, 경로 등)
└── src/
    ├── step1_data_integration.py   # 데이터 통합 + PU Learning
    ├── step2_feature_engineering.py # 파생변수 생성
    ├── step3_clustering.py         # 클러스터링
    ├── step4_modeling.py           # 모델 학습
    ├── step5_ensemble.py           # 앙상블
    ├── step6_forecast.py           # 5개년 예측
    ├── utils.py                    # 유틸리티 함수
    ├── config_loader.py            # 설정 로더
    └── pipeline_logger.py          # 파이프라인 로깅
```

## Tech Stack

- Python 3.10+
- scikit-learn (RandomForest, KMeans)
- XGBoost, LightGBM
- pandas, numpy
- Oracle DB (cx_Oracle)
- PostgreSQL (psycopg2) - 로그 저장
- Apache Airflow - 스케줄링

## Usage

### CLI 실행

```bash
# 전체 파이프라인 실행
python main.py --config configs/config.yaml

# 특정 스텝만 실행
python main.py --config configs/config.yaml --step 3
```

### Airflow 실행

```bash
# DAG 파일을 Airflow dags 폴더에 배치
cp binzib_dag.py $AIRFLOW_HOME/dags/

# Airflow UI에서 binzib_pipeline DAG 실행
```

## Configuration

`configs/config.yaml` 예시:

```yaml
common:
  base_path: /home/was/project/binzib-deploy/binzib_pipeline
  output_path: /home/was/project/binzib-deploy/output

prod:
  oracle:
    host: localhost
    port: 1521
    service_name: ORCL
    user: binzib
    password: ${ORACLE_PASSWORD}
  
  postgresql:
    host: localhost
    port: 5432
    database: binzib_log
    user: binzib
    password: ${PG_PASSWORD}
```

## Data Sources

| 데이터 | 출처 | 용도 |
|--------|------|------|
| 빈집 실태조사 | 국토교통부 | 타겟 변수 (빈집 여부) |
| 건축물대장 | 국토교통부 | 건물 속성 (연면적, 용도, 노후도) |
| 전기 사용량 | 한국전력 | 거주 활동 패턴 |
| 집계구 통계 | 통계청 | 인구, 가구, 주택 현황 |
| 시설 데이터 | 공공데이터포털 | 인프라 접근성 |

## Model Performance

클러스터별 앙상블 모델 성능 (5-fold CV 평균):

| Metric | Score |
|--------|-------|
| AUC-ROC | 0.85+ |
| F1 Score | 0.72+ |
| Precision | 0.70+ |
| Recall | 0.75+ |

## License

Private - 선도소프트 내부 프로젝트
