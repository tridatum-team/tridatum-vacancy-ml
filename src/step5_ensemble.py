# -*- coding: utf-8 -*-
"""
Step 5: 앙상블 및 평가

처리 내용:
1. 전역 앙상블 (가중 평균)
2. 클러스터별 앙상블
3. 표준편차 기반 4단계 빈집 인덱스 생성
4. 성능 평가 (RMSE, MAE, R2, F1)
5. 혼동 행렬 분석
6. 결과 저장 (DB + Parquet + CSV)

출력:
- 19_final_ensemble.parquet
- 19_final_ensemble.csv
- model_metrics.csv
- confusion_matrix.csv
- vacant_index_params.pkl
"""
import os
import gc
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split

from .utils import log_status, normalize_oa_cd, to_db


def create_vacant_index(df, ratio_col='VACANT_RATIO', index_col='VACANT_INDEX', logger=None):
    """
    표준편차 기반 4단계 빈집 인덱스 생성
    
    1단계 (안전): 0 또는 매우 낮음 (< mean)
    2단계 (주의): mean ~ mean + 1*std
    3단계 (경고): mean + 1*std ~ mean + 2*std
    4단계 (위험): mean + 2*std 이상
    """
    print("\n" + "="*70)
    print("표준편차 기반 4단계 빈집 인덱스 생성")
    print("="*70)
    
    if ratio_col not in df.columns:
        print(f"  오류: {ratio_col} 컬럼이 없습니다.")
        return df, {}
    
    mean_val = df[ratio_col].mean()
    std_val = df[ratio_col].std()
    
    print(f"\n  기본 통계:")
    print(f"    평균 (mean): {mean_val:.6f} ({mean_val*100:.4f}%)")
    print(f"    표준편차 (std): {std_val:.6f} ({std_val*100:.4f}%)")
    print(f"    최소값: {df[ratio_col].min():.6f}")
    print(f"    최대값: {df[ratio_col].max():.6f}")
    
    threshold_1 = mean_val
    threshold_2 = mean_val + 1 * std_val
    threshold_3 = mean_val + 2 * std_val
    
    print(f"\n  경계값:")
    print(f"    1단계 (안전): 0 ~ {threshold_1:.6f}")
    print(f"    2단계 (주의): {threshold_1:.6f} ~ {threshold_2:.6f}")
    print(f"    3단계 (경고): {threshold_2:.6f} ~ {threshold_3:.6f}")
    print(f"    4단계 (위험): {threshold_3:.6f} 이상")
    
    def assign_index(value):
        if value <= threshold_1:
            return 1
        elif value <= threshold_2:
            return 2
        elif value <= threshold_3:
            return 3
        else:
            return 4
    
    df[index_col] = df[ratio_col].apply(assign_index)
    
    dist = df[index_col].value_counts().sort_index()
    
    print(f"\n  인덱스 분포:")
    index_labels = {1: '안전', 2: '주의', 3: '경고', 4: '위험'}
    
    for idx in [1, 2, 3, 4]:
        count = dist.get(idx, 0)
        pct = count / len(df) * 100
        avg_ratio = df[df[index_col] == idx][ratio_col].mean() if count > 0 else 0
        print(f"    {idx}단계 ({index_labels[idx]}): {count:,}개 ({pct:.1f}%), 평균 빈집률: {avg_ratio:.4f}")
    
    params = {
        'method': 'std',
        'n_levels': 4,
        'mean': mean_val,
        'std': std_val,
        'threshold_1': threshold_1,
        'threshold_2': threshold_2,
        'threshold_3': threshold_3
    }
    
    return df, params


def apply_vacant_index(ratio_values, params, logger=None):
    """저장된 파라미터로 비율을 인덱스로 변환"""
    if params.get('method') != 'std':
        raise ValueError("파라미터가 표준편차 방식이 아닙니다.")
    
    threshold_1 = params['threshold_1']
    threshold_2 = params['threshold_2']
    threshold_3 = params['threshold_3']
    
    def assign_index(value):
        if pd.isna(value):
            return 1
        elif value <= threshold_1:
            return 1
        elif value <= threshold_2:
            return 2
        elif value <= threshold_3:
            return 3
        else:
            return 4
    
    if isinstance(ratio_values, pd.Series):
        return ratio_values.apply(assign_index).astype(int)
    else:
        return np.array([assign_index(v) for v in ratio_values])


def run_step5(input_dir, output_dir, use_cluster_ensemble=True,
              pg_connection=None, run_id=None, timestemp=None, logger=None):
    """Step 5 실행: 앙상블 및 평가"""
    print("\n" + "="*80)
    print("Step 5: 앙상블 및 평가 (표준편차 기반 4단계)")
    print("="*80)
    start_time = datetime.now()
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n[Step 5-1] 데이터 로드")
    df = pd.read_parquet(os.path.join(input_dir, "16_model_predictions.parquet"))
    
    if 'oa_cd' in df.columns:
        df['oa_cd'] = df['oa_cd'].astype(str).str.strip()
    
    print(f"  데이터: {df.shape}")
    
    print("\n[Step 5-2] 전역 앙상블")
    
    has_gwr = 'VACANT_RATIO_P_GWR' in df.columns
    
    if has_gwr:
        df['VACANT_RATIO_P_ENSEMBLE'] = (
            df['VACANT_RATIO_P_XGB'] * 0.25 +
            df['VACANT_RATIO_P_LGB'] * 0.25 +
            df['VACANT_RATIO_P_CAT'] * 0.25 +
            df['VACANT_RATIO_P_GWR'] * 0.25
        ).clip(0, 1)
    else:
        df['VACANT_RATIO_P_ENSEMBLE'] = (
            df['VACANT_RATIO_P_XGB'] * 0.333 +
            df['VACANT_RATIO_P_LGB'] * 0.333 +
            df['VACANT_RATIO_P_CAT'] * 0.334
        ).clip(0, 1)
    
    print(f"  앙상블 평균: {df['VACANT_RATIO_P_ENSEMBLE'].mean():.4f}")
    
    if 'cluster' in df.columns and use_cluster_ensemble:
        print("\n[Step 5-3] 클러스터별 앙상블")
        
        df['VACANT_RATIO_P_CLUSTER_ENSEMBLE'] = 0.0
        
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_mask = df['cluster'] == cluster_id
            
            xgb_col = f'VACANT_RATIO_P_XGB_C{cluster_id}'
            lgb_col = f'VACANT_RATIO_P_LGB_C{cluster_id}'
            cat_col = f'VACANT_RATIO_P_CAT_C{cluster_id}'
            
            if all(col in df.columns for col in [xgb_col, lgb_col, cat_col]):
                cluster_ensemble = (
                    df.loc[cluster_mask, xgb_col] * 0.333 +
                    df.loc[cluster_mask, lgb_col] * 0.333 +
                    df.loc[cluster_mask, cat_col] * 0.334
                ).clip(0, 1)
                df.loc[cluster_mask, 'VACANT_RATIO_P_CLUSTER_ENSEMBLE'] = cluster_ensemble
            else:
                df.loc[cluster_mask, 'VACANT_RATIO_P_CLUSTER_ENSEMBLE'] = df.loc[cluster_mask, 'VACANT_RATIO_P_ENSEMBLE']
    
    print("\n[Step 5-4] 표준편차 기반 4단계 인덱스 생성")
    
    df, index_params = create_vacant_index(df, 'VACANT_RATIO', 'VACANT_INDEX')
    
    params_path = os.path.join(output_dir, "vacant_index_params.pkl")
    with open(params_path, 'wb') as f:
        pickle.dump(index_params, f)
    print(f"  파라미터 저장: {params_path}")
    
    df['VACANT_P_INDEX_XGB'] = apply_vacant_index(df['VACANT_RATIO_P_XGB'], index_params)
    df['VACANT_P_INDEX_LGB'] = apply_vacant_index(df['VACANT_RATIO_P_LGB'], index_params)
    df['VACANT_P_INDEX_CAT'] = apply_vacant_index(df['VACANT_RATIO_P_CAT'], index_params)
    
    if has_gwr:
        df['VACANT_P_INDEX_GWR'] = apply_vacant_index(df['VACANT_RATIO_P_GWR'], index_params)
    
    df['VACANT_P_INDEX_ENSEMBLE'] = apply_vacant_index(df['VACANT_RATIO_P_ENSEMBLE'], index_params)
    
    if 'VACANT_RATIO_P_CLUSTER_ENSEMBLE' in df.columns:
        df['VACANT_P_INDEX_CLUSTER_ENSEMBLE'] = apply_vacant_index(
            df['VACANT_RATIO_P_CLUSTER_ENSEMBLE'], index_params
        )
    
    print("\n[Step 5-5] 성능 평가")
    
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=42,
        stratify=df['cluster'] if 'cluster' in df.columns else None
    )
    
    y_test = df.loc[test_idx, 'VACANT_RATIO']
    y_test_index = df.loc[test_idx, 'VACANT_INDEX']
    
    results = []
    
    models_to_eval = [
        ('XGBoost', 'VACANT_RATIO_P_XGB', 'VACANT_P_INDEX_XGB'),
        ('LightGBM', 'VACANT_RATIO_P_LGB', 'VACANT_P_INDEX_LGB'),
        ('CatBoost', 'VACANT_RATIO_P_CAT', 'VACANT_P_INDEX_CAT'),
        ('ENSEMBLE', 'VACANT_RATIO_P_ENSEMBLE', 'VACANT_P_INDEX_ENSEMBLE')
    ]
    
    if has_gwr:
        models_to_eval.insert(3, ('GWR', 'VACANT_RATIO_P_GWR', 'VACANT_P_INDEX_GWR'))
    
    if 'VACANT_RATIO_P_CLUSTER_ENSEMBLE' in df.columns:
        models_to_eval.append(('CLUSTER_ENS', 'VACANT_RATIO_P_CLUSTER_ENSEMBLE', 'VACANT_P_INDEX_CLUSTER_ENSEMBLE'))
    
    for name, ratio_col, index_col in models_to_eval:
        y_pred = df.loc[test_idx, ratio_col]
        y_pred_index = df.loc[test_idx, index_col]
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        acc = accuracy_score(y_test_index, y_pred_index)
        f1 = f1_score(y_test_index, y_pred_index, average='weighted')
        
        print(f"\n  [{name}]")
        print(f"    회귀 - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        print(f"    분류 - Accuracy: {acc*100:.2f}%, F1: {f1:.4f}")
        
        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Accuracy': acc,
            'F1_Score': f1
        })
    
    print("\n[Step 5-6] 혼동 행렬 (앙상블)")
    
    cm = confusion_matrix(df['VACANT_INDEX'], df['VACANT_P_INDEX_ENSEMBLE'])
    
    print("\n  실제\\예측    1(안전)  2(주의)  3(경고)  4(위험)")
    index_labels = {1: '안전', 2: '주의', 3: '경고', 4: '위험'}
    for i, row in enumerate(cm, 1):
        row_str = f"    {i}({index_labels[i]})  " + "  ".join([f"{val:7d}" for val in row])
        print(row_str)
    
    print("\n[Step 5-7] 결과 저장")
    
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, "model_metrics.csv")
    results_df.to_csv(results_path, index=False, encoding='cp949')
    print(f"  메트릭: {results_path}")
    
    cm_df = pd.DataFrame(
        cm,
        index=[f'실제_{i}({index_labels[i]})' for i in range(1, 5)],
        columns=[f'예측_{i}({index_labels[i]})' for i in range(1, 5)]
    )
    cm_path = os.path.join(output_dir, "confusion_matrix.csv")
    cm_df.to_csv(cm_path, encoding='cp949')
    print(f"  혼동 행렬: {cm_path}")
    
    output_cols = ['oa_cd', 'ANLS_EFFECT_FACTORb']
    if 'cluster' in df.columns:
        output_cols.append('cluster')
    
    output_cols.extend([
        'VACANT_RATIO', 'VACANT_INDEX',
        'VACANT_RATIO_P_XGB', 'VACANT_RATIO_P_LGB', 'VACANT_RATIO_P_CAT',
        'VACANT_RATIO_P_ENSEMBLE',
        'VACANT_P_INDEX_XGB', 'VACANT_P_INDEX_LGB', 'VACANT_P_INDEX_CAT',
        'VACANT_P_INDEX_ENSEMBLE'
    ])
    
    if has_gwr:
        output_cols.insert(output_cols.index('VACANT_RATIO_P_ENSEMBLE'), 'VACANT_RATIO_P_GWR')
        output_cols.insert(output_cols.index('VACANT_P_INDEX_ENSEMBLE'), 'VACANT_P_INDEX_GWR')
    
    if 'VACANT_RATIO_P_CLUSTER_ENSEMBLE' in df.columns:
        output_cols.extend(['VACANT_RATIO_P_CLUSTER_ENSEMBLE', 'VACANT_P_INDEX_CLUSTER_ENSEMBLE'])
    
    if 'X' in df.columns:
        output_cols.append('X')
    if 'Y' in df.columns:
        output_cols.append('Y')
    
    df_output = df[[c for c in output_cols if c in df.columns]].copy()
    
    numeric_cols = df_output.select_dtypes(include=['float32', 'float64']).columns
    for col in numeric_cols:
        df_output[col] = df_output[col].replace([np.inf, -np.inf], np.nan)
        df_output[col] = df_output[col].fillna(0)
    
    clip_threshold = 1e9
    for col in numeric_cols:
        max_val = df_output[col].abs().max()
        if max_val > clip_threshold:
            df_output[col] = df_output[col].clip(-clip_threshold, clip_threshold)
    
    for col in numeric_cols:
        df_output[col] = df_output[col].astype('float32')
    
    model_path = os.path.join(input_dir, "13_two_stage_models.pkl")
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            models = pickle.load(f)

        cluster_top_features = models.get('cluster_top_features', {})

        def get_top5_features(cluster_id):
            features = cluster_top_features.get(cluster_id, [])[:5]
            kor_features = []
            seen = set()
            for f in features:
                name = f.replace('_', ' ')
                if (name.endswith('_x') or name.endswith('_y')) and len(name) > 2:
                    name = name[:-2]
                name = name.replace('_',' ')

                if name not in seen:
                    kor_features.append(name)
                    seen.add(name)

            return ' '.join(kor_features[:5])
        
        if 'cluster' in df.columns:
            df['ANLS_EFFECT_FACTOR'] = df['cluster'].apply(get_top5_features)
    else:
        df['ANLS_EFFECT_FACTOR'] = ''

    if pg_connection is not None and timestamp is not None:
        df_output['run_id'] = run_id
        #value 값 상위 5개 upper // 변수는 한국어로 넣기 ex. 문화,체육,... 
        to_db(df_output, f'tb_pipeline_prediction_ensemble', pg_connection,
              db_type='postgresql', force_text_cols=['oa_cd'], if_exists='append',
              varchar_lengths={'oa_cd': 20}, primary_key=['oa_cd','run_id'])
    
    output_path = os.path.join(output_dir, "19_final_ensemble.parquet")
    df_output.to_parquet(output_path, index=False)
    print(f"  최종 데이터: {output_path}")
    
    output_csv = os.path.join(output_dir, "19_final_ensemble.csv")
    df_output.to_csv(output_csv, index=False, encoding='cp949')
    print(f"  최종 데이터: {output_csv}")
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds() / 60
    
    print("\n" + "="*80)
    print(f"Step 5 완료! 소요 시간: {elapsed:.1f}분")
    print("="*80)
    
    return df_output


if __name__ == "__main__":
    import yaml
    import psycopg2
    
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    base_path = config['paths']['base_path']
    output_dirs = [d for d in os.listdir(base_path) if d.startswith('output_')]
    latest_output = sorted(output_dirs)[-1]
    
    input_dir = os.path.join(base_path, latest_output, "step4")
    output_dir = os.path.join(base_path, latest_output, "step5")
    
    pg_conn = None
    if config['postgresql'].get('enable', False):
        try:
            pg_conn = psycopg2.connect(
                user=config['postgresql']['user'],
                password=config['postgresql']['password'],
                host=config['postgresql']['host'],
                port=config['postgresql']['port'],
                database=config['postgresql']['database']
            )
        except Exception as e:
            print(f"PostgreSQL 연결 실패: {e}")
    
    run_step5(
        input_dir=input_dir,
        output_dir=output_dir,
        use_cluster_ensemble=config['step5'].get('use_cluster_ensemble', True),
        pg_connection=pg_conn
    )
    
    if pg_conn:
        pg_conn.close()
