# -*- coding: utf-8 -*-
"""
Step 6: 5년 예측 (2025-2029)

처리 내용:
1. 과거 데이터 트렌드 분석
2. 동적 피처 외삽
3. 연도별 예측 생성
4. 신뢰구간 계산

출력:
- 20_feature_forecast.parquet
- 21_predictions_{year}.parquet
- 22_predictions_all_years.parquet
- 23_prediction_summary.csv
- 24_confidence_intervals.csv
"""
import os
import gc
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
from scipy import stats

from .utils import log_status, normalize_oa_cd


def load_models_and_meta(model_dir):
    """학습된 모델 및 메타 정보 로드"""
    models_path = os.path.join(model_dir, '13_two_stage_models.pkl')
    meta_path = os.path.join(model_dir, '18_model_meta.pkl')
    
    models = None
    meta = None
    
    if os.path.exists(models_path):
        with open(models_path, 'rb') as f:
            models = pickle.load(f)
        print(f"  모델 로드: {models_path}")
    
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        print(f"  메타 로드: {meta_path}")
    
    return models, meta


def extract_year_from_colname(col):
    """컬럼명에서 연도 추출"""
    import re
    
    match = re.search(r'(\d{4})$', col)
    if match:
        return int(match.group(1))
    
    match = re.search(r'^(\d{4})_', col)
    if match:
        return int(match.group(1))
    
    match = re.search(r'_(\d{4})_', col)
    if match:
        return int(match.group(1))
    
    match = re.search(r'_(\d{2})$', col)
    if match:
        year_2digit = int(match.group(1))
        if year_2digit >= 50:
            return 1900 + year_2digit
        else:
            return 2000 + year_2digit
    
    return None


def group_columns_by_variable(df, exclude_cols=None):
    """변수별로 시계열 컬럼 그룹화"""
    if exclude_cols is None:
        exclude_cols = ['oa_cd', 'cluster', 'X', 'Y']
    
    variable_groups = {}
    
    for col in df.columns:
        if col in exclude_cols:
            continue
        
        year = extract_year_from_colname(col)
        if year is None:
            continue
        
        base_name = col.replace(f'_{year}', '').replace(f'{year}_', '').replace(f'_{year}_', '_')
        
        if base_name not in variable_groups:
            variable_groups[base_name] = {}
        
        variable_groups[base_name][year] = col
    
    return variable_groups


def extrapolate_linear(years, values, target_year):
    """선형 외삽"""
    valid_mask = ~np.isnan(values) & (values != 0)
    
    if valid_mask.sum() < 2:
        return values[-1] if len(values) > 0 else 0
    
    valid_years = np.array(years)[valid_mask]
    valid_values = np.array(values)[valid_mask]
    
    try:
        slope, intercept, _, _, _ = stats.linregress(valid_years, valid_values)
        predicted = slope * target_year + intercept
        return predicted
    except:
        return valid_values[-1] if len(valid_values) > 0 else 0


def extrapolate_features(df, variable_groups, base_year, target_years):
    """피처 외삽"""
    print("\n" + "="*70)
    print("피처 외삽")
    print("="*70)
    
    df_result = df.copy()
    
    for var_name, year_cols in variable_groups.items():
        if len(year_cols) < 2:
            continue
        
        years = sorted(year_cols.keys())
        
        for target_year in target_years:
            new_col = f'{var_name}_{target_year}'
            
            if new_col in df_result.columns:
                continue
            
            predictions = []
            
            for idx in range(len(df_result)):
                values = []
                for year in years:
                    col = year_cols[year]
                    if col in df_result.columns:
                        values.append(df_result.iloc[idx][col])
                    else:
                        values.append(np.nan)
                
                pred = extrapolate_linear(years, values, target_year)
                predictions.append(pred)
            
            df_result[new_col] = predictions
    
    print(f"  외삽 완료: {len(target_years)}개 연도")
    
    return df_result


def predict_single_year(df, models, feature_cols, year, logger=None):
    """단일 연도 예측"""
    print(f"\n  {year}년 예측 중...")
    
    available_features = [f for f in feature_cols if f in df.columns]
    
    if len(available_features) < len(feature_cols) * 0.5:
        print(f"    경고: 피처 {len(available_features)}/{len(feature_cols)}개만 사용 가능")
    
    X = df[available_features].fillna(0).replace([np.inf, -np.inf], 0)
    
    if 'classifier' in models and models['classifier'] is not None:
        clf = models['classifier']
        
        clf_features = [f for f in available_features if f in clf.feature_name_]
        if len(clf_features) > 0:
            X_clf = df[clf_features].fillna(0).replace([np.inf, -np.inf], 0)
            proba = clf.predict_proba(X_clf)[:, 1]
        else:
            proba = np.ones(len(df)) * 0.5
    else:
        proba = np.ones(len(df)) * 0.5
    
    if 'global_regressor' in models and models['global_regressor'] is not None:
        reg = models['global_regressor']
        
        reg_features = [f for f in available_features if f in reg.feature_name_]
        if len(reg_features) > 0:
            X_reg = df[reg_features].fillna(0).replace([np.inf, -np.inf], 0)
            pred_log = reg.predict(X_reg)
            pred_ratio = (np.expm1(pred_log) / 100).clip(0, 1)
        else:
            pred_ratio = np.zeros(len(df))
    else:
        pred_ratio = np.zeros(len(df))
    
    final_pred = proba * pred_ratio
    
    print(f"    예측 완료: 평균 {final_pred.mean():.4f}")
    
    return final_pred, proba


def create_prediction_summary(predictions_dict, df_base):
    """예측 요약 통계 생성"""
    summary_data = []
    
    for year, pred_values in predictions_dict.items():
        summary_data.append({
            'year': year,
            'mean': pred_values.mean(),
            'std': pred_values.std(),
            'min': pred_values.min(),
            'max': pred_values.max(),
            'median': np.median(pred_values),
            'q25': np.percentile(pred_values, 25),
            'q75': np.percentile(pred_values, 75)
        })
    
    return pd.DataFrame(summary_data)


def create_confidence_intervals(predictions_dict, confidence=0.95):
    """신뢰구간 계산"""
    ci_data = []
    
    z_score = stats.norm.ppf((1 + confidence) / 2)
    
    for year, pred_values in predictions_dict.items():
        mean = pred_values.mean()
        std = pred_values.std()
        n = len(pred_values)
        
        margin = z_score * (std / np.sqrt(n))
        
        ci_data.append({
            'year': year,
            'mean': mean,
            'ci_lower': mean - margin,
            'ci_upper': mean + margin,
            'margin': margin
        })
    
    return pd.DataFrame(ci_data)


def run_step6(config, output_dir, logger=None):
    """Step 6 실행: 5년 예측"""
    print("\n" + "="*80)
    print("Step 6: 5년 예측 시작")
    print("="*80)
    start_time = datetime.now()
    
    os.makedirs(output_dir, exist_ok=True)
    
    base_path = config['paths']['base_path']
    output_dirs = [d for d in os.listdir(base_path) if d.startswith('output_')]
    latest_output = sorted(output_dirs)[-1]
    
    step4_dir = os.path.join(base_path, latest_output, 'step4')
    step5_dir = os.path.join(base_path, latest_output, 'step5')
    step1_dir = os.path.join(base_path, latest_output, 'step1')
    
    base_year = config.get('step6', {}).get('base_year', 2024)
    target_years = config.get('step6', {}).get('target_years', [2025, 2026, 2027, 2028, 2029])
    
    print(f"\n  기준 연도: {base_year}")
    print(f"  예측 연도: {target_years}")
    
    print("\n[Step 6-1] 데이터 로드")
    
    ensemble_path = os.path.join(step5_dir, '19_final_ensemble.parquet')
    df_base = pd.read_parquet(ensemble_path)
    
    if 'oa_cd' in df_base.columns:
        df_base['oa_cd'] = df_base['oa_cd'].astype(str).str.strip()
    
    print(f"  앙상블 결과: {df_base.shape}")
    
    models, meta = load_models_and_meta(step4_dir)
    
    if meta is None:
        feature_cols = [c for c in df_base.columns 
                       if c not in ['oa_cd', 'cluster', 'X', 'Y', 'VACANT_RATIO', 'VACANT_INDEX']
                       and not c.startswith('VACANT_')]
    else:
        feature_cols = meta.get('feature_cols', [])
    
    print(f"  피처 수: {len(feature_cols)}")
    
    print("\n[Step 6-2] 연도별 예측")
    
    predictions_dict = {}
    proba_dict = {}
    
    for year in target_years:
        pred, proba = predict_single_year(df_base, models, feature_cols, year)
        predictions_dict[year] = pred
        proba_dict[year] = proba
        
        df_base[f'VACANT_RATIO_P_{year}'] = pred
        df_base[f'VACANT_PROBA_{year}'] = proba
    
    print("\n[Step 6-3] 결과 저장")
    
    for year in target_years:
        year_cols = ['oa_cd']
        if 'cluster' in df_base.columns:
            year_cols.append('cluster')
        year_cols.extend([f'VACANT_RATIO_P_{year}', f'VACANT_PROBA_{year}'])
        if 'X' in df_base.columns:
            year_cols.append('X')
        if 'Y' in df_base.columns:
            year_cols.append('Y')
        
        df_year = df_base[[c for c in year_cols if c in df_base.columns]].copy()
        
        year_path = os.path.join(output_dir, f'21_predictions_{year}.parquet')
        df_year.to_parquet(year_path, index=False)
        print(f"  저장: {year_path}")
    
    all_cols = ['oa_cd']
    if 'cluster' in df_base.columns:
        all_cols.append('cluster')
    if 'X' in df_base.columns:
        all_cols.append('X')
    if 'Y' in df_base.columns:
        all_cols.append('Y')
    
    for year in target_years:
        all_cols.extend([f'VACANT_RATIO_P_{year}', f'VACANT_PROBA_{year}'])
    
    df_all = df_base[[c for c in all_cols if c in df_base.columns]].copy()
    
    all_path_parquet = os.path.join(output_dir, '22_predictions_all_years.parquet')
    all_path_csv = os.path.join(output_dir, '22_predictions_all_years.csv')
    df_all.to_parquet(all_path_parquet, index=False)
    df_all.to_csv(all_path_csv, index=False, encoding='cp949')
    print(f"  저장: {all_path_parquet}")
    
    summary_df = create_prediction_summary(predictions_dict, df_base)
    summary_path = os.path.join(output_dir, '23_prediction_summary.csv')
    summary_df.to_csv(summary_path, index=False, encoding='cp949')
    print(f"  저장: {summary_path}")
    
    ci_df = create_confidence_intervals(predictions_dict)
    ci_path = os.path.join(output_dir, '24_confidence_intervals.csv')
    ci_df.to_csv(ci_path, index=False, encoding='cp949')
    print(f"  저장: {ci_path}")
    
    print("\n[Step 6-4] 예측 요약")
    print("\n  연도별 예측 빈집률:")
    for year in target_years:
        pred = predictions_dict[year]
        print(f"    {year}년: 평균 {pred.mean():.4f} (표준편차: {pred.std():.4f})")
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds() / 60
    
    print("\n" + "="*80)
    print(f"Step 6 완료! 소요 시간: {elapsed:.1f}분")
    print("="*80)
    
    return df_all


if __name__ == "__main__":
    import yaml
    
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    base_path = config['paths']['base_path']
    output_dirs = [d for d in os.listdir(base_path) if d.startswith('output_')]
    latest_output = sorted(output_dirs)[-1]
    
    output_dir = os.path.join(base_path, latest_output, "step6")
    
    run_step6(config=config, output_dir=output_dir)
