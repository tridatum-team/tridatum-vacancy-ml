# -*- coding: utf-8 -*-
"""
Step 1: 데이터 통합

처리 내용:
1. MAIN_DATA 로드 (Oracle DB)
2. 전기 데이터 와이드 변환 + 파생변수
3. PU Learning (빈집 라벨 보정)
4. 외부 집계구 데이터 로드
5. 외부 시설/기초수급 데이터 로드

출력:
- 01_main_data.parquet
- 02_energy_wide.parquet
- 03_merged_main_energy.parquet
- 04_external_oa_v1.parquet
- 05_external_facility.parquet
"""
import os
import gc
import pickle
from datetime import datetime
from multiprocessing import cpu_count
from shapely.geometry import Point

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .utils import (
    log_status, norm_str, normalize_oa_cd, 
    map_oa_cd_spatial_with_nearest, to_db, N_CORES
)


def prepare_pu_data(df, target_col='VACANT', exclude_cols=None):
    """PU Learning용 데이터 준비"""
    print("\n" + "="*70)
    print("PU Learning 데이터 준비")
    print("="*70)
    
    if exclude_cols is None:
        exclude_cols = []
    
    default_exclude = [
        target_col, 'oa_cd', 'BLDRGST_SEQNO', 'PNU', 'PNU_CD',
        'X', 'Y', 'X_num', 'Y_num', 'geometry',
        'VACANT_sum', 'VACANT_mean', 'VACANT_RATIO', 'building_count',
        '시도', '시군구', '읍면동', 'SGG_CD', 'EMD_CD', 'SD_CD', 'AMD_CD'
    ]
    exclude_cols = list(set(exclude_cols + default_exclude))
    
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    
    mask_labeled = df[target_col].notna()
    df_labeled = df[mask_labeled].copy()
    df_unlabeled = df[~mask_labeled].copy()
    
    print(f"\n데이터 분리:")
    print(f"  Labeled (조사됨): {len(df_labeled):,}건")
    print(f"  Unlabeled (미조사): {len(df_unlabeled):,}건")
    
    n_positive = (df_labeled[target_col] == 1).sum()
    n_negative = (df_labeled[target_col] == 0).sum()
    print(f"\nLabeled 내 분포:")
    print(f"  Positive (빈집=1): {n_positive:,}건 ({n_positive/len(df_labeled)*100:.2f}%)")
    print(f"  Negative (비빈집=0): {n_negative:,}건 ({n_negative/len(df_labeled)*100:.2f}%)")
    
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols 
                   and df[col].dtype in ['int8', 'int16', 'int32', 'int64', 
                                         'uint8', 'uint16', 'uint32',
                                         'float16', 'float32', 'float64']]
    
    print(f"\n피처 컬럼: {len(feature_cols)}개")
    
    return df_labeled, df_unlabeled, feature_cols


def train_pu_classifier(df_labeled, feature_cols, target_col='VACANT', 
                        model_type='lightgbm', n_estimators=200):
    """PU Learning 1단계: Labeled 데이터로 분류기 학습"""
    print("\n" + "="*70)
    print(f"PU Learning 분류기 학습 ({model_type})")
    print("="*70)
    
    X = df_labeled[feature_cols].copy()
    y = df_labeled[target_col].astype(int)
    
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n학습 데이터: {len(X_train):,}건")
    print(f"검증 데이터: {len(X_val):,}건")
    
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1
    
    print(f"\n클래스 불균형:")
    print(f"  Positive: {n_pos:,}건")
    print(f"  Negative: {n_neg:,}건")
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")
    
    if model_type == 'lightgbm':
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=31,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=N_CORES,
            verbose=-1
        )
    else:
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=N_CORES,
            verbosity=0
        )
    
    log_status("모델 학습 시작")
    model.fit(X_train, y_train)
    log_status("모델 학습 완료")
    
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    print(f"\n검증 성능:")
    print(f"  Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"  Precision: {precision_score(y_val, y_val_pred):.4f}")
    print(f"  Recall: {recall_score(y_val, y_val_pred):.4f}")
    print(f"  F1-Score: {f1_score(y_val, y_val_pred):.4f}")
    print(f"  ROC-AUC: {roc_auc_score(y_val, y_val_proba):.4f}")
    
    return model, X_train, X_val, y_train, y_val


def estimate_class_prior(model, X_positive):
    """PU Learning: Class Prior (c) 추정"""
    print("\n" + "="*70)
    print("Class Prior (c) 추정")
    print("="*70)
    
    proba_positive = model.predict_proba(X_positive)[:, 1]
    c = proba_positive.mean()
    
    print(f"  Positive 샘플 수: {len(X_positive):,}건")
    print(f"  예측 확률 평균: {proba_positive.mean():.4f}")
    print(f"  예측 확률 중간값: {np.median(proba_positive):.4f}")
    print(f"  추정된 c: {c:.4f}")
    
    return c


def predict_unlabeled(model, df_unlabeled, feature_cols, c=None, 
                      target_vacancy_rate=0.007, chunk_size=500000):
    """PU Learning 2단계: Unlabeled 데이터 예측"""
    print("\n" + "="*70)
    print("Unlabeled 데이터 예측")
    print("="*70)
    
    n_total = len(df_unlabeled)
    print(f"  총 예측 대상: {n_total:,}건")
    
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    all_proba = []
    
    log_status(f"청크 단위 예측 시작 ({n_chunks}개 청크)")
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_total)
        
        X_chunk = df_unlabeled.iloc[start_idx:end_idx][feature_cols].copy()
        X_chunk = X_chunk.fillna(0).replace([np.inf, -np.inf], 0)
        
        proba_chunk = model.predict_proba(X_chunk)[:, 1]
        all_proba.extend(proba_chunk)
        
        if (i + 1) % 10 == 0 or i == n_chunks - 1:
            print(f"    청크 {i+1}/{n_chunks} 완료")
        
        del X_chunk
        gc.collect()
    
    proba_raw = np.array(all_proba)
    
    if c is not None and c > 0:
        print(f"\n확률 보정 (c={c:.4f})")
        proba_calibrated = proba_raw / c
        proba_calibrated = np.clip(proba_calibrated, 0, 1)
    else:
        proba_calibrated = proba_raw
    
    print(f"\n임계값 결정 (목표 빈집률: {target_vacancy_rate*100:.2f}%)")
    
    sorted_proba = np.sort(proba_calibrated)[::-1]
    target_count = int(n_total * target_vacancy_rate)
    
    if target_count > 0 and target_count < len(sorted_proba):
        threshold = sorted_proba[target_count]
    else:
        threshold = 0.5
    
    print(f"  계산된 임계값: {threshold:.4f}")
    print(f"  예상 빈집 수: {target_count:,}건")
    
    df_unlabeled['VACANT_proba'] = proba_calibrated
    df_unlabeled['VACANT_pred'] = (proba_calibrated >= threshold).astype(int)
    
    n_pred_vacant = df_unlabeled['VACANT_pred'].sum()
    pred_rate = n_pred_vacant / n_total
    
    print(f"\n예측 결과:")
    print(f"  예측 빈집: {n_pred_vacant:,}건 ({pred_rate*100:.4f}%)")
    print(f"  예측 비빈집: {n_total - n_pred_vacant:,}건")
    

    return df_unlabeled, threshold

def run_pu_learning(df, target_col='VACANT', model_type='lightgbm',
                    target_vacancy_rate=0.007, output_dir=None):
    """PU Learning 전체 파이프라인 실행"""
    print("\n" + "="*80)
    print("PU Learning 파이프라인 시작")
    print("="*80)
    start_time = datetime.now()
    
    df_labeled, df_unlabeled, feature_cols = prepare_pu_data(df, target_col)
    
    model, X_train, X_val, y_train, y_val = train_pu_classifier(
        df_labeled, feature_cols, target_col, model_type
    )
    
    mask_positive = df_labeled[target_col] == 1
    X_positive = df_labeled.loc[mask_positive, feature_cols].fillna(0)
    c = estimate_class_prior(model, X_positive)
    
    df_unlabeled, threshold = predict_unlabeled(
        model, df_unlabeled, feature_cols, c, target_vacancy_rate
    )
    
    print("\n" + "="*70)
    print("결과 병합")
    print("="*70)
    
    df_labeled['VACANT_proba'] = model.predict_proba(
        df_labeled[feature_cols].fillna(0)
    )[:, 1]
    df_labeled['VACANT_pred'] = df_labeled[target_col].astype(int)
    
    df_result = pd.concat([df_labeled, df_unlabeled], ignore_index=True)
    df_result['VACANT_final'] = df_result[target_col].fillna(df_result['VACANT_pred'])
    
    print(f"\n최종 결과:")
    print(f"  전체: {len(df_result):,}건")
    print(f"  빈집 (VACANT_final=1): {(df_result['VACANT_final']==1).sum():,}건")
    print(f"  비빈집 (VACANT_final=0): {(df_result['VACANT_final']==0).sum():,}건")
    print(f"  최종 빈집률: {df_result['VACANT_final'].mean()*100:.4f}%")
    
    print("\n" + "="*70)
    print("Feature Importance (Top 20)")
    print("="*70)
    
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in importance.head(20).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, f'pu_model_{model_type}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        log_status(f"모델 저장: {model_path}")
        
        importance_path = os.path.join(output_dir, 'feature_importance.csv')
        importance.to_csv(importance_path, index=False, encoding='cp949')
        log_status(f"Feature Importance 저장: {importance_path}")
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds() / 60
    
    print("\n" + "="*80)
    print(f"PU Learning 완료! 소요 시간: {elapsed:.1f}분")
    print("="*80)
    
    return df_result, model

def load_facility_from_db(connection, xy_mapper_path, config=None, logger=None):
    #LOCALDATA_TEMP -> LOCALDATA로 수정
    sql="""
    SELECT FACILITY_TYPE, X, Y, TRD_STATE_NM
    FROM LOCALDATA_TEMP
    WHERE X IS NOT NULL
      AND X IS NOT NULL
      AND X != '0'
      AND Y != '0'
    """

    df = pd.read_sql(sql,connection)

    df['TRD_STATE_NM'] = df['TRD_STATE_NM'].astype(str)
    df= df[df['TRD_STATE_NM'].str.contains('영업', na=False)].copy()

    if len(df) == 0 :
        return pd.DataFrame(columns=['oa_cd'])

    df['X'] = pd.to_numeric(df['X'], errors='coerce')
    df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
    df = df[(df['X'].notna()) & (df['Y'].notna())].copy()

    df = map_facility_to_oa(df, xy_mapper_path, config=config, source_crs='EPSG:5174')

    df = df[df['oa_cd'].notna()].copy()

    if len(df) == 0:
        print("매핑 실패(시설데이터)")
    
    df_agg =df.groupby(['oa_cd', 'FACILITY_TYPE']).size().unstack(fill_value=0)
    df_agg.columns = [f'{col}_2024' for col in df_agg.columns]
    df_agg = df_agg.reset_index()

    df_agg = normalize_oa_cd(df_agg, 'oa_cd')

    print(f"\n 집계 완료: {df_agg.shape}")
    for col in df_agg.columns:
        if col != 'oa_cd':
            print(f" {col}: 합계 {df_agg[col].sum():,}건")
    log_status("시설데이터 집계 완료")
    return df_agg

def map_facility_to_oa(df_xy_mapper_path, config=None, source_crs='EPSG:5174'):
    from .utils import load_oa_boundary, norm_str

    gdf_boundary = load_oa_boundary(config, xy_mapper_path)

    if gdf_boundary.crs is None:
        gdf_boundary.set_crs('EPSG:5186', inplace=True)

    target_crs = str(gdf_boundary.crs)

    geometry = [Point(xy) for xy in zip(df['X'], df['Y'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=source_crs)

    if source_crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    
    gdf_joined = gpd.sjoin(
        gdf,
        gdf_boundary[['TOT_REG_CD', 'geometry']],
        how='left',
        predicate='intersects'
    )

    gdf_joined = gdf_joined[~gdf_joined.index.duplicated(keep='first')]

    oa_col = 'TOT_REG_CD_right' if 'TOT_REG_CD_right' in gdf_joined.columns else 'TOT_REG_CD'
    df['oa_cd'] = gdf_joined[oa_col].values if oa_col in gdf_joined.columns else None

    matched = df['oa_cd'].notna().sum()

    del gdf_boundary, gdf, gdf_joined
    gc.collect()

    return df

def load_main_data(connection, xy_mapper_path, config=None, table_name='ML_BIS_WEB_TOT_BLDG_MAIN',
                   distance_threshold=2.0, remove_unmapped=True, logger=None):
    """MAIN_DATA 로드"""
    print("\n" + "="*70)
    print("MAIN_DATA 로드")
    print("="*70)
    
    df = pd.read_sql(f"SELECT * FROM {table_name}", connection)
    df = df.astype(str)
    
    print(f"  로드 완료: {df.shape}")
    print(f"  컬럼 수: {len(df.columns)}")
    print(f"  메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    
    if 'BLDRGST_SEQNO' in df.columns:
        unique_seqnos = df['BLDRGST_SEQNO'].nunique()
        print(f"  고유 BLDRGST_SEQNO: {unique_seqnos:,}개")
        df['BLDRGST_SEQNO'] = norm_str(df['BLDRGST_SEQNO'])
    
    df, base_year = map_oa_cd_spatial_with_nearest(df, xy_mapper_path, config=None,
                                        nearest_threshold_m=distance_threshold * 1000)
    
    if remove_unmapped:
        before_len = len(df)
        df = df[df['oa_cd'].notna()].copy()
        after_len = len(df)
        removed = before_len - after_len
        print(f"\n미매칭 제거: {removed:,}건 제거 ({before_len:,} -> {after_len:,})")
    
    drop_cols = ['lon_mapper', 'lat_mapper', 'X_num', 'Y_num', 'lon', 'lat']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    df = normalize_oa_cd(df, 'oa_cd')
    
    log_status("MAIN_DATA 로드 완료")
    return df, base_year


def process_energy_unified(connection, target_seqnos=None, table_name='ENGY_USE_YEAR_PT', logger=None):
    """전기 데이터 와이드 변환 + 파생변수 생성 통합"""
    print("\n" + "="*70)
    print("전기 데이터 처리")
    print("="*70)
    
    month_cols = [f'U_{i}' for i in range(1, 13)]
    use_cols = ['BLDRGST_SEQNO', 'YEAR'] + month_cols
    cols_str = ', '.join(use_cols)
    
    df_energy = pd.read_sql(f"SELECT {cols_str} FROM {table_name}", connection)
    print(f"  로드 완료: {df_energy.shape}")
    
    df_energy['BLDRGST_SEQNO'] = df_energy['BLDRGST_SEQNO'].astype(str).str.strip()
    df_energy['YEAR'] = df_energy['YEAR'].astype(str).str.strip()
    
    if target_seqnos:
        log_status("target_seqnos 필터링")
        original_len = len(df_energy)
        df_energy = df_energy[df_energy['BLDRGST_SEQNO'].isin(target_seqnos)]
        filtered_len = len(df_energy)
        print(f"  필터링: {original_len:,} -> {filtered_len:,} ({filtered_len/original_len*100:.1f}%)")
    
    for col in month_cols:
        df_energy[col] = pd.to_numeric(df_energy[col], errors='coerce').fillna(0)
    
    log_status("와이드 변환 시작")
    
    unique_years = sorted(df_energy['YEAR'].unique())
    print(f"    연도: {unique_years}")
    
    df_wide = None
    
    for year in unique_years:
        df_year = df_energy[df_energy['YEAR'] == year].copy()
        
        rename_dict = {f'U_{i}': f'{year}_U_{i}' for i in range(1, 13)}
        df_year = df_year.rename(columns=rename_dict)
        df_year = df_year.drop(columns=['YEAR'])
        
        year_cols = [col for col in df_year.columns if col != 'BLDRGST_SEQNO']
        agg_dict = {col: 'sum' for col in year_cols}
        df_year = df_year.groupby('BLDRGST_SEQNO', as_index=False).agg(agg_dict)
        
        if df_wide is None:
            df_wide = df_year
        else:
            df_wide = df_wide.merge(df_year, on='BLDRGST_SEQNO', how='outer')
        
        print(f"  {year} 처리 완료: {len(df_year):,}건")
        
        del df_year
        gc.collect()
    
    df_wide = df_wide.fillna(0)
    print(f"  와이드 변환 완료: {df_wide.shape}")
    
    log_status("파생변수 생성")
    
    year_month_cols = [col for col in df_wide.columns if col != 'BLDRGST_SEQNO']
    
    if len(year_month_cols) == 0:
        df_wide['mean_kwh'] = 0.0
        df_wide['zero_ratio'] = 0.0
        df_wide['drop_ratio'] = 0.0
        df_wide['sudden_drop_ratio'] = 0.0
        return df_wide
    
    print("  연도_월 타입 변환")
    for col in year_month_cols:
        df_wide[col] = pd.to_numeric(df_wide[col], errors='coerce').fillna(0).astype('float32')
    
    chunk_size = 400000
    n_chunks = (len(df_wide) + chunk_size - 1) // chunk_size
    
    all_mean_kwh = []
    all_zero_ratio = []
    all_drop_ratio = []
    all_sudden_drop_ratio = []
    all_consecutive_zero = []
    all_max_drop = []
    all_yoy_mean = []
    all_seasonal = []
    all_cv_data = []
    all_recent_zero = []
    
    import time
    start_time = time.time()
    
    for i in range(n_chunks):
        chunk_start = time.time()
        
        start_idx = i * chunk_size
        end_idx = min((i+1) * chunk_size, len(df_wide))
        df_chunk = df_wide.iloc[start_idx:end_idx].copy()
        
        df_long_chunk = df_chunk[['BLDRGST_SEQNO'] + year_month_cols].melt(
            id_vars=['BLDRGST_SEQNO'],
            value_vars=year_month_cols,
            var_name='연도_월',
            value_name='사용량'
        )
        
        df_long_chunk['year'] = df_long_chunk['연도_월'].str.split('_').str[0]
        df_long_chunk['month'] = df_long_chunk['연도_월'].str.split('_').str[2].astype(int)
        df_long_chunk['사용량'] = df_long_chunk['사용량'].astype('float32')
        
        mean_kwh_chunk = df_long_chunk.groupby('BLDRGST_SEQNO')['사용량'].mean().reset_index()
        mean_kwh_chunk.columns = ['BLDRGST_SEQNO', 'mean_kwh']
        all_mean_kwh.append(mean_kwh_chunk)
        
        zero_ratio_chunk = (df_long_chunk['사용량'] == 0).groupby(df_long_chunk['BLDRGST_SEQNO']).mean().reset_index()
        zero_ratio_chunk.columns = ['BLDRGST_SEQNO', 'zero_ratio']
        all_zero_ratio.append(zero_ratio_chunk)
        
        df_diff = df_long_chunk.sort_values(['BLDRGST_SEQNO', '연도_월']).copy()
        df_diff['사용량_prev'] = df_diff.groupby('BLDRGST_SEQNO')['사용량'].shift(1)
        df_diff['pct_change'] = (df_diff['사용량'] - df_diff['사용량_prev']) / (df_diff['사용량_prev'] + 1e-6)
        df_diff['amount_change'] = df_diff['사용량'] - df_diff['사용량_prev']
        
        drop_ratio_chunk = (df_diff['pct_change'] < -0.3).groupby(df_diff['BLDRGST_SEQNO']).mean().reset_index()
        drop_ratio_chunk.columns = ['BLDRGST_SEQNO', 'drop_ratio']
        all_drop_ratio.append(drop_ratio_chunk)
        
        sudden_drop_ratio_chunk = (df_diff['pct_change'] < -0.5).groupby(df_diff['BLDRGST_SEQNO']).mean().reset_index()
        sudden_drop_ratio_chunk.columns = ['BLDRGST_SEQNO', 'sudden_drop_ratio']
        all_sudden_drop_ratio.append(sudden_drop_ratio_chunk)
        
        max_drop = df_diff.groupby('BLDRGST_SEQNO')['amount_change'].min().reset_index()
        max_drop.columns = ['BLDRGST_SEQNO', 'max_drop_amount']
        all_max_drop.append(max_drop)
        
        df_sorted = df_long_chunk.sort_values(['BLDRGST_SEQNO', '연도_월']).copy()
        df_sorted['is_zero'] = (df_sorted['사용량'] == 0).astype(int)
        df_sorted['zero_group'] = (df_sorted['is_zero'] != df_sorted.groupby('BLDRGST_SEQNO')['is_zero'].shift(1)).astype(int)
        df_sorted['zero_group'] = df_sorted.groupby('BLDRGST_SEQNO')['zero_group'].cumsum()
        
        consecutive_zero = df_sorted[df_sorted['is_zero'] == 1].groupby(
            ['BLDRGST_SEQNO', 'zero_group']).size().groupby('BLDRGST_SEQNO').max().reset_index()
        consecutive_zero.columns = ['BLDRGST_SEQNO', 'consecutive_zero_max']
        all_consecutive_zero.append(consecutive_zero)
        
        unique_years_list = sorted(df_long_chunk['year'].unique())
        if len(unique_years_list) >= 2:
            yoy_changes = []
            for j in range(1, len(unique_years_list)):
                prev_year = unique_years_list[j-1]
                curr_year = unique_years_list[j]
                
                prev_data = df_long_chunk[df_long_chunk['year'] == prev_year].groupby('BLDRGST_SEQNO')['사용량'].mean().reset_index()
                prev_data.columns = ['BLDRGST_SEQNO', 'prev_mean']
                
                curr_data = df_long_chunk[df_long_chunk['year'] == curr_year].groupby('BLDRGST_SEQNO')['사용량'].mean().reset_index()
                curr_data.columns = ['BLDRGST_SEQNO', 'curr_mean']
                
                yoy = prev_data.merge(curr_data, on='BLDRGST_SEQNO', how='inner')
                yoy['yoy_change'] = (yoy['curr_mean'] - yoy['prev_mean']) / (yoy['prev_mean'] + 1e-6)
                yoy_changes.append(yoy[['BLDRGST_SEQNO', 'yoy_change']])
            
            if yoy_changes:
                yoy_all = pd.concat(yoy_changes, ignore_index=True)
                yoy_mean = yoy_all.groupby('BLDRGST_SEQNO')['yoy_change'].mean().reset_index()
                yoy_mean.columns = ['BLDRGST_SEQNO', 'yoy_change_mean']
                all_yoy_mean.append(yoy_mean)
        
        summer_data = df_long_chunk[df_long_chunk['month'].isin([6, 7, 8])].groupby('BLDRGST_SEQNO')['사용량'].mean().reset_index()
        summer_data.columns = ['BLDRGST_SEQNO', 'summer_mean']
        
        winter_data = df_long_chunk[df_long_chunk['month'].isin([12, 1, 2])].groupby('BLDRGST_SEQNO')['사용량'].mean().reset_index()
        winter_data.columns = ['BLDRGST_SEQNO', 'winter_mean']
        
        seasonal = summer_data.merge(winter_data, on='BLDRGST_SEQNO', how='outer').fillna(0)
        seasonal['seasonal_diff'] = seasonal['summer_mean'] - seasonal['winter_mean']
        all_seasonal.append(seasonal)
        
        cv_data = df_long_chunk.groupby('BLDRGST_SEQNO')['사용량'].agg(['mean', 'std']).reset_index()
        cv_data['cv'] = cv_data['std'] / (cv_data['mean'] + 1e-6)
        cv_data = cv_data[['BLDRGST_SEQNO', 'cv']]
        all_cv_data.append(cv_data)
        
        df_sorted2 = df_long_chunk.sort_values(['BLDRGST_SEQNO', '연도_월'])
        last_3 = df_sorted2.groupby('BLDRGST_SEQNO').tail(3)
        
        zero_count = (last_3['사용량'] == 0).groupby(last_3['BLDRGST_SEQNO']).sum()
        total_count = last_3.groupby('BLDRGST_SEQNO').size()
        
        recent_zero = pd.DataFrame({
            'BLDRGST_SEQNO': total_count.index,
            'recent_3month_zero': ((total_count == 3) & (zero_count == 3)).astype(float)
        }).reset_index(drop=True)
        
        all_recent_zero.append(recent_zero)
        
        del df_chunk, df_long_chunk, df_diff, df_sorted
        gc.collect()
        
        chunk_time = time.time() - chunk_start
        elapsed = time.time() - start_time
        eta = (chunk_time * (n_chunks - i - 1)) / 60
        
        if i % 10 == 0 or i == n_chunks - 1:
            print(f"  청크 {i+1}/{n_chunks} | 처리: {chunk_time:.1f}초 | 경과: {elapsed/60:.1f}분 | 남은 시간: {eta:.1f}분")
    
    df_final = df_wide
    
    df_final = df_final.merge(pd.concat(all_mean_kwh, ignore_index=True), on='BLDRGST_SEQNO', how='left')
    df_final = df_final.merge(pd.concat(all_zero_ratio, ignore_index=True), on='BLDRGST_SEQNO', how='left')
    df_final = df_final.merge(pd.concat(all_drop_ratio, ignore_index=True), on='BLDRGST_SEQNO', how='left')
    df_final = df_final.merge(pd.concat(all_sudden_drop_ratio, ignore_index=True), on='BLDRGST_SEQNO', how='left')
    df_final = df_final.merge(pd.concat(all_consecutive_zero, ignore_index=True), on='BLDRGST_SEQNO', how='left')
    df_final = df_final.merge(pd.concat(all_max_drop, ignore_index=True), on='BLDRGST_SEQNO', how='left')
    
    if all_yoy_mean:
        df_final = df_final.merge(pd.concat(all_yoy_mean, ignore_index=True), on='BLDRGST_SEQNO', how='left')
    else:
        df_final['yoy_change_mean'] = 0.0
    
    df_final = df_final.merge(pd.concat(all_seasonal, ignore_index=True), on='BLDRGST_SEQNO', how='left')
    df_final = df_final.merge(pd.concat(all_cv_data, ignore_index=True), on='BLDRGST_SEQNO', how='left')
    df_final = df_final.merge(pd.concat(all_recent_zero, ignore_index=True), on='BLDRGST_SEQNO', how='left')
    df_final = df_final.fillna(0)
    
    numeric_cols = df_final.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if col != 'BLDRGST_SEQNO':
            df_final[col] = df_final[col].astype('float32')
    
    del df_wide, all_mean_kwh, all_zero_ratio, all_drop_ratio, all_sudden_drop_ratio
    del all_consecutive_zero, all_max_drop, all_yoy_mean, all_seasonal, all_cv_data, all_recent_zero
    gc.collect()
    
    log_status("전기 데이터 통합 처리 완료")
    return df_final


def merge_main_energy(df_main, df_energy, logger=None):
    """메인 데이터 + 전기 데이터 병합"""
    print("\n" + "="*70)
    print("메인 + 전기 데이터 병합")
    print("="*70)
    
    if 'BLDRGST_SEQNO' in df_main.columns:
        df_main['BLDRGST_SEQNO'] = df_main['BLDRGST_SEQNO'].astype(str).str.strip()
    
    if 'BLDRGST_SEQNO' in df_energy.columns:
        df_energy['BLDRGST_SEQNO'] = df_energy['BLDRGST_SEQNO'].astype(str).str.strip()
    
    log_status("병합 시작")
    df_merged = df_main.merge(df_energy, on='BLDRGST_SEQNO', how='left')
    
    print(f"  병합 완료: {df_merged.shape}")
    print(f"  메모리 사용량: {df_merged.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    df_merged = normalize_oa_cd(df_merged, 'oa_cd')
    
    log_status("병합 완료")
    return df_merged


def load_external_oa_v1(external_oa_path, logger=None):
    """외부 집계구 v1 데이터 로드"""
    print("\n" + "="*70)
    print("외부 집계구 v1 데이터 로드")
    print("="*70)
    
    log_status(f"파일 경로: {external_oa_path}")
    
    df = pd.read_csv(
        external_oa_path,
        encoding='cp949',
        sep=',',
        dtype=str,
        on_bad_lines='skip'
    )
    
    print(f"  로드 완료: {df.shape}")
    print(f"  컬럼 수: {len(df.columns)}")
    
    if 'oa_cd' in df.columns:
        unique_oa = df['oa_cd'].nunique()
        print(f"  고유 oa_cd: {unique_oa:,}개")
    
    df = normalize_oa_cd(df, 'oa_cd')
    
    log_status("외부 집계구 v1 로드 완료")
    return df


def load_external_facility(facility_path, logger=None):
    """외부 시설 기초 외국인 데이터 로드"""
    print("\n" + "="*70)
    print("외부 시설 기초 외국인 데이터 로드")
    print("="*70)
    
    log_status(f"파일 경로: {facility_path}")
    
    df = pd.read_csv(
        facility_path,
        encoding='cp949',
        dtype=str,
        on_bad_lines='skip'
    )
    
    print(f"  로드 완료: {df.shape}")
    print(f"  컬럼 수: {len(df.columns)}")
    
    if 'oa_cd' in df.columns:
        unique_oa = df['oa_cd'].nunique()
        print(f"  고유 oa_cd: {unique_oa:,}개")
    
    df = normalize_oa_cd(df, 'oa_cd')
    
    log_status("외부 시설 기초 외국인 로드 완료")
    return df


def run_step1(connection, xy_mapper_path, external_oa_path, external_facility_path, 
              output_dir, main_table='ML_BIS_WEB_TOT_BLDG_MAIN',
              energy_table='ENGY_USE_YEAR_PT', distance_threshold=2.0, 
              remove_unmapped=True, pg_connection=None, run_id=None, timestamp=None, config=None, logger=None):
    """Step 1 실행: 데이터 통합"""
    print("\n" + "="*80)
    print("Step 1: 데이터 통합 시작")
    print("="*80)
    start_time = datetime.now()
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(output_dir, exist_ok=True)
    
    df_main, base_year = load_main_data(
        connection=connection,
        xy_mapper_path=xy_mapper_path,
        config=config,
        table_name=main_table,
        distance_threshold=distance_threshold,
        remove_unmapped=remove_unmapped
    )

    if logger and base_year:
        logger.update_base_year(base_year)
    
    target_seqnos = None
    if 'BLDRGST_SEQNO' in df_main.columns:
        target_seqnos = set(df_main['BLDRGST_SEQNO'].unique())
        log_status(f"target_seqnos: {len(target_seqnos):,}개")
    
    output_path_1 = os.path.join(output_dir, "01_main_data.parquet")
    df_main.to_parquet(output_path_1, index=False)
    log_status(f"저장: {output_path_1}")
    
    df_energy = process_energy_unified(connection, target_seqnos=target_seqnos, table_name=energy_table)
    
    output_path_2_parquet = os.path.join(output_dir, "02_energy_wide.parquet")
    output_path_2_csv = os.path.join(output_dir, "02_energy_wide.csv")
    
    df_energy.to_parquet(output_path_2_parquet, index=False)
    df_energy.to_csv(output_path_2_csv, index=False, encoding='cp949')
    
    log_status(f"저장: {output_path_2_parquet}")
    log_status(f"저장: {output_path_2_csv}")
    
    df_merged = merge_main_energy(df_main, df_energy)
    
    del df_main, df_energy
    gc.collect()
    
    df_merged, pu_model = run_pu_learning(
        df=df_merged,
        target_col='VACANT',
        model_type='lightgbm',
        target_vacancy_rate=0.007,
        output_dir=output_dir
    )
    
    output_path_3_parquet = os.path.join(output_dir, "03_merged_main_energy.parquet")
    output_path_3_csv = os.path.join(output_dir, "03_merged_main_energy_sample.csv")
    df_merged.to_parquet(output_path_3_parquet, index=False)
    df_merged.head(10000).to_csv(output_path_3_csv, index=False, encoding='cp949')
    log_status(f"저장: {output_path_3_parquet}")
    log_status(f"저장: {output_path_3_csv} (샘플 10000행)")
    
    df_external_oa = load_external_oa_v1(external_oa_path)
    
    output_path_4 = os.path.join(output_dir, "04_external_oa_v1.parquet")
    df_external_oa.to_parquet(output_path_4, index=False)
    log_status(f"저장: {output_path_4}")
    
    del df_external_oa
    gc.collect()
    
    #df_facility = load_external_facility(external_facility_path)
    df_facility = load_facility_from_db(
        connection=connection,
        xy_mapper_path=xy_mapper_path,
        config=config
    )
    
    output_path_5 = os.path.join(output_dir, "05_external_facility.parquet")
    df_facility.to_parquet(output_path_5, index=False)
    log_status(f"저장: {output_path_5}")
    
    del df_facility
    gc.collect()
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds() / 60
    
    print("\n" + "="*80)
    print(f"Step 1 완료! 소요 시간: {elapsed:.1f}분")
    print(f"저장 위치: {output_dir}")
    print("="*80)
    
    print("\n생성된 파일:")
    print("  1. 01_main_data.parquet")
    print("  2. 02_energy_wide.parquet / .csv")
    print("  3. 03_merged_main_energy.parquet / _sample.csv")
    print("  4. 04_external_oa_v1.parquet")
    print("  5. 05_external_facility.parquet")
    
    return df_merged


if __name__ == "__main__":
    import yaml
    import oracledb
    
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if config['oracle'].get('client_path'):
        oracledb.init_oracle_client(lib_dir=config['oracle']['client_path'])
    
    oracle_conn = oracledb.connect(
        user=config['oracle']['user'],
        password=config['oracle']['password'],
        host=config['oracle']['host'],
        port=config['oracle']['port'],
        service_name=config['oracle']['service_name'],
        tcp_connect_timeout=300,
        expire_time=0
    )
    oracle_conn.ping()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['paths']['base_path'], f"output_{timestamp}", "step1")
    
    try:
        run_step1(
            connection=oracle_conn,
            xy_mapper_path=config['paths']['xy_mapper_path'],
            external_oa_path=config['paths']['external_oa_path'],
            external_facility_path=config['paths']['external_facility_path'],
            output_dir=output_dir,
            distance_threshold=config['step1'].get('distance_threshold', 2.0),
            remove_unmapped=config['step1'].get('remove_unmapped', True),
            timestamp=timestamp
        )
    finally:
        oracle_conn.close()
        print("DB 연결 종료")
