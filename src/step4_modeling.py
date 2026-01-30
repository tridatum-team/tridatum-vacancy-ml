# -*- coding: utf-8 -*-
"""
Step 4: 모델링

처리 내용:
1. 클러스터별 데이터 분할
2. 4개 모델 학습 (XGBoost, LightGBM, CatBoost, GWR)
3. 2단계 모델 (분류 + 회귀)
4. 변수 중요도 저장
5. 모델 저장

출력:
- 13_two_stage_models.pkl
- 14_cluster_importance.csv
- 15_cluster_regression_metrics.csv
- 16_model_predictions.parquet
- 18_global_metrics.csv
"""
import os
import gc
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from .utils import log_status, normalize_oa_cd, to_db


def train_xgboost(X_train, y_train, X_test, y_test, logger=None):
    """XGBoost 학습"""
    print("\n  [XGBoost] 학습 중...")
    
    params = {
        'n_estimators': 300,
        'max_depth': 7,
        'learning_rate': 0.05,
        'random_state': 42,
        'n_jobs': -1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
    }
    
    print(f"    RMSE: {metrics['rmse']:.4f}")
    print(f"    MAE: {metrics['mae']:.4f}")
    print(f"    R2: {metrics['r2']:.4f}")
    
    return model, metrics


def train_lightgbm(X_train, y_train, X_test, y_test, logger=None):
    """LightGBM 학습"""
    print("\n  [LightGBM] 학습 중...")
    
    params = {
        'n_estimators': 300,
        'max_depth': 7,
        'learning_rate': 0.05,
        'random_state': 42,
        'n_jobs': -1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
    }
    
    print(f"    RMSE: {metrics['rmse']:.4f}")
    print(f"    MAE: {metrics['mae']:.4f}")
    print(f"    R2: {metrics['r2']:.4f}")
    
    return model, metrics


def train_catboost(X_train, y_train, X_test, y_test, use_class_weight=True, logger=None):
    """CatBoost 학습"""
    print("\n  [CatBoost] 학습 중...")
    
    params = {
        'iterations': 300,
        'depth': 7,
        'learning_rate': 0.05,
        'random_state': 42,
        'thread_count': -1,
        'verbose': False,
        'l2_leaf_reg': 3
    }
    
    model = CatBoostRegressor(**params)
    
    if use_class_weight and y_train is not None:
        class_weights = np.ones(len(y_train))
        positive_mask = y_train > 0
        if positive_mask.sum() > 0:
            ratio = (~positive_mask).sum() / (positive_mask.sum() + 1)
            class_weights[positive_mask] = min(ratio, 10.0)
    else:
        class_weights = None
    
    model.fit(X_train, y_train, sample_weight=class_weights)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
    }
    
    print(f"    RMSE: {metrics['rmse']:.4f}")
    print(f"    MAE: {metrics['mae']:.4f}")
    print(f"    R2: {metrics['r2']:.4f}")
    
    return model, metrics


def get_feature_cols(df, exclude_cols=None):
    """피처 컬럼 추출"""
    if exclude_cols is None:
        exclude_cols = [
            'oa_cd', 'cluster', 'VACANT', 'VACANT_RATIO',
            'VACANT_sum', 'VACANT_mean', 'building_count',
            'X', 'Y', 'VACANT_PROBA', 'HAS_VACANT', 'VACANT_RATIO_LOG'
        ]
    
    numeric_types = [np.int8, np.int16, np.int32, np.int64,
                     np.uint8, np.uint16, np.uint32,
                     np.float16, np.float32, np.float64]
    
    feature_cols = [col for col in df.columns
                    if col not in exclude_cols
                    and df[col].dtype in numeric_types]
    
    return feature_cols


def run_step4(input_dir_step2, input_dir_step3, output_dir, 
              top_n_features=30, use_two_stage=True, use_gwr=False,
              pg_connection=None, run_id=None, timestamp=None, logger=None):
    """Step 4 실행: 모델링"""
    print("\n" + "="*80)
    print("Step 4: 모델링 시작")
    print("="*80)
    start_time = datetime.now()
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n[Step 4-1] 데이터 로드")
    
    clustered_path = os.path.join(input_dir_step3, "11_clustered_data.parquet")
    df = pd.read_parquet(clustered_path)
    
    if 'oa_cd' in df.columns:
        df['oa_cd'] = df['oa_cd'].astype(str).str.strip()
    
    print(f"  클러스터 데이터: {df.shape}")
    print(f"  클러스터 수: {df['cluster'].nunique()}")
    
    if 'VACANT_RATIO' not in df.columns:
        full_path = os.path.join(input_dir_step2, "09_final_before_clustering.parquet")
        df_full = pd.read_parquet(full_path)
        
        if 'VACANT_RATIO' in df_full.columns:
            df_full['oa_cd'] = df_full['oa_cd'].astype(str).str.strip()
            df = df.merge(df_full[['oa_cd', 'VACANT_RATIO']], on='oa_cd', how='left')
        
        del df_full
        gc.collect()
    
    all_feature_cols = get_feature_cols(df)
    print(f"  전체 피처 수: {len(all_feature_cols)}")
    
    print("\n[Step 4-2] 피처 중요도 기반 선택")
    
    X_all = df[all_feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_all = df['VACANT_RATIO'].fillna(0)
    
    temp_model = lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, verbose=-1, n_jobs=-1)
    temp_model.fit(X_all, y_all)
    
    importance_df = pd.DataFrame({
        'feature': all_feature_cols,
        'importance': temp_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_features = importance_df.head(top_n_features)['feature'].tolist()
    print(f"  선택된 Top {top_n_features} 피처")
    
    importance_path = os.path.join(output_dir, "14_feature_importance.csv")
    importance_df.to_csv(importance_path, index=False, encoding='cp949')
    log_status(f"저장: {importance_path}")
    
    if use_two_stage:
        print("\n[Step 4-3] 2단계 모델링 (분류 + 회귀)")
        
        df['HAS_VACANT'] = (df['VACANT_RATIO'] > 0).astype(int)
        df['VACANT_RATIO_LOG'] = np.log1p(df['VACANT_RATIO'] * 100)
        
        X_clf = df[top_features].fillna(0).replace([np.inf, -np.inf], 0)
        y_clf = df['HAS_VACANT']
        
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
        )
        
        print("\n  === 1단계: 빈집 유무 분류 ===")
        
        clf_model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.03,
            random_state=42, n_jobs=-1, verbose=-1,
            scale_pos_weight=(y_train_c == 0).sum() / (y_train_c == 1).sum()
        )
        clf_model.fit(X_train_c, y_train_c)
        
        y_pred_proba = clf_model.predict_proba(X_test_c)[:, 1]
        from sklearn.metrics import roc_auc_score, accuracy_score
        auc_score = roc_auc_score(y_test_c, y_pred_proba)
        acc_score = accuracy_score(y_test_c, clf_model.predict(X_test_c))
        print(f"  분류 AUC: {auc_score:.4f}, Accuracy: {acc_score:.4f}")
        
        df['VACANT_PROBA'] = clf_model.predict_proba(X_clf)[:, 1]
        
        print("\n  === 2단계: 클러스터별 회귀 ===")
        
        df_has_vacant = df[df['VACANT_RATIO'] > 0].copy()
        print(f"  빈집 있는 집계구: {len(df_has_vacant):,}건")
        
        cluster_reg_models = {}
        cluster_metrics = []
        cluster_top_features = {}
        
        for cluster_id in sorted(df['cluster'].unique()):
            df_c = df_has_vacant[df_has_vacant['cluster'] == cluster_id].copy()
            
            if len(df_c) < 50:
                print(f"\n  Cluster {cluster_id}: 샘플 부족 ({len(df_c)}건) - 스킵")
                continue
            
            print(f"\n  Cluster {cluster_id}: {len(df_c):,}건")
            
            X_c = df_c[top_features].fillna(0).replace([np.inf, -np.inf], 0)
            y_c = df_c['VACANT_RATIO_LOG']
            
            temp_rf = lgb.LGBMRegressor(n_estimators=50, max_depth=5, random_state=42, verbose=-1, n_jobs=-1)
            temp_rf.fit(X_c, y_c)
            
            c_importance = pd.DataFrame({
                'feature': top_features,
                'importance': temp_rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            features_c = c_importance.head(min(20, len(top_features)))['feature'].tolist()
            cluster_top_features[cluster_id] = features_c
            
            X_c_sel = df_c[features_c].fillna(0).replace([np.inf, -np.inf], 0)
            
            X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
                X_c_sel, y_c, test_size=0.2, random_state=42
            )
            
            reg_model = lgb.LGBMRegressor(
                n_estimators=300, max_depth=8, learning_rate=0.03,
                random_state=42, n_jobs=-1, verbose=-1
            )
            reg_model.fit(X_train_r, y_train_r)
            
            y_pred_r = reg_model.predict(X_test_r)
            y_pred_orig = (np.expm1(y_pred_r) / 100).clip(0, 1)
            y_test_orig = (np.expm1(y_test_r) / 100).clip(0, 1)
            
            mae_r = mean_absolute_error(y_test_orig, y_pred_orig)
            rmse_r = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
            r2_r = r2_score(y_test_orig, y_pred_orig)
            
            print(f"    MAE: {mae_r:.4f}, RMSE: {rmse_r:.4f}, R2: {r2_r:.4f}")
            
            cluster_reg_models[cluster_id] = {
                'model': reg_model,
                'features': features_c
            }
            
            cluster_metrics.append({
                'cluster': cluster_id,
                'n_samples': len(df_c),
                'n_features': len(features_c),
                'mae': mae_r,
                'rmse': rmse_r,
                'r2': r2_r
            })
        
        print("\n  === 전역 회귀 모델 (폴백용) ===")
        
        X_global = df_has_vacant[top_features].fillna(0).replace([np.inf, -np.inf], 0)
        y_global = df_has_vacant['VACANT_RATIO_LOG']
        
        X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
            X_global, y_global, test_size=0.2, random_state=42
        )
        
        global_reg_model = lgb.LGBMRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.03,
            random_state=42, n_jobs=-1, verbose=-1
        )
        global_reg_model.fit(X_train_g, y_train_g)
        
        y_pred_g_log = global_reg_model.predict(X_test_g)
        y_pred_g = (np.expm1(y_pred_g_log) / 100).clip(0, 1)
        y_test_g_orig = (np.expm1(y_test_g) / 100).clip(0, 1)
        
        r2_global = r2_score(y_test_g_orig, y_pred_g)
        print(f"  전역 모델 R2: {r2_global:.4f}")
        
        print("\n[Step 4-4] 전체 데이터 예측")
        
        df['VACANT_RATIO_P'] = 0.0
        
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_mask = df['cluster'] == cluster_id
            
            if cluster_id in cluster_reg_models:
                model_info = cluster_reg_models[cluster_id]
                reg_model = model_info['model']
                features = model_info['features']
                
                X_pred = df.loc[cluster_mask, features].fillna(0).replace([np.inf, -np.inf], 0)
                pred_log = reg_model.predict(X_pred)
                pred_ratio = (np.expm1(pred_log) / 100).clip(0, 1)
            else:
                X_pred = df.loc[cluster_mask, top_features].fillna(0).replace([np.inf, -np.inf], 0)
                pred_log = global_reg_model.predict(X_pred)
                pred_ratio = (np.expm1(pred_log) / 100).clip(0, 1)
            
            df.loc[cluster_mask, 'VACANT_RATIO_P'] = (
                df.loc[cluster_mask, 'VACANT_PROBA'] * pred_ratio
            )
            
            print(f"  Cluster {cluster_id}: {cluster_mask.sum():,}건 예측 완료")
        
        mae_final = mean_absolute_error(df['VACANT_RATIO'], df['VACANT_RATIO_P'])
        rmse_final = np.sqrt(mean_squared_error(df['VACANT_RATIO'], df['VACANT_RATIO_P']))
        r2_final = r2_score(df['VACANT_RATIO'], df['VACANT_RATIO_P'])
        
        print(f"\n  2단계 모델 최종 성능:")
        print(f"    MAE: {mae_final:.4f}")
        print(f"    RMSE: {rmse_final:.4f}")
        print(f"    R2: {r2_final:.4f}")
        
        models = {
            'classifier': clf_model,
            'cluster_regressors': cluster_reg_models,
            'global_regressor': global_reg_model,
            'feature_cols': top_features,
            'cluster_top_features': cluster_top_features
        }
        
        model_path = os.path.join(output_dir, '13_two_stage_models.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(models, f)
        print(f"\n  모델 저장: {model_path}")
        
        if cluster_metrics:
            metrics_df = pd.DataFrame(cluster_metrics)
            metrics_path = os.path.join(output_dir, '15_cluster_regression_metrics.csv')
            metrics_df.to_csv(metrics_path, index=False, encoding='cp949')
            print(f"  메트릭 저장: {metrics_path}")
    
    print("\n[Step 4-5] 기존 방식 모델 학습")
    
    X_all = df[top_features].fillna(0).replace([np.inf, -np.inf], 0)
    y_all = df['VACANT_RATIO']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42,
        stratify=df['cluster']
    )
    
    xgb_model, _ = train_xgboost(X_train, y_train, X_test, y_test)
    df['VACANT_RATIO_P_XGB'] = xgb_model.predict(X_all).clip(0, 1)
    
    y_pred_xgb = xgb_model.predict(X_test).clip(0, 1)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    print(f"  XGBoost R2: {r2_xgb:.4f}")
    
    lgb_model, _ = train_lightgbm(X_train, y_train, X_test, y_test)
    df['VACANT_RATIO_P_LGB'] = lgb_model.predict(X_all).clip(0, 1)
    
    y_pred_lgb = lgb_model.predict(X_test).clip(0, 1)
    r2_lgb = r2_score(y_test, y_pred_lgb)
    print(f"  LightGBM R2: {r2_lgb:.4f}")
    
    cat_model, _ = train_catboost(X_train, y_train, X_test, y_test)
    df['VACANT_RATIO_P_CAT'] = cat_model.predict(X_all).clip(0, 1)
    
    y_pred_cat = cat_model.predict(X_test).clip(0, 1)
    r2_cat = r2_score(y_test, y_pred_cat)
    print(f"  CatBoost R2: {r2_cat:.4f}")
    
    print("\n[Step 4-6] 결과 저장")
    
    print("\n  === 모델 성능 비교 ===")
    print(f"  {'모델':<25} {'R2':>10}")
    print(f"  {'-'*35}")
    print(f"  {'XGBoost':<25} {r2_xgb:>10.4f}")
    print(f"  {'LightGBM':<25} {r2_lgb:>10.4f}")
    print(f"  {'CatBoost':<25} {r2_cat:>10.4f}")
    if use_two_stage:
        print(f"  {'2단계 모델':<25} {r2_final:>10.4f}")
    
    df = normalize_oa_cd(df, 'oa_cd')
    
    numeric_cols = df.select_dtypes(include=['float32', 'float64']).columns
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(0)
        df[col] = df[col].astype('float32')
    
    output_path = os.path.join(output_dir, '16_model_predictions.parquet')
    df.to_parquet(output_path, index=False)
    print(f"\n  Parquet 저장: {output_path}")
    
    output_csv = os.path.join(output_dir, '16_model_predictions.csv')
    df.to_csv(output_csv, index=False, encoding='cp949')
    print(f"  CSV 저장: {output_csv}")
    
    if pg_connection is not None and timestamp is not None:
        df['run_id'] = run_id
        to_db(df, f'tb_pipeline_prediction_result', pg_connection,
              db_type='postgresql', force_text_cols=['oa_cd'],
              varchar_lengths={'oa_cd': 20}, primary_key=['oa_cd','run_id'])
    
    global_metrics = [
        {'model': 'XGBoost', 'r2': r2_xgb},
        {'model': 'LightGBM', 'r2': r2_lgb},
        {'model': 'CatBoost', 'r2': r2_cat}
    ]
    if use_two_stage:
        global_metrics.append({'model': 'Two-Stage', 'r2': r2_final})
    
    global_metrics_df = pd.DataFrame(global_metrics)
    global_metrics_path = os.path.join(output_dir, '18_global_metrics.csv')
    global_metrics_df.to_csv(global_metrics_path, index=False, encoding='cp949')
    
    meta = {
        'feature_cols': top_features,
        'timestamp': timestamp
    }
    meta_path = os.path.join(output_dir, '18_model_meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds() / 60
    
    print("\n" + "="*80)
    print(f"Step 4 완료! 소요 시간: {elapsed:.1f}분")
    print("="*80)
    
    return df


if __name__ == "__main__":
    import yaml
    
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    base_path = config['paths']['base_path']
    output_dirs = [d for d in os.listdir(base_path) if d.startswith('output_')]
    latest_output = sorted(output_dirs)[-1]
    
    input_dir_step2 = os.path.join(base_path, latest_output, "step2")
    input_dir_step3 = os.path.join(base_path, latest_output, "step3")
    output_dir = os.path.join(base_path, latest_output, "step4")
    
    run_step4(
        input_dir_step2=input_dir_step2,
        input_dir_step3=input_dir_step3,
        output_dir=output_dir,
        top_n_features=30,
        use_two_stage=True,
        use_gwr=False
    )
