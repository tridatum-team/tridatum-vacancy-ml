# -*- coding: utf-8 -*-
"""
Step 3: 클러스터링

처리 내용:
1. 클러스터링용 데이터 준비
2. 최적 K 탐색 (Elbow, Silhouette)
3. KMeans 클러스터링 실행
4. 클러스터 프로파일 분석
5. 클러스터별 변수 중요도 추출

출력:
- 10_ready_for_clustering.parquet
- 11_clustered_data.parquet
- 12_cluster_profiles.csv
- clustering_metrics.png
"""
import os
import gc
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

from .utils import log_status, normalize_oa_cd, to_db


def prepare_clustering_data(df, features=None, logger=None):
    """클러스터링용 데이터 준비"""
    print("\n" + "="*70)
    print("클러스터링 데이터 준비")
    print("="*70)
    
    protected_cols = {
        'oa_cd', 'VACANT', 'VACANT_RATIO', 'VACANT_sum', 'VACANT_mean',
        'building_count', 'X', 'Y'
    }
    
    if features:
        use_cols = ['oa_cd'] + features
        use_cols = [c for c in use_cols if c in df.columns]
        df_num = df[use_cols].select_dtypes(include=[np.number]).copy()
    else:
        df_num = df.select_dtypes(include=[np.number]).copy()
    
    if 'oa_cd' in df_num.columns:
        df_num = df_num.drop(columns=['oa_cd'])
    
    print(f"  원본 숫자형 변수: {len(df_num.columns)}개")
    
    var = df_num.var()
    const_cols = var[var < 1e-8].index.tolist()
    
    zero_ratio = (df_num == 0).mean()
    high_zero_cols = zero_ratio[zero_ratio >= 0.80].index.tolist()
    
    dup_cols = df_num.T.duplicated()
    dup_cols = df_num.columns[dup_cols].tolist()
    
    const_cols = [c for c in const_cols if c not in protected_cols]
    high_zero_cols = [c for c in high_zero_cols if c not in protected_cols]
    dup_cols = [c for c in dup_cols if c not in protected_cols]
    
    drop_cols = set(const_cols) | set(high_zero_cols) | set(dup_cols)
    
    feature_cols = [c for c in df_num.columns if c not in drop_cols]
    df_clean = df_num[feature_cols].copy()
    
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    nan_ratio = df_clean.isna().mean()
    drop_na_cols = nan_ratio[nan_ratio > 0.7].index.tolist()
    drop_na_cols = [c for c in drop_na_cols if c not in protected_cols]
    
    if drop_na_cols:
        df_clean = df_clean.drop(columns=drop_na_cols)
    
    df_clean = df_clean.fillna(df_clean.median())
    
    print(f"  정제 후 변수: {len(df_clean.columns)}개")
    print(f"    삭제: 상수 {len(const_cols)}, 고비율0 {len(high_zero_cols)}, 중복 {len(dup_cols)}, 결측 {len(drop_na_cols)}")
    
    if 'oa_cd' in df.columns:
        df_clean.insert(0, 'oa_cd', df['oa_cd'].values)
    
    log_status("클러스터링 데이터 준비 완료")
    
    return df_clean


def find_elbow_point(k_range, values, logger=None):
    """엘보 포인트 계산"""
    if len(k_range) < 3:
        return k_range[0]
    
    coords = np.array([[k, v] for k, v in zip(k_range, values)])
    
    line_vec = coords[-1] - coords[0]
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    
    vec_from_first = coords - coords[0]
    
    scalar_proj = np.dot(vec_from_first, line_vec_norm)
    vec_proj = np.outer(scalar_proj, line_vec_norm)
    
    vec_to_line = vec_from_first - vec_proj
    dist_to_line = np.linalg.norm(vec_to_line, axis=1)
    
    elbow_idx = np.argmax(dist_to_line)
    
    return k_range[elbow_idx]


def run_kmeans_explore(df_ready, k_range=None, output_dir=None, logger=None):
    """최적 K 탐색 (Elbow, Silhouette)"""
    print("\n" + "="*70)
    print("KMeans 최적 K 탐색")
    print("="*70)
    
    if k_range is None:
        k_range = range(2, 11)
    
    df_num = df_ready.select_dtypes(include=[np.number]).copy()
    if 'oa_cd' in df_num.columns:
        df_num = df_num.drop(columns=['oa_cd'])
    
    X_scaled = StandardScaler().fit_transform(df_num)
    
    print(f"  데이터: {X_scaled.shape}")
    print(f"  K 범위: {min(k_range)} ~ {max(k_range)}")
    
    silhouette_scores = []
    calinski_scores = []
    inertias = []
    
    for k in k_range:
        print(f"  K={k} 평가 중...", end=' ')
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        sil = silhouette_score(X_scaled, labels)
        cal = calinski_harabasz_score(X_scaled, labels)
        inertia = kmeans.inertia_
        
        silhouette_scores.append(sil)
        calinski_scores.append(cal)
        inertias.append(inertia)
        
        print(f"Silhouette: {sil:.3f}, Calinski: {cal:.1f}")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(k_range, silhouette_scores, 'o-', color='blue', linewidth=2, markersize=8)
    axes[0].set_xlabel('K', fontsize=12)
    axes[0].set_ylabel('Silhouette Score', fontsize=12)
    axes[0].set_title('Silhouette Score by K', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(k_range, calinski_scores, 'o-', color='green', linewidth=2, markersize=8)
    axes[1].set_xlabel('K', fontsize=12)
    axes[1].set_ylabel('Calinski-Harabasz Score', fontsize=12)
    axes[1].set_title('Calinski-Harabasz Score by K', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(k_range, inertias, 'o-', color='red', linewidth=2, markersize=8)
    axes[2].set_xlabel('K', fontsize=12)
    axes[2].set_ylabel('Inertia', fontsize=12)
    axes[2].set_title('Elbow Method (Inertia)', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    elbow_k = find_elbow_point(list(k_range), inertias)
    axes[2].axvline(x=elbow_k, color='orange', linestyle='--', linewidth=2, label=f'Elbow: K={elbow_k}')
    axes[2].legend()
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'clustering_metrics.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        log_status(f"저장: {plot_path}")
    
    plt.close()
    
    best_k_sil = k_range[np.nanargmax(silhouette_scores)]
    print(f"\n추천 K (Silhouette 기준): {best_k_sil}")
    print(f"추천 K (Elbow 기준): {elbow_k}")
    
    return best_k_sil, {
        'silhouette': silhouette_scores,
        'calinski': calinski_scores,
        'inertia': inertias,
        'elbow_k': elbow_k
    }


def run_kmeans_clustering(df_ready, k, logger=None):
    """KMeans 클러스터링 실행"""
    print("\n" + "="*70)
    print(f"KMeans 클러스터링 (K={k})")
    print("="*70)
    
    df_num = df_ready.select_dtypes(include=[np.number]).copy()
    if 'oa_cd' in df_num.columns:
        df_num = df_num.drop(columns=['oa_cd'])
    
    X_scaled = StandardScaler().fit_transform(df_num)
    
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    df_result = df_ready.copy()
    df_result['cluster'] = labels
    
    print(f"\n클러스터 분포:")
    cluster_counts = df_result['cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        pct = count / len(df_result) * 100
        print(f"  Cluster {cluster_id}: {count:,}건 ({pct:.1f}%)")
    
    sil = silhouette_score(X_scaled, labels)
    cal = calinski_harabasz_score(X_scaled, labels)
    
    print(f"\n클러스터링 품질:")
    print(f"  Silhouette Score: {sil:.4f}")
    print(f"  Calinski-Harabasz Score: {cal:.1f}")
    
    return df_result


def analyze_cluster_profiles(df_clustered, output_dir=None, logger=None):
    """클러스터 프로파일 분석"""
    print("\n" + "="*70)
    print("클러스터 프로파일 분석")
    print("="*70)
    
    df_num = df_clustered.select_dtypes(include=[np.number])
    exclude_cols = ['cluster', 'oa_cd']
    feature_cols = [c for c in df_num.columns if c not in exclude_cols]
    
    profiles = df_clustered.groupby('cluster')[feature_cols].mean()
    
    global_mean = df_clustered[feature_cols].mean()
    
    profiles_norm = profiles.divide(global_mean + 1e-6)
    
    print("\n클러스터별 특성 (전체 평균 대비):")
    for cluster_id in profiles.index:
        print(f"\n  Cluster {cluster_id}:")
        
        cluster_profile = profiles_norm.loc[cluster_id].sort_values(ascending=False)
        
        print("    높은 특성 (Top 5):")
        for var, val in cluster_profile.head(5).items():
            if val > 1.0:
                print(f"      {var}: {val:.2f}x")
        
        print("    낮은 특성 (Bottom 5):")
        for var, val in cluster_profile.tail(5).items():
            if val < 1.0:
                print(f"      {var}: {val:.2f}x")
    
    if output_dir:
        profile_path = os.path.join(output_dir, "12_cluster_profiles.csv")
        profiles.to_csv(profile_path, encoding='cp949')
        log_status(f"저장: {profile_path}")
    
    return profiles


def extract_cluster_feature_importance(df_clustered, cluster_col='cluster', n_top=20, output_dir=None, logger=None):
    """클러스터별 변수 중요도 추출"""
    print("\n" + "="*70)
    print("클러스터별 변수 중요도 추출")
    print("="*70)
    
    exclude_cols = [cluster_col, 'oa_cd', 'VACANT', 'VACANT_RATIO', 'VACANT_sum', 
                    'VACANT_mean', 'building_count', 'X', 'Y']
    
    feature_cols = [c for c in df_clustered.columns 
                   if c not in exclude_cols 
                   and df_clustered[c].dtype in [np.float32, np.float64, np.int32, np.int64]]
    
    if len(feature_cols) == 0:
        print("  변수 없음")
        return {}
    
    X = df_clustered[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = df_clustered[cluster_col]
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n  전체 TOP {n_top} 중요 변수:")
    for idx, row in importance_df.head(n_top).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    if output_dir:
        csv_path = os.path.join(output_dir, "cluster_feature_importance.csv")
        importance_df.to_csv(csv_path, index=False, encoding='cp949')
        log_status(f"저장: {csv_path}")
    
    importance_results = {'global': importance_df}
    
    for cluster_id in sorted(df_clustered[cluster_col].unique()):
        cluster_mask = df_clustered[cluster_col] == cluster_id
        
        if cluster_mask.sum() < 50:
            continue
        
        y_binary = cluster_mask.astype(int)
        
        rf_cluster = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        rf_cluster.fit(X, y_binary)
        
        cluster_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_cluster.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_results[cluster_id] = cluster_importance
        
        print(f"\n  Cluster {cluster_id} TOP {n_top} 변수:")
        for idx, row in cluster_importance.head(n_top).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
        
        if output_dir:
            csv_path = os.path.join(output_dir, f"cluster_{cluster_id}_feature_importance.csv")
            cluster_importance.to_csv(csv_path, index=False, encoding='cp949')
    
    return importance_results


def run_step3(input_dir, output_dir, k_range=None, k_final=None, 
              pg_connection=None, run_id=None, timestamp=None, logger=None):
    """Step 3 실행: 클러스터링"""
    print("\n" + "="*80)
    print("Step 3: 클러스터링 시작")
    print("="*80)
    start_time = datetime.now()
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n[Step 3-1] 데이터 로드")
    log_status("파일 로드: 09_final_before_clustering.parquet")
    df = pd.read_parquet(os.path.join(input_dir, "09_final_before_clustering.parquet"))
    
    if 'oa_cd' in df.columns:
        df['oa_cd'] = df['oa_cd'].astype(str).str.strip()
    elif '집계구코드' in df.columns:
        df = df.rename(columns={'집계구코드': 'oa_cd'})
        df['oa_cd'] = df['oa_cd'].astype(str).str.strip()
    
    print(f"  로드 완료: {df.shape}")
    
    clustering_features = [
        '건물구조_불량', '건물구조_양호', '평균건물나이', '노후건물비율',
        '아파트비율', '평균토지면적', '위반건축물비율',
        '최신공시지가', '공시지가_3년변동률', '공시지가_하락비율',
        '공시지가_변동성', '저가토지_비율',
        '전기_평균사용량', '전기_제로비율', '전기_감소비율', '전기_급감비율',
        'VACANT_sum', 'VACANT_mean', 'building_count', 'VACANT_RATIO',
        'X', 'Y',
        '총인구', '남성인구', '여성인구', '평균나이', '인구밀도',
        '노령화지수', '유소년부양비', '노년부양비',
        '총가구수', '평균가구원수', '총주택수', '총사업체수', '총종사자수'
    ]
    
    available_features = [f for f in clustering_features if f in df.columns]
    print(f"  사용 가능 피처: {len(available_features)}개")
    
    print("\n[Step 3-2] 클러스터링용 데이터 준비")
    df_ready = prepare_clustering_data(df, features=available_features if available_features else None)
    
    df_ready = normalize_oa_cd(df_ready, 'oa_cd')
    
    output_path_10_parquet = os.path.join(output_dir, "10_ready_for_clustering.parquet")
    output_path_10_csv = os.path.join(output_dir, "10_ready_for_clustering.csv")
    df_ready.to_parquet(output_path_10_parquet, index=False)
    df_ready.to_csv(output_path_10_csv, index=False, encoding='cp949')
    log_status(f"저장: {output_path_10_parquet}")
    
    del df
    gc.collect()
    
    print("\n[Step 3-3] 최적 K 탐색")
    if k_range is None:
        k_range = range(2, 11)
    
    best_k_sil, metrics = run_kmeans_explore(df_ready, k_range=k_range, output_dir=output_dir)
    
    if k_final is None:
        k_final = best_k_sil
        print(f"\n자동 선택된 K: {k_final}")
    else:
        print(f"\n지정된 K: {k_final}")
    
    print("\n[Step 3-4] KMeans 클러스터링")
    df_clustered = run_kmeans_clustering(df_ready, k=k_final)
    
    importance_results = extract_cluster_feature_importance(
        df_clustered, cluster_col='cluster', n_top=20, output_dir=output_dir
    )
    
    numeric_cols = df_clustered.select_dtypes(include=['float32', 'float64']).columns
    for col in numeric_cols:
        df_clustered[col] = df_clustered[col].replace([np.inf, -np.inf], np.nan)
        df_clustered[col] = df_clustered[col].fillna(0)
    
    df_clustered = normalize_oa_cd(df_clustered, 'oa_cd')
    
    if pg_connection is not None and timestamp is not None:
        df_clustered['run_id'] = run_id
        to_db(df_clustered, f'tb_pipeline_cluster', pg_connection, 
              db_type='postgresql', force_text_cols=['oa_cd'], if_exists='append',
              varchar_lengths={'oa_cd': 20}, primary_key=['oa_cd','run_id'])
    
    output_path_11_parquet = os.path.join(output_dir, "11_clustered_data.parquet")
    output_path_11_csv = os.path.join(output_dir, "11_clustered_data.csv")
    df_clustered.to_parquet(output_path_11_parquet, index=False)
    df_clustered.to_csv(output_path_11_csv, index=False, encoding='cp949')
    log_status(f"저장: {output_path_11_parquet}")
    log_status(f"저장: {output_path_11_csv}")
    
    print("\n[Step 3-5] 클러스터 프로파일 분석")
    df_profiles = analyze_cluster_profiles(df_clustered, output_dir=output_dir)
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds() / 60
    
    print("\n" + "="*80)
    print(f"Step 3 완료! 소요 시간: {elapsed:.1f}분")
    print(f"최종 클러스터 수: {k_final}")
    print(f"저장 위치: {output_dir}")
    print("="*80)
    
    return df_clustered, df_profiles


if __name__ == "__main__":
    import yaml
    
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    base_path = config['paths']['base_path']
    output_dirs = [d for d in os.listdir(base_path) if d.startswith('output_')]
    latest_output = sorted(output_dirs)[-1]
    
    input_dir = os.path.join(base_path, latest_output, "step2")
    output_dir = os.path.join(base_path, latest_output, "step3")
    
    run_step3(input_dir=input_dir, output_dir=output_dir, k_range=range(2, 11), k_final=None)
