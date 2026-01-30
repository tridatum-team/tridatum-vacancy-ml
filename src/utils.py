# -*- coding: utf-8 -*-
"""
공통 유틸리티 함수
"""
import os
import gc
import pickle
import logging
from datetime import datetime
from multiprocessing import cpu_count

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

N_CORES = cpu_count()


def setup_logger(name, log_file=None, level=logging.INFO):
    """로거 설정"""
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger


def log_status(message, logger=None):
    """상태 로깅"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    msg = f"[{timestamp}] {message}"
    if logger:
        logger.info(message)
    else:
        print(msg)


def get_memory_usage():
    """현재 메모리 사용량 반환 (GB)"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**3
    except ImportError:
        return 0


def log_memory(logger=None):
    """메모리 사용량 로깅"""
    mem_gb = get_memory_usage()
    if mem_gb > 0:
        log_status(f"메모리 사용량: {mem_gb:.2f} GB", logger)


def optimize_dtypes(df, float_dtype='float32', int_dtype='int32'):
    """데이터 타입 최적화"""
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type == 'float64':
            df[col] = df[col].astype(float_dtype)
        elif col_type == 'int64':
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= 0:
                if col_max < 256:
                    df[col] = df[col].astype('uint8')
                elif col_max < 65536:
                    df[col] = df[col].astype('uint16')
                else:
                    df[col] = df[col].astype(int_dtype)
            else:
                if col_min > -128 and col_max < 128:
                    df[col] = df[col].astype('int8')
                elif col_min > -32768 and col_max < 32768:
                    df[col] = df[col].astype('int16')
                else:
                    df[col] = df[col].astype(int_dtype)
    
    return df


def normalize_oa_cd(df, oa_col='oa_cd'):
    """집계구 코드 정규화"""
    if oa_col in df.columns:
        df[oa_col] = df[oa_col].astype(str).str.strip()
        df[oa_col] = df[oa_col].str.replace(r'\.0$', '', regex=True)
    return df


def normalize_key(df, key_col):
    """키 컬럼 정규화"""
    if key_col in df.columns:
        df[key_col] = df[key_col].astype(str).str.strip()
        df[key_col] = df[key_col].str.replace(r'\.0$', '', regex=True)
    return df


def norm_str(series):
    """문자열 정규화"""
    result = series.astype(str).str.strip()
    result = result.str.replace(r'\.0$', '', regex=True)
    result = result.str.replace('nan', '').str.replace('None', '').str.replace('NaN', '')
    return result


def safe_join(df_left, df_right, on, how='left'):
    """안전한 조인 (중복 체크)"""
    if df_right.empty or on not in df_right.columns:
        return df_left
    
    if df_right[on].duplicated().any():
        df_right = df_right.drop_duplicates(subset=[on], keep='first')
    
    if on in df_left.columns and on in df_right.columns:
        df_left[on] = df_left[on].astype(str).str.strip()
        df_right[on] = df_right[on].astype(str).str.strip()
    
    return df_left.merge(df_right, on=on, how=how)


def fix_merged_dtypes(df):
    """병합 후 데이터 타입 정리"""
    object_cols = [
        'BLDRGST_SEQNO', 'PNU', 'PNU_CD',
        '시도', '시군구', '읍면동',
        'SGG_CD', 'EMD_CD', 'AMD_CD', 'SD_CD',
        'HP_BLDG_BDATE', 'STRCT_CD_NM', 'USEAPR_DATE',
        'VACANT_PNU', 'NEW_JDG_DT', 'BIS_TYPE', 'APT_TYPE',
        'UGRND_FLR_CNT', 'VACANT_X', 'VACANT_Y',
        'BIS_TYPE_NEW', 'TBOOK_SOYUH', 'TBOOK_SOYUH_NM',
        'GIMOK_NM', 'oa_cd'
    ]
    
    existing_object_cols = [col for col in object_cols if col in df.columns]
    
    if 'oa_cd' in df.columns:
        df['oa_cd'] = df['oa_cd'].astype(str).str.strip()
    
    if 'BLDRGST_SEQNO' in df.columns:
        df['BLDRGST_SEQNO'] = df['BLDRGST_SEQNO'].astype(str).str.strip()
    
    for col in existing_object_cols:
        if col not in ['oa_cd', 'BLDRGST_SEQNO']:
            df[col] = df[col].astype(str).str.strip()
    
    numeric_candidate_cols = [col for col in df.columns if col not in existing_object_cols]
    
    for col in numeric_candidate_cols:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    
    numeric_cols = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df


def clean_numeric_columns(df, threshold=1e9):
    """숫자 컬럼 정리 (inf, NaN 처리)"""
    numeric_cols = df.select_dtypes(include=['float32', 'float64']).columns
    
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(0)
        
        max_val = df[col].abs().max()
        if max_val > threshold:
            df[col] = df[col].clip(-threshold, threshold)
        
        df[col] = df[col].astype('float32')
    
    return df


def save_checkpoint(data, path, logger=None):
    """체크포인트 저장"""
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    if path.endswith('.parquet'):
        data.to_parquet(path, index=False)
    elif path.endswith('.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    elif path.endswith('.csv'):
        data.to_csv(path, index=False, encoding='utf-8-sig')
    
    log_status(f"저장: {path}", logger)


def load_checkpoint(path, logger=None):
    """체크포인트 로드"""
    if not os.path.exists(path):
        return None
    
    log_status(f"로드: {path}", logger)
    
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    elif path.endswith('.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)
    elif path.endswith('.csv'):
        return pd.read_csv(path, encoding='utf-8-sig')
    
    return None


def to_db(df, table_name, connection, db_type='oracle', 
          if_exists='replace', chunksize=10000, 
          force_text_cols=None, primary_key=None, varchar_lengths=None):
    """DataFrame을 DB 테이블로 저장"""
    if connection is None:
        print(f"  [to_db] 연결 없음, 저장 스킵: {table_name}")
        return
    
    cursor = connection.cursor()
    df_clean = df.copy()
    
    if db_type == 'postgresql':
        df_clean.columns = df_clean.columns.str.lower()
        if force_text_cols:
            force_text_cols = [col.lower() for col in force_text_cols]
        if primary_key:
            if isinstance(primary_key, str):
                primary_key = primary_key.lower()
            else:
                primary_key = [col.lower() for col in primary_key]
        if varchar_lengths:
            varchar_lengths = {k.lower(): v for k, v in varchar_lengths.items()}
    
    if force_text_cols:
        for col in force_text_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip()
                df_clean[col] = df_clean[col].replace({'nan': None, '<NA>': None, 'None': None})
    
    if if_exists == 'replace':
        try:
            if db_type == 'oracle':
                drop_sql = f"DROP TABLE {table_name} PURGE"
            else:
                drop_sql = f"DROP TABLE IF EXISTS {table_name}"
            cursor.execute(drop_sql)
            connection.commit()
        except:
            connection.rollback()
        
        cols = []

        for col, dtype in df_clean.dtypes.items():
            dtype_str = str(dtype)
            
            if col in (force_text_cols or []):
                length = varchar_lengths.get(col, 100) if varchar_lengths else 100
                if db_type == 'oracle':
                    cols.append(f'"{col}" VARCHAR2({length})')
                else:
                    cols.append(f'"{col}" VARCHAR({length})')
            elif 'int' in dtype_str:
                if db_type == 'oracle':
                    cols.append(f'"{col}" NUMBER(15)')
                else:
                    cols.append(f'"{col}" BIGINT')
            elif 'float' in dtype_str:
                if db_type == 'oracle':
                    cols.append(f'"{col}" NUMBER(20,6)')
                else:
                    cols.append(f'"{col}" DOUBLE PRECISION')
            elif 'datetime' in dtype_str:
                if db_type == 'oracle':
                    cols.append(f'"{col}" TIMESTAMP')
                else:
                    cols.append(f'"{col}" TIMESTAMP')
            else:
                if db_type == 'oracle':
                    cols.append(f'"{col}" VARCHAR2(500)')
                else:
                    cols.append(f'"{col}" TEXT')
        
        if primary_key:
            if isinstance(primary_key, str):
                pk_cols = f'"{primary_key}"'
            else:
                pk_cols = ', '.join([f'"{pk}"' for pk in primary_key])
            cols.append(f'PRIMARY KEY ({pk_cols})')
        
        create_sql = f'CREATE TABLE {table_name} ({", ".join(cols)})'
        
        try:
            cursor.execute(create_sql)
            connection.commit()
            print(f"  [to_db] 테이블 생성: {table_name}")
        except Exception as e:
            print(f"  [to_db] 테이블 생성 실패: {e}")
            connection.rollback()
            return
    
    col_names = ', '.join([f'"{col}"' for col in df_clean.columns])
    
    if db_type == 'oracle':
        placeholders = ', '.join([f':{i+1}' for i in range(len(df_clean.columns))])
    else:
        placeholders = ', '.join(['%s'] * len(df_clean.columns))
    
    insert_sql = f'INSERT INTO {table_name} ({col_names}) VALUES ({placeholders})'
    
    total_rows = len(df_clean)
    inserted = 0
    
    for start in range(0, total_rows, chunksize):
        end = min(start + chunksize, total_rows)
        chunk = df_clean.iloc[start:end]
        
        data = []
        for _, row in chunk.iterrows():
            row_data = []
            for val in row:
                if pd.isna(val):
                    row_data.append(None)
                elif isinstance(val, (np.integer, np.floating)):
                    row_data.append(float(val) if isinstance(val, np.floating) else int(val))
                else:
                    row_data.append(val)
            data.append(tuple(row_data))
        
        try:
            cursor.executemany(insert_sql, data)
            connection.commit()
            inserted += len(data)
        except Exception as e:
            print(f"  [to_db] 삽입 오류 (청크 {start}-{end}): {e}")
            connection.rollback()
    
    cursor.close()
    print(f"  [to_db] 완료: {table_name} ({inserted:,}/{total_rows:,}건)")

# 집계구 shp db에서 가져오는데 컬럼 확인 해봐야됨 // [확인사항]
def load_oa_boundary(config, xy_mapper_path=None):

    if config is None:
        gdf = gpd.read_file(xy_mapper_path, encoding='utf-8')
        gdf['TOT_REG_CD'] = norm_str(gdf['TOT_REG_CD'])
        return gdf, None

    oa_config = config.get('oa_boundary', {})
    source = oa_config.get('source', 'file')
    base_year = None

    if source == 'db':
        from sqlalchemy import create_engine, text
        from sqlalchemy.engine import URL

        pg = config['postgresql']
        url = URL.create(
            drivername='postgresql+psycopg2',
            username=pg['user'],
            password=pg['password'],
            host=pg['host'],
            port=pg['port'],
            database=pg['database']
        )
        engine = create_engine(url)

        table = oa_config.get('table_name', 'sgis_stats_area_boundary')
        geom_col = oa_config.get('geometry_column', 'geom')
        oa_cd_col = oa_config.get('oa_cd_column', 'stats_area_cd')
        base_year_col = oa_config.get('base_year_column', 'base_year')

        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT MAX({base_year_col}) FROM {table}"))
            base_year = result.scalar()
        print(f"[OA SHP] BASE_YEAR: {base_year}")

        sql = text(f"SELECT * FROM {table} WHERE {base_year_col} = '{base_year}'")
        gdf = gpd.read_postgis(sql, engine, geom_col=geom_col)
        gdf['TOT_REG_CD'] = gdf[oa_cd_col].astype(str)
        print(f"[OA_SHP] DB LOAD COMPLETE: {len(gdf)}건 (BASE_YEAR={base_year})")
    # 혹시 모르니까
    else:
        gdf = gpd.read_file(xy_mapper_path, encoding='utf-8')
        print(f"[OA_SHP] DB Failed, LOAD BY FILE")
    
    gdf['TOT_REG_CD'] = norm_str(gdf['TOT_REG_CD'])
    return gdf, base_year 

def map_oa_cd_spatial(df_main, xy_mapper_path, config=None):
    """
    shp 파일 기반 point-in-polygon 매칭
    """
    print("\n" + "="*70)
    print("집계구 경계 기반 공간 매칭")
    print("="*70)
    
    log_status("집계구 경계 shp 파일 로드")

    gdf_boundary, base_year = load_oa_boundary(config, xy_mapper_path)
    # gdf_boundary = gpd.read_file(xy_mapper_path, encoding='utf-8')
    # gdf_boundary['TOT_REG_CD'] = norm_str(gdf_boundary['TOT_REG_CD'])
    
    print(f"  집계구 경계: {len(gdf_boundary):,}개")
    print(f"  좌표계: {gdf_boundary.crs}")
    
    if 'oa_cd' not in df_main.columns:
        df_main['oa_cd'] = None
    
    if 'X' not in df_main.columns or 'Y' not in df_main.columns:
        print("  경고: X, Y 컬럼 없음")
        del gdf_boundary
        gc.collect()
        return df_main
    
    log_status("좌표 숫자 변환")
    df_main['X_num'] = pd.to_numeric(df_main['X'], errors='coerce')
    df_main['Y_num'] = pd.to_numeric(df_main['Y'], errors='coerce')
    
    mask_valid = (df_main['X_num'].notna()) & (df_main['Y_num'].notna()) & \
                 (df_main['X_num'] != 0) & (df_main['Y_num'] != 0)
    n_valid = mask_valid.sum()
    n_invalid = (~mask_valid).sum()
    
    print(f"\n  메인 데이터: {len(df_main):,}건")
    print(f"  유효 좌표: {n_valid:,}건")
    print(f"  무효 좌표: {n_invalid:,}건")
    
    if n_valid == 0:
        print("  유효 좌표 없음")
        del gdf_boundary
        gc.collect()
        return df_main
    
    mx = df_main.loc[mask_valid, 'X_num']
    my = df_main.loc[mask_valid, 'Y_num']
    
    if (mx.between(120, 140).mean() > 0.9) and (my.between(30, 45).mean() > 0.9):
        main_crs = 'EPSG:4326'
        print(f"  메인 데이터 좌표계: WGS84 (EPSG:4326)")
    elif (mx.between(-50000, 700000).mean() > 0.5) and (my.between(-50000, 700000).mean() > 0.5):
        main_crs = 'EPSG:5181'
        print(f"  메인 데이터 좌표계: UTM-K (EPSG:5181)")
    elif (mx.between(700000, 1500000).mean() > 0.5) and (my.between(1200000, 2500000).mean() > 0.5):
        main_crs = 'EPSG:5179'
        print(f"  메인 데이터 좌표계: TM (EPSG:5179)")
    else:
        main_crs = 'EPSG:5186'
        print(f"  메인 데이터 좌표계: 기본값 (EPSG:5186)")
    
    if gdf_boundary.crs is None:
        print("  경고: shp 파일에 좌표계 정보 없음, 메인 데이터와 동일하다고 가정")
        gdf_boundary.set_crs(main_crs, inplace=True)
    elif gdf_boundary.crs.to_epsg() != int(main_crs.split(':')[1]):
        log_status(f"shp 파일 좌표계 변환: {gdf_boundary.crs} -> {main_crs}")
        gdf_boundary = gdf_boundary.to_crs(main_crs)
    else:
        print(f"  shp 파일 좌표계 일치: {gdf_boundary.crs}")
    
    log_status("메인 데이터를 GeoDataFrame으로 변환")
    
    df_valid = df_main[mask_valid].copy()
    geometry = [Point(xy) for xy in zip(df_valid['X_num'], df_valid['Y_num'])]
    
    gdf_main = gpd.GeoDataFrame(
        df_valid,
        geometry=geometry,
        crs=main_crs
    )
    
    print(f"  GeoDataFrame 생성: {len(gdf_main):,}건")
    
    log_status("공간 조인 시작 (시간 소요)")
    
    gdf_joined = gpd.sjoin(
        gdf_main,
        gdf_boundary[['TOT_REG_CD', 'geometry']],
        how='left',
        predicate='intersects'
    )
    
    oa_col = 'TOT_REG_CD_right' if 'TOT_REG_CD_right' in gdf_joined.columns else 'TOT_REG_CD'
    
    if oa_col in gdf_joined.columns:
        gdf_joined_unique = gdf_joined[~gdf_joined.index.duplicated(keep='first')]
        matched_idx = gdf_joined_unique.index[gdf_joined_unique[oa_col].notna()]
        
        if len(matched_idx) > 0:
            df_main.loc[matched_idx, 'oa_cd'] = gdf_joined_unique.loc[matched_idx, oa_col]
    
    matched = df_main['oa_cd'].notna().sum()
    total = len(df_main)
    match_rate = matched / total * 100
    
    print(f"\n공간 매칭 결과:")
    print(f"  성공: {matched:,} / {total:,} ({match_rate:.1f}%)")
    print(f"  실패: {total - matched:,}건")
    
    del gdf_boundary, gdf_main, gdf_joined, df_valid, geometry
    gc.collect()
    
    return df_main, base_year


def map_oa_cd_spatial_with_nearest(df_main, xy_mapper_path, config=None, nearest_threshold_m=500):
    """
    shp 파일 기반 point-in-polygon 매칭 + 실패시 nearest 매칭
    """
    print("\n" + "="*70)
    print("집계구 경계 기반 공간 매칭 (nearest 포함)")
    print("="*70)
    
    #gdf_boundary = gpd.read_file(xy_mapper_path, encoding='utf-8')
    #gdf_boundary['TOT_REG_CD'] = norm_str(gdf_boundary['TOT_REG_CD'])
    gdf_boundary, base_year = load_oa_boundary(config, xy_mapper_path)
    gdf_boundary = gdf_boundary.drop_duplicates(subset=['TOT_REG_CD'], keep='first')
    
    df_main, base_year = map_oa_cd_spatial(df_main, xy_mapper_path)
    
    n_missing = df_main['oa_cd'].isna().sum()
    
    if n_missing == 0:
        print("\n  2차 매칭 불필요 (모두 1차에서 완료)")
        del gdf_boundary
        gc.collect()
        return df_main
    
    print("\n" + "="*70)
    print(f"2차 매칭: 최근접 집계구 매칭 (threshold: {nearest_threshold_m}m)")
    print("="*70)
    print(f"  2차 매칭 대상: {n_missing:,}건")
    
    mask_valid = (df_main['X_num'].notna()) & (df_main['Y_num'].notna())
    mask_missing = df_main['oa_cd'].isna() & mask_valid
    
    df_missing = df_main[mask_missing].copy()
    n_process = len(df_missing)
    
    print(f"  처리 가능: {n_process:,}건")
    
    if n_process == 0:
        print("  처리 가능한 데이터 없음")
        del gdf_boundary
        gc.collect()
        return df_main
    
    mx = df_missing['X_num']
    my = df_missing['Y_num']
    
    if (mx.between(120, 140).mean() > 0.9) and (my.between(30, 45).mean() > 0.9):
        main_crs = 'EPSG:4326'
    elif (mx.between(-50000, 700000).mean() > 0.5) and (my.between(-50000, 700000).mean() > 0.5):
        main_crs = 'EPSG:5181'
    elif (mx.between(700000, 1500000).mean() > 0.5) and (my.between(1200000, 2500000).mean() > 0.5):
        main_crs = 'EPSG:5179'
    else:
        main_crs = 'EPSG:5186'
    
    if gdf_boundary.crs is None:
        gdf_boundary.set_crs(main_crs, inplace=True)
    elif gdf_boundary.crs.to_epsg() != int(main_crs.split(':')[1]):
        gdf_boundary = gdf_boundary.to_crs(main_crs)
    
    geometry = [Point(xy) for xy in zip(df_missing['X_num'], df_missing['Y_num'])]
    gdf_missing = gpd.GeoDataFrame(
        df_missing,
        geometry=geometry,
        crs=main_crs
    )
    
    log_status("TM 좌표계로 변환 (거리 계산)")
    gdf_missing_tm = gdf_missing.to_crs('EPSG:5186')
    gdf_boundary_tm = gdf_boundary.to_crs('EPSG:5186')
    
    log_status("최근접 집계구 탐색 (시간 소요)")
    gdf_nearest = gpd.sjoin_nearest(
        gdf_missing_tm,
        gdf_boundary_tm[['TOT_REG_CD', 'geometry']],
        how='left',
        distance_col='distance_m'
    )
    
    oa_col = 'TOT_REG_CD_right' if 'TOT_REG_CD_right' in gdf_nearest.columns else 'TOT_REG_CD'
    
    if 'distance_m' in gdf_nearest.columns and oa_col in gdf_nearest.columns:
        gdf_nearest_unique = gdf_nearest[~gdf_nearest.index.duplicated(keep='first')]
        mask_valid_match = (gdf_nearest_unique[oa_col].notna()) & \
                          (gdf_nearest_unique['distance_m'] <= nearest_threshold_m)
        
        print(f"\n  거리 임계값 이내: {mask_valid_match.sum():,}건")
        
        if mask_valid_match.sum() > 0:
            print(f"  평균 거리: {gdf_nearest_unique.loc[mask_valid_match, 'distance_m'].mean():.1f}m")
            print(f"  중간 거리: {gdf_nearest_unique.loc[mask_valid_match, 'distance_m'].median():.1f}m")
            
            matched_idx = gdf_nearest_unique.index[mask_valid_match]
            df_main.loc[matched_idx, 'oa_cd'] = gdf_nearest_unique.loc[matched_idx, oa_col]
    
    final_matched = df_main['oa_cd'].notna().sum()
    final_total = len(df_main)
    final_rate = final_matched / final_total * 100
    
    print(f"\n최종 매칭 결과:")
    print(f"  성공: {final_matched:,} / {final_total:,} ({final_rate:.1f}%)")
    print(f"  실패: {final_total - final_matched:,}건")
    
    del gdf_boundary, gdf_missing, gdf_nearest, gdf_missing_tm, gdf_boundary_tm, geometry
    gc.collect()
    
    return df_main, base_year
