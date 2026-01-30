# -*- coding: utf-8 -*-
"""
Step 2: 피처 엔지니어링

처리 내용:
1. 빈집 관련 파생변수 (vacant_risk_score)
2. 내부 데이터 집계구 단위 집계
   - 건축물 파생변수 (7개)
   - 토지 파생변수 (13개)
   - 전기 파생변수 (11개)
3. 외부 집계구 v1 파생변수 (300개)
4. 외부 시설 파생변수 (130개)
5. 전체 병합 및 정규화

출력:
- 06_internal_aggregated.parquet
- 07_external_oa_v1_features.parquet
- 08_external_facility_features.parquet
- 09_final_before_clustering.parquet
"""
import os
import gc
from datetime import datetime

import pandas as pd
import numpy as np

from .utils import (
    log_status, normalize_oa_cd, optimize_dtypes, 
    to_db, clean_numeric_columns
)


def normalize_key(df, key_col):
    """키 컬럼 정규화"""
    if key_col in df.columns:
        df[key_col] = df[key_col].astype(str).str.strip()
    return df


def safe_join(left, right, on, how='left'):
    """안전한 병합"""
    if right.empty or on not in right.columns:
        return left
    
    if right[on].duplicated().any():
        right = right.drop_duplicates(subset=[on], keep='first')
    
    if on in left.columns and on in right.columns:
        left[on] = left[on].astype(str).str.strip()
        right[on] = right[on].astype(str).str.strip()
    
    return left.merge(right, on=on, how=how)


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


def process_building(df, logger=None):
    """건축물 파생변수 생성"""
    print("\n" + "="*70)
    print("건축물 파생변수 생성")
    print("="*70)
    
    df = normalize_key(df, 'oa_cd')
    out = pd.DataFrame({'oa_cd': df['oa_cd'].unique()})
    print(f"  고유 oa_cd 수: {len(out):,}개")
    
    drop_cols = []
    
    if 'STRCT_CD_NM' in df.columns:
        print("\n1. 건물구조 분류")
        good_structures = {
            '철근콘크리트구조', '철골철근콘크리트구조', '철골구조',
            '철골콘크리트구조', '조적구조', '프리캐스트콘크리트구조', '강구조'
        }
        structure_class = np.where(df['STRCT_CD_NM'].isin(good_structures), '양호', '불량')
        dummies = pd.get_dummies(structure_class, prefix='건물구조')
        agg = dummies.groupby(df['oa_cd']).mean().reset_index()
        out = safe_join(out, agg, on='oa_cd')
        if '건물구조_양호' in out.columns:
            print(f"  양호 구조 비율: {out['건물구조_양호'].mean()*100:.1f}%")
        drop_cols.append('STRCT_CD_NM')
    
    print("\n2. 건물나이 및 노후 비율")
    date_col = None
    if 'HP_BLDG_BDATE' in df.columns:
        date_col = 'HP_BLDG_BDATE'
        date_format = '%Y%m%d'
    elif 'USEAPR_DATE' in df.columns:
        date_col = 'USEAPR_DATE'
        date_format = None
    
    if date_col:
        current_year = pd.Timestamp.now().year
        
        if date_col == 'USEAPR_DATE':
            date_str = df[date_col].astype(str).str.strip()
            year_4digit = date_str.str.extract(r'^(\d{4})$')[0]
            date_8digit = pd.to_datetime(date_str, format='%Y%m%d', errors='coerce')
            year_8digit = date_8digit.dt.year
            use_year = pd.to_numeric(year_4digit.combine_first(year_8digit.astype(str)), errors='coerce')
            building_age = current_year - use_year
            building_age = building_age.fillna(30)
        else:
            use_date = pd.to_datetime(df[date_col], format=date_format, errors='coerce')
            building_age = current_year - use_date.dt.year
            building_age = building_age.fillna(30)
        
        avg_age = building_age.groupby(df['oa_cd']).mean().rename('평균건물나이').reset_index()
        out = safe_join(out, avg_age, on='oa_cd')
        
        old_ratio = (building_age >= 30).groupby(df['oa_cd']).mean().rename('노후건물비율').reset_index()
        out = safe_join(out, old_ratio, on='oa_cd')
        
        print(f"  평균 건물나이: {out['평균건물나이'].mean():.1f}년")
        print(f"  노후건물 비율: {out['노후건물비율'].mean()*100:.1f}%")
        drop_cols.append(date_col)
    
    if 'APT_TYPE' in df.columns:
        print("\n3. 아파트 비율")
        apt_ratio = (df['APT_TYPE'] == '1').groupby(df['oa_cd']).mean().rename('아파트비율').reset_index()
        out = safe_join(out, apt_ratio, on='oa_cd')
        print(f"  아파트 비율: {out['아파트비율'].mean()*100:.1f}%")
        drop_cols.append('APT_TYPE')
    
    if 'TBOOK_AREA' in df.columns:
        print("\n4. 토지면적")
        area_numeric = pd.to_numeric(df['TBOOK_AREA'], errors='coerce').fillna(0)
        avg_area = area_numeric.groupby(df['oa_cd']).mean().rename('평균토지면적').reset_index()
        out = safe_join(out, avg_area, on='oa_cd')
        print(f"  평균 토지면적: {out['평균토지면적'].mean():.1f}m2")
        drop_cols.append('TBOOK_AREA')
    
    violation_cols = ['GS_GIYUK', 'GS_HUNG', 'GS_JUB']
    found_violation = [col for col in violation_cols if col in df.columns]
    if found_violation:
        print(f"\n5. 위반건축물")
        violation_sum = pd.to_numeric(df[found_violation[0]], errors='coerce').fillna(0)
        for col in found_violation[1:]:
            violation_sum += pd.to_numeric(df[col], errors='coerce').fillna(0)
        violation_ratio = (violation_sum > 0).groupby(df['oa_cd']).mean().rename('위반건축물비율').reset_index()
        out = safe_join(out, violation_ratio, on='oa_cd')
        print(f"  위반건축물 비율: {out['위반건축물비율'].mean()*100:.1f}%")
        drop_cols.extend(found_violation)
    
    out = out.fillna(0)
    out = optimize_dtypes(out)
    
    total_vars = len(out.columns) - 1
    print(f"\n건축물 파생변수 완료: {out.shape} (총 {total_vars}개 변수)")
    
    return out, drop_cols


def process_land(df, logger=None):
    """토지 파생변수 생성"""
    print("\n" + "="*70)
    print("토지 파생변수 생성")
    print("="*70)
    
    df = normalize_key(df, 'oa_cd')
    out = pd.DataFrame({'oa_cd': df['oa_cd'].unique()})
    
    drop_cols = []
    
    price_cols = [col for col in df.columns if col.startswith('PRICE') and col[5:].isdigit()]
    price_cols = sorted(price_cols)
    
    if len(price_cols) > 0:
        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        latest_price_col = price_cols[-1]
        latest_price = df.groupby('oa_cd')[latest_price_col].mean().rename('최신공시지가').reset_index()
        out = safe_join(out, latest_price, on='oa_cd')
        
        if len(price_cols) >= 3:
            recent_3_cols = price_cols[-3:]
            df['최근공시지가_변동률'] = (
                (df[recent_3_cols[-1]] - df[recent_3_cols[0]]) /
                (df[recent_3_cols[0]] + 1) * 100
            )
            change_rate = df.groupby('oa_cd')['최근공시지가_변동률'].mean().rename('공시지가_3년변동률').reset_index()
            out = safe_join(out, change_rate, on='oa_cd')
        
        if len(price_cols) >= 2:
            df['공시지가_하락여부'] = (df[price_cols[-1]] < df[price_cols[-2]]).astype(int)
            decline_ratio = df.groupby('oa_cd')['공시지가_하락여부'].mean().rename('공시지가_하락비율').reset_index()
            out = safe_join(out, decline_ratio, on='oa_cd')
        
        df['공시지가_변동성'] = df[price_cols].std(axis=1)
        volatility = df.groupby('oa_cd')['공시지가_변동성'].mean().rename('공시지가_변동성').reset_index()
        out = safe_join(out, volatility, on='oa_cd')
        
        if latest_price_col in df.columns:
            price_25th = df[df[latest_price_col] > 0][latest_price_col].quantile(0.25)
            df['저가토지'] = (df[latest_price_col] <= price_25th).astype(int)
            low_price_ratio = df.groupby('oa_cd')['저가토지'].mean().rename('저가토지_비율').reset_index()
            out = safe_join(out, low_price_ratio, on='oa_cd')
        
        drop_cols.extend(price_cols)
    
    gakuka_cols = [col for col in df.columns if col.startswith('GAKUKA') and col[6:].isdigit()]
    gakuka_cols = sorted(gakuka_cols)
    
    if len(gakuka_cols) > 0:
        for col in gakuka_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        latest_gakuka_col = gakuka_cols[-1]
        latest_gakuka = df.groupby('oa_cd')[latest_gakuka_col].mean().rename('최신개별공시지가').reset_index()
        out = safe_join(out, latest_gakuka, on='oa_cd')
        
        if len(gakuka_cols) >= 3:
            recent_3_cols = gakuka_cols[-3:]
            df['최근개별공시지가_변동률'] = (
                (df[recent_3_cols[-1]] - df[recent_3_cols[0]]) /
                (df[recent_3_cols[0]] + 1) * 100
            )
            change_rate = df.groupby('oa_cd')['최근개별공시지가_변동률'].mean().rename('개별공시지가_3년변동률').reset_index()
            out = safe_join(out, change_rate, on='oa_cd')
        
        if len(gakuka_cols) >= 2:
            df['개별공시지가_하락여부'] = (df[gakuka_cols[-1]] < df[gakuka_cols[-2]]).astype(int)
            decline_ratio = df.groupby('oa_cd')['개별공시지가_하락여부'].mean().rename('개별공시지가_하락비율').reset_index()
            out = safe_join(out, decline_ratio, on='oa_cd')
        
        df['개별공시지가_변동성'] = df[gakuka_cols].std(axis=1)
        volatility = df.groupby('oa_cd')['개별공시지가_변동성'].mean().rename('개별공시지가_변동성').reset_index()
        out = safe_join(out, volatility, on='oa_cd')
        
        if latest_gakuka_col in df.columns:
            gakuka_25th = df[df[latest_gakuka_col] > 0][latest_gakuka_col].quantile(0.25)
            df['개별저가토지'] = (df[latest_gakuka_col] <= gakuka_25th).astype(int)
            low_gakuka_ratio = df.groupby('oa_cd')['개별저가토지'].mean().rename('개별저가토지_비율').reset_index()
            out = safe_join(out, low_gakuka_ratio, on='oa_cd')
        
        drop_cols.extend(gakuka_cols)
    
    if len(price_cols) > 0 and 'TBOOK_AREA' in df.columns:
        area = pd.to_numeric(df['TBOOK_AREA'], errors='coerce').fillna(1)
        latest_price_col = price_cols[-1]
        df['단위면적당_공시지가'] = df[latest_price_col] / area
        unit_price = df.groupby('oa_cd')['단위면적당_공시지가'].mean().rename('단위면적당_공시지가').reset_index()
        out = safe_join(out, unit_price, on='oa_cd')
    
    land_use_patterns = ['전', '답', '과수원', '목장용지', '임야', '대', '공장용지', '도로']
    
    print("\n토지 이용 현황 집계")
    for pattern in land_use_patterns:
        matching_cols = [col for col in df.columns if pattern in str(col)]
        if matching_cols:
            for col in matching_cols[:1]:
                val_numeric = pd.to_numeric(df[col], errors='coerce').fillna(0)
                agg = val_numeric.groupby(df['oa_cd']).mean().rename(f'토지_{pattern}_x').reset_index()
                out = safe_join(out, agg, on='oa_cd')
                drop_cols.append(col)
    
    out = out.fillna(0)
    out = optimize_dtypes(out)
    
    total_vars = len(out.columns) - 1
    print(f"\n토지 파생변수 완료: {out.shape} (총 {total_vars}개 변수)")
    
    return out, drop_cols


def process_energy(df, logger=None):
    """전기 파생변수 집계"""
    print("\n" + "="*70)
    print("전기 파생변수 집계")
    print("="*70)
    
    df = normalize_key(df, 'oa_cd')
    out = pd.DataFrame({'oa_cd': df['oa_cd'].unique()})
    
    energy_cols = [
        'mean_kwh', 'zero_ratio', 'drop_ratio', 'sudden_drop_ratio',
        'max_drop_amount', 'consecutive_zero_max', 'yoy_change_mean',
        'summer_mean', 'winter_mean', 'cv', 'recent_3month_zero'
    ]
    existing_energy = [col for col in energy_cols if col in df.columns]
    
    if existing_energy:
        df_energy_agg = df.groupby('oa_cd')[existing_energy].mean().reset_index()
        
        rename_map = {
            'mean_kwh': '전기_평균사용량',
            'zero_ratio': '전기_제로비율',
            'drop_ratio': '전기_감소비율',
            'sudden_drop_ratio': '전기_급감비율',
            'max_drop_amount': '전기_최대감소량',
            'consecutive_zero_max': '전기_연속제로최대',
            'yoy_change_mean': '전기_전년대비변화율',
            'summer_mean': '전기_여름평균',
            'winter_mean': '전기_겨울평균',
            'cv': '전기변동계수',
            'recent_3month_zero': '전기_최근3개월제로'
        }
        
        df_energy_agg.columns = ['oa_cd'] + [rename_map.get(col, col) for col in existing_energy]
        out = safe_join(out, df_energy_agg, on='oa_cd')
        print(f"  전기 파생변수: {len(existing_energy)}개")
    else:
        print("  전기 파생변수 없음")
    
    out = out.fillna(0)
    out = optimize_dtypes(out)
    
    total_vars = len(out.columns) - 1
    print(f"\n전기 파생변수 완료: {out.shape} (총 {total_vars}개 변수)")
    
    return out, []


def process_external_oa_v1(df, logger=None):
    """외부 집계구 v1 데이터 파생변수 생성"""
    print("\n" + "="*70)
    print("외부 집계구 v1 파생변수 생성")
    print("="*70)
    
    df = normalize_key(df, 'oa_cd')
    out = pd.DataFrame({'oa_cd': df['oa_cd'].unique()})
    print(f"  고유 oa_cd 수: {len(out):,}개")
    
    population_sum_mappings = {
        'to_in_001': '총인구',
        'to_in_007': '남성인구',
        'to_in_008': '여성인구',
    }
    
    population_mean_mappings = {
        'to_in_002': '평균나이',
        'to_in_003': '인구밀도',
        'to_in_004': '노령화지수',
        'to_in_005': '유소년부양비',
        'to_in_006': '노년부양비',
    }
    
    print("\n[1] 인구 총합")
    for code, name in population_sum_mappings.items():
        if code in df.columns:
            val = pd.to_numeric(df[code], errors='coerce').fillna(0)
            agg = val.groupby(df['oa_cd']).sum().rename(name).reset_index()
            out = safe_join(out, agg, on='oa_cd')
    
    for code, name in population_mean_mappings.items():
        if code in df.columns:
            val = pd.to_numeric(df[code], errors='coerce').fillna(0)
            agg = val.groupby(df['oa_cd']).mean().rename(name).reset_index()
            out = safe_join(out, agg, on='oa_cd')
    
    household_mappings = {
        'to_ga_001': ('총가구수', 'sum'),
        'to_ga_002': ('평균가구원수', 'mean'),
    }
    
    print("\n[2] 가구 총합")
    for code, (name, agg_method) in household_mappings.items():
        if code in df.columns:
            val = pd.to_numeric(df[code], errors='coerce').fillna(0)
            if agg_method == 'sum':
                agg = val.groupby(df['oa_cd']).sum().rename(name).reset_index()
            else:
                agg = val.groupby(df['oa_cd']).mean().rename(name).reset_index()
            out = safe_join(out, agg, on='oa_cd')
    
    housing_mappings = {'to_ho_001': '총주택수'}
    print("\n[3] 주택 총합")
    for code, name in housing_mappings.items():
        if code in df.columns:
            val = pd.to_numeric(df[code], errors='coerce').fillna(0)
            agg = val.groupby(df['oa_cd']).sum().rename(name).reset_index()
            out = safe_join(out, agg, on='oa_cd')
    
    business_mappings = {'to_fa_010': '총사업체수'}
    print("\n[4] 사업체 총합")
    for code, name in business_mappings.items():
        if code in df.columns:
            val = pd.to_numeric(df[code], errors='coerce').fillna(0)
            agg = val.groupby(df['oa_cd']).sum().rename(name).reset_index()
            out = safe_join(out, agg, on='oa_cd')
    
    employee_mappings = {'to_em_020': '총종사자수'}
    print("\n[5] 종사자 총합")
    for code, name in employee_mappings.items():
        if code in df.columns:
            val = pd.to_numeric(df[code], errors='coerce').fillna(0)
            agg = val.groupby(df['oa_cd']).sum().rename(name).reset_index()
            out = safe_join(out, agg, on='oa_cd')
    
    print("\n[6] 연령대별 인구")
    age_patterns = ['in_age_', 'in_grp_', 'n_age_', 'n_grp_']
    age_count = 0
    for col in df.columns:
        if any(pattern in col for pattern in age_patterns):
            val = pd.to_numeric(df[col], errors='coerce').fillna(0)
            agg = val.groupby(df['oa_cd']).sum().rename(f'연령_{col}').reset_index()
            out = safe_join(out, agg, on='oa_cd')
            age_count += 1
    print(f"  연령대별 인구: {age_count}개")
    
    out = out.fillna(0)
    out = optimize_dtypes(out)
    
    total_vars = len(out.columns) - 1
    print(f"\n외부 집계구 v1 파생변수 완료: {out.shape} (총 {total_vars}개 변수)")
    
    return out, []


def process_external_facility(df_facility, df_population=None, logger=None):
    """외부 시설 기초 외국인 데이터 파생변수 생성"""
    print("\n" + "="*70)
    print("외부 시설 기초 외국인 데이터 파생변수 생성")
    print("="*70)
    
    df = normalize_key(df_facility, 'oa_cd')
    print(f"  원본 행 수: {len(df):,}개")
    print(f"  고유 oa_cd 수: {df['oa_cd'].nunique():,}개")
    
    out = pd.DataFrame({'oa_cd': df['oa_cd'].unique()})
    
    value_cols = [col for col in df.columns if col != 'oa_cd']
    
    foreigner_cols = []
    for col in value_cols:
        if col.startswith('외국인') or col.startswith('기초생활수급자수'):
            foreigner_cols.append(col)
    
    for col in foreigner_cols:
        val = pd.to_numeric(df[col], errors='coerce').fillna(0)
        agg = val.groupby(df['oa_cd']).sum().rename(col).reset_index()
        out = safe_join(out, agg, on='oa_cd')
    
    rate_cols = []
    for col in value_cols:
        if col not in foreigner_cols:
            if any(x in col for x in ['증가율', '증가량']):
                rate_cols.append(col)
    
    for col in rate_cols:
        val = pd.to_numeric(df[col], errors='coerce').fillna(0)
        agg = val.groupby(df['oa_cd']).sum().rename(col).reset_index()
        out = safe_join(out, agg, on='oa_cd')
    
    density_cols = []
    for col in value_cols:
        if col not in foreigner_cols and col not in rate_cols:
            if '밀도' in col:
                density_cols.append(col)
    
    for col in density_cols:
        val = pd.to_numeric(df[col], errors='coerce').fillna(0)
        agg = val.groupby(df['oa_cd']).sum().rename(col).reset_index()
        out = safe_join(out, agg, on='oa_cd')
    
    facility_2024_cols = []
    for col in value_cols:
        if col not in foreigner_cols and col not in rate_cols and col not in density_cols:
            if col.endswith('_2024'):
                facility_2024_cols.append(col)
    
    for col in facility_2024_cols:
        val = pd.to_numeric(df[col], errors='coerce').fillna(0)
        agg = val.groupby(df['oa_cd']).sum().rename(col).reset_index()
        out = safe_join(out, agg, on='oa_cd')
    
    if df_population is not None:
        pop_cols_to_merge = ['oa_cd']
        if '총인구' in df_population.columns:
            pop_cols_to_merge.append('총인구')
        if '총가구수' in df_population.columns:
            pop_cols_to_merge.append('총가구수')
        if '총주택수' in df_population.columns:
            pop_cols_to_merge.append('총주택수')
        
        if len(pop_cols_to_merge) > 1:
            out = safe_join(out, df_population[pop_cols_to_merge], on='oa_cd')
            
            if '총가구수' in out.columns:
                for col in facility_2024_cols:
                    facility_name = col.replace('_2024', '')
                    density_col = f'{facility_name}_가구밀도'
                    out[density_col] = np.where(
                        out['총가구수'] > 0,
                        (out[col] / out['총가구수']) * 10000,
                        0
                    )
            
            if '총인구' in out.columns:
                for col in facility_2024_cols:
                    facility_name = col.replace('_2024', '')
                    density_col = f'{facility_name}_인구밀도'
                    out[density_col] = np.where(
                        out['총인구'] > 0,
                        (out[col] / out['총인구']) * 10000,
                        0
                    )
            
            if '총주택수' in out.columns:
                for col in facility_2024_cols:
                    facility_name = col.replace('_2024', '')
                    density_col = f'{facility_name}_주택밀도'
                    out[density_col] = np.where(
                        out['총주택수'] > 0,
                        (out[col] / out['총주택수']) * 10000,
                        0
                    )
            
            numeric_cols = out.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                out[col] = out[col].replace([np.inf, -np.inf], 0)
            out = out.drop(columns=['총인구', '총가구수', '총주택수'], errors='ignore')
    
    out = out.fillna(0)
    out = optimize_dtypes(out)
    
    total_vars = len(out.columns) - 1
    print(f"\n외부 시설 파생변수 완료: {out.shape} (총 {total_vars}개 변수)")
    
    return out, []


def aggregate_to_oa(df_internal, logger=None):
    """내부 데이터를 집계구 단위로 집계"""
    print("\n" + "="*70)
    print("내부 데이터 집계구 단위 집계")
    print("="*70)
    
    df_internal = normalize_key(df_internal, 'oa_cd')
    df_internal = df_internal[df_internal['oa_cd'].notna()].copy()
    
    print(f"  원본 건물 수: {len(df_internal):,}개")
    print(f"  고유 집계구 수: {df_internal['oa_cd'].nunique():,}개")
    
    has_vacant = 'VACANT' in df_internal.columns
    coord_agg = None
    vacant_agg = None
    
    if has_vacant:
        df_internal['VACANT'] = pd.to_numeric(df_internal['VACANT'], errors='coerce').fillna(0)
        
        agg_dict = {
            'VACANT': ['sum', 'mean'],
            'BLDRGST_SEQNO': 'count'
        }
        
        if 'X' in df_internal.columns:
            df_internal['X'] = pd.to_numeric(df_internal['X'], errors='coerce')
            agg_dict['X'] = 'mean'
        if 'Y' in df_internal.columns:
            df_internal['Y'] = pd.to_numeric(df_internal['Y'], errors='coerce')
            agg_dict['Y'] = 'mean'
        
        vacant_agg = df_internal.groupby('oa_cd').agg(agg_dict)
        
        new_cols = []
        for col in vacant_agg.columns:
            if isinstance(col, tuple):
                if col[0] == 'VACANT' and col[1] == 'sum':
                    new_cols.append('VACANT_sum')
                elif col[0] == 'VACANT' and col[1] == 'mean':
                    new_cols.append('VACANT_mean')
                elif col[0] == 'BLDRGST_SEQNO':
                    new_cols.append('building_count')
                elif col[0] in ['X', 'Y']:
                    new_cols.append(col[0])
            else:
                new_cols.append(col)
        
        vacant_agg.columns = new_cols
        vacant_agg['VACANT_RATIO'] = vacant_agg['VACANT_sum'] / vacant_agg['building_count']
        vacant_agg = vacant_agg.reset_index()
        
        print(f"  평균 빈집 비율: {vacant_agg['VACANT_RATIO'].mean():.2%}")
    else:
        if 'X' in df_internal.columns and 'Y' in df_internal.columns:
            df_internal['X'] = pd.to_numeric(df_internal['X'], errors='coerce')
            df_internal['Y'] = pd.to_numeric(df_internal['Y'], errors='coerce')
            coord_agg = df_internal.groupby('oa_cd')[['X', 'Y']].mean().reset_index()
    
    print("\n[1] 건축물 파생변수 생성")
    df_building, building_drop_cols = process_building(df_internal)
    
    print("\n[2] 토지 파생변수 생성")
    df_land, land_drop_cols = process_land(df_internal)
    
    print("\n[3] 전기 파생변수 집계")
    df_energy_agg, _ = process_energy(df_internal)
    
    print("\n[4] 전체 병합")
    df_final = df_building
    df_final = safe_join(df_final, df_land, on='oa_cd')
    df_final = safe_join(df_final, df_energy_agg, on='oa_cd')
    
    if has_vacant:
        merge_cols = ['oa_cd', 'VACANT_sum', 'VACANT_mean', 'building_count', 'VACANT_RATIO']
        if 'X' in vacant_agg.columns:
            merge_cols.append('X')
        if 'Y' in vacant_agg.columns:
            merge_cols.append('Y')
        df_final = safe_join(df_final, vacant_agg[merge_cols], on='oa_cd')
        print(f"  VACANT_RATIO 추가됨 (타겟 변수)")
    elif coord_agg is not None:
        df_final = safe_join(df_final, coord_agg, on='oa_cd')
    
    df_final = df_final.fillna(0)
    df_final = optimize_dtypes(df_final)
    
    print(f"\n최종 내부 집계구 데이터: {df_final.shape} ({len(df_final.columns)-1}개 변수)")
    
    return df_final


def run_step2(input_dir, output_dir, pg_connection=None, run_id=None,timestamp=None, logger=None):
    """Step 2 실행: 파생변수 생성 및 집계구 집계"""
    print("\n" + "="*80)
    print("Step 2: 파생변수 생성 및 집계구 집계 시작")
    print("="*80)
    start_time = datetime.now()
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n[Step 2-1] 병합된 메인 데이터 로드")
    log_status("파일 로드: 03_merged_main_energy.parquet")
    
    parquet_path = os.path.join(input_dir, "03_merged_main_energy.parquet")
    if not os.path.exists(parquet_path):
        parquet_path = os.path.join(input_dir, "03_merged_main_energy.parquet")
    
    df_merged = pd.read_parquet(parquet_path)
    
    if 'oa_cd' in df_merged.columns:
        df_merged['oa_cd'] = df_merged['oa_cd'].astype(str).str.strip()
    elif '집계구코드' in df_merged.columns:
        df_merged = df_merged.rename(columns={'집계구코드': 'oa_cd'})
        df_merged['oa_cd'] = df_merged['oa_cd'].astype(str).str.strip()
    
    print(f"  로드 완료: {df_merged.shape}")
    
    df_merged = fix_merged_dtypes(df_merged)
    
    print("\n[Step 2-2] 내부 데이터 집계구 집계")
    df_internal_oa = aggregate_to_oa(df_merged)
    
    df_internal_oa = normalize_oa_cd(df_internal_oa, 'oa_cd')
    
    output_path_6_parquet = os.path.join(output_dir, "06_internal_aggregated.parquet")
    output_path_6_csv = os.path.join(output_dir, "06_internal_aggregated.csv")
    df_internal_oa.to_parquet(output_path_6_parquet, index=False)
    df_internal_oa.to_csv(output_path_6_csv, index=False, encoding='cp949')
    log_status(f"저장: {output_path_6_parquet}")
    log_status(f"저장: {output_path_6_csv}")
    
    del df_merged
    gc.collect()
    
    print("\n[Step 2-3] 외부 집계구 v1 데이터 처리")
    external_oa_path = os.path.join(input_dir, "04_external_oa_v1.parquet")
    df_external_oa_raw = pd.read_parquet(external_oa_path)
    
    df_external_oa_raw = normalize_oa_cd(df_external_oa_raw, 'oa_cd')
    
    df_external_oa_v1, _ = process_external_oa_v1(df_external_oa_raw)
    df_external_oa_v1 = normalize_oa_cd(df_external_oa_v1, 'oa_cd')
    
    output_path_7_parquet = os.path.join(output_dir, "07_external_oa_v1_features.parquet")
    output_path_7_csv = os.path.join(output_dir, "07_external_oa_v1_features.csv")
    df_external_oa_v1.to_parquet(output_path_7_parquet, index=False)
    df_external_oa_v1.to_csv(output_path_7_csv, index=False, encoding='cp949')
    log_status(f"저장: {output_path_7_parquet}")
    log_status(f"저장: {output_path_7_csv}")
    
    del df_external_oa_raw
    gc.collect()
    
    print("\n[Step 2-4] 외부 시설 데이터 처리")
    facility_path = os.path.join(input_dir, "05_external_facility.parquet")
    df_facility = pd.read_parquet(facility_path)
    
    unnamed_cols = [col for col in df_facility.columns if col.startswith('Unnamed')]
    if unnamed_cols:
        df_facility = df_facility.drop(columns=unnamed_cols)
    
    df_facility = normalize_oa_cd(df_facility, 'oa_cd')
    
    df_external_facility, _ = process_external_facility(df_facility, df_population=df_external_oa_v1)
    df_external_facility = normalize_oa_cd(df_external_facility, 'oa_cd')
    
    output_path_8_parquet = os.path.join(output_dir, "08_external_facility_features.parquet")
    output_path_8_csv = os.path.join(output_dir, "08_external_facility_features.csv")
    df_external_facility.to_parquet(output_path_8_parquet, index=False)
    df_external_facility.to_csv(output_path_8_csv, index=False, encoding='cp949')
    log_status(f"저장: {output_path_8_parquet}")
    log_status(f"저장: {output_path_8_csv}")
    
    del df_facility
    gc.collect()
    
    print("\n[Step 2-5] 내부 + 외부 데이터 최종 병합")
    df_final = df_internal_oa
    
    overlap = set(df_final.columns) & set(df_external_oa_v1.columns) - {'oa_cd'}
    if overlap:
        df_external_oa_v1 = df_external_oa_v1.drop(columns=list(overlap))
    
    df_final = safe_join(df_final, df_external_oa_v1, on='oa_cd')
    print(f"  외부 v1 병합 후: {df_final.shape}")
    
    overlap = set(df_final.columns) & set(df_external_facility.columns) - {'oa_cd'}
    if overlap:
        df_external_facility = df_external_facility.drop(columns=list(overlap))
    
    df_final = safe_join(df_final, df_external_facility, on='oa_cd')
    df_final = normalize_oa_cd(df_final, 'oa_cd')
    print(f"  외부 시설 병합 후: {df_final.shape}")
    print(f"  총 변수 수: {len(df_final.columns)-1}개")
    
    df_final = clean_numeric_columns(df_final)
    
    output_path_9_parquet = os.path.join(output_dir, "09_final_before_clustering.parquet")
    output_path_9_csv = os.path.join(output_dir, "09_final_before_clustering.csv")
    df_final.to_parquet(output_path_9_parquet, index=False)
    df_final.to_csv(output_path_9_csv, index=False, encoding='cp949')
    
    if pg_connection is not None and timestamp is not None:
        
        mapping_df = pd.read_excel('/home/was/project/data/feature_mapping.xlsx')
        kor_to_eng = dict(zip(mapping_df['기존 변수명'], mapping_df['영어 변수명']))
        df_final = df_final.rename(columns=kor_to_eng)
        

        df_final['run_id'] = run_id
        to_db(df_final, 'tb_pipeline_study', pg_connection, 
              db_type='postgresql', force_text_cols=['oa_cd'], if_exists=append,
              varchar_lengths={'oa_cd': 20}, primary_key=['oa_cd','run_id'])
    
    log_status(f"저장: {output_path_9_parquet}")
    log_status(f"저장: {output_path_9_csv}")
    
    del df_internal_oa, df_external_oa_v1, df_external_facility
    gc.collect()
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds() / 60
    
    print("\n" + "="*80)
    print(f"Step 2 완료! 소요 시간: {elapsed:.1f}분")
    print(f"최종 데이터: {df_final.shape}")
    print(f"저장 위치: {output_dir}")
    print("="*80)
    
    return df_final


if __name__ == "__main__":
    import yaml
    
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    base_path = config['paths']['base_path']
    output_dirs = [d for d in os.listdir(base_path) if d.startswith('output_')]
    latest_output = sorted(output_dirs)[-1]
    
    input_dir = os.path.join(base_path, latest_output, "step1")
    output_dir = os.path.join(base_path, latest_output, "step2")
    
    run_step2(input_dir=input_dir, output_dir=output_dir)
