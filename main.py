# -*- coding: utf-8 -*-
"""
빈집 예측 파이프라인 - 메인 실행 파일

사용법:
    python main.py                          # 전체 파이프라인 실행
    python main.py --config config.yaml     # 설정 파일 지정
    python main.py --step 2                 # 특정 Step만 실행
    python main.py --start-step 3 --end-step 5  # Step 3~5 실행
"""
import os
import sys
import argparse
import gc
import yaml
from datetime import datetime

import yaml

def init_oracle_client(client_path):
    """Oracle Client 초기화"""
    if client_path and os.path.exists(client_path):
        try:
            import oracledb
            oracledb.init_oracle_client(lib_dir=client_path)
            print(f"Oracle Client 초기화 완료: {client_path}")
        except Exception as e:
            print(f"Oracle Client 초기화 실패: {e}")


def load_config(config_path='configs/config.yaml'):
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

from src.config_loader import load_config 


def create_output_dirs(base_path, include_step6=False):
    """출력 디렉토리 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = os.path.join(base_path, f"output_{timestamp}")
    
    dirs = {
        'base': output_base,
        'step1': os.path.join(output_base, 'step1'),
        'step2': os.path.join(output_base, 'step2'),
        'step3': os.path.join(output_base, 'step3'),
        'step4': os.path.join(output_base, 'step4'),
        'step5': os.path.join(output_base, 'step5'),
    }
    if include_step6:
        dirs['step6'] = os.path.join(output_base, 'step6')
    
    for path in dirs.values():
        os.makedirs(path, exist_ok=True) 

    return dirs, timestamp


def connect_oracle(config):
    """Oracle DB 연결"""
    if not config['oracle'].get('enable', False):
        return None
    
    try:
        import oracledb
        
        connection = oracledb.connect(
            user=config['oracle']['user'],
            password=config['oracle']['password'],
            host=config['oracle']['host'],
            port=config['oracle']['port'],
            service_name=config['oracle']['service_name']
        )
        print("Oracle DB 연결 성공")
        return connection
    except Exception as e:
        print(f"Oracle DB 연결 실패: {e}")
        return None


def connect_postgresql(config):
    """PostgreSQL DB 연결"""
    if not config['postgresql'].get('enable', False):
        return None
    
    try:
        import psycopg2
        
        connection = psycopg2.connect(
            user=config['postgresql']['user'],
            password=config['postgresql']['password'],
            host=config['postgresql']['host'],
            port=config['postgresql']['port'],
            database=config['postgresql']['database']
        )
        print("Po:stgreSQL DB 연결 성공")
        return connection
    except Exception as e:
        print(f"PostgreSQL DB 연결 실패: {e}")
        return None


def find_latest_step_dir(base_path, step_name):
    """최신 Step 디렉토리 찾기"""
    output_dirs = [d for d in os.listdir(base_path) if d.startswith('output_')]
    if not output_dirs:
        raise FileNotFoundError(f"output 디렉토리를 찾을 수 없습니다: {base_path}")
    
    latest_output = sorted(output_dirs)[-1]
    step_dir = os.path.join(base_path, latest_output, step_name)
    
    if not os.path.exists(step_dir):
        raise FileNotFoundError(f"{step_name} 디렉토리를 찾을 수 없습니다: {step_dir}")
    
    return step_dir


def run_pipeline(config, output_dirs, oracle_conn, pg_conn, timestamp,
                 start_step=1, end_step=5, pipeline_logger=None, run_id=None):
    """파이프라인 실행"""
    from src.step1_data_integration import run_step1
    from src.step2_feature_engineering import run_step2
    from src.step3_clustering import run_step3
    from src.step4_modeling import run_step4
    from src.step5_ensemble import run_step5
    
    results = {}
    
    base_path = config['paths']['base_path']

    print("\n" + "="*80)
    print("빈집 예측 파이프라인 시작")
    print(f"실행 Step: {start_step} ~ {end_step}")
    print("="*80)
    
    pipeline_start = datetime.now()
    
    # Step 1: 데이터 통합
    if start_step <= 1 <= end_step:
        if oracle_conn is None:
            print("\nStep 1 스킵: Oracle 연결 필요")
        else:

            xy_mapper_full = os.path.join(base_path, config['paths']['xy_mapper_path'])
            external_oa_full = os.path.join(base_path, config['paths']['external_oa_path'])
            external_facility_full = os.path.join(base_path, config['paths']['external_facility_path'])

            results['step1'] = run_step1(
                connection=oracle_conn,
                xy_mapper_path=xy_mapper_full,
                external_oa_path=external_oa_full,
                external_facility_path=external_facility_full,
                output_dir=output_dirs['step1'],
                distance_threshold=config['step1'].get('distance_threshold', 2.0),
                remove_unmapped=config['step1'].get('remove_unmapped', True),
                pg_connection=pg_conn,
                run_id=run_id,
                timestamp=timestamp,
                logger=None
            )
            if pipeline_logger:
                pipeline_logger.update('step1')
        gc.collect()
    
    # Step 2: 피처 엔지니어링
    if start_step <= 2 <= end_step:
        input_dir = output_dirs['step1']
        if start_step > 1:
            input_dir = find_latest_step_dir(config['paths']['base_path'], 'step1')
        
        results['step2'] = run_step2(
            input_dir=input_dir,
            output_dir=output_dirs['step2'],
            pg_connection=pg_conn,
            run_id=run_id,
            logger=None
        )
        if pipeline_logger:
            pipeline_logger.update('step2')
        gc.collect()
    
    # Step 3: 클러스터링
    if start_step <= 3 <= end_step:
        input_dir = output_dirs['step2']
        if start_step > 2:
            input_dir = find_latest_step_dir(config['paths']['base_path'], 'step2')
        
        results['step3'] = run_step3(
            input_dir=input_dir,
            output_dir=output_dirs['step3'],
            k_range=range(
                config['step3'].get('k_min', 2),
                config['step3'].get('k_max', 11)
            ),
            k_final=config['step3'].get('k_final', None),
            pg_connection=pg_conn,
            run_id=run_id,
            logger=None
        )
        if pipeline_logger:
            pipeline_logger.update('step3')
        gc.collect()
    
    # Step 4: 모델링
    if start_step <= 4 <= end_step:
        input_dir_step2 = output_dirs['step2']
        input_dir_step3 = output_dirs['step3']
        
        if start_step > 2:
            input_dir_step2 = find_latest_step_dir(config['paths']['base_path'], 'step2')
        if start_step > 3:
            input_dir_step3 = find_latest_step_dir(config['paths']['base_path'], 'step3')
        
        results['step4'] = run_step4(
            input_dir_step2=input_dir_step2,
            input_dir_step3=input_dir_step3,
            output_dir=output_dirs['step4'],
            top_n_features=config['step4'].get('top_n_features', 30),
            use_two_stage=config['step4'].get('use_two_stage', True),
            use_gwr=config['step4'].get('use_gwr', False),
            pg_connection=pg_conn,
            run_id=run_id,
            logger=None
        )
        if pipeline_logger:
            pipeline_logger.update('step4')
        gc.collect()
    
    # Step 5: 앙상블 및 평가
    if start_step <= 5 <= end_step:
        input_dir = output_dirs['step4']
        if start_step > 4:
            input_dir = find_latest_step_dir(config['paths']['base_path'], 'step4')
        
        results['step5'] = run_step5(
            input_dir=input_dir,
            output_dir=output_dirs['step5'],
            use_cluster_ensemble=config['step5'].get('use_cluster_ensemble', True),
            pg_connection=pg_conn,
            run_id=run_id,
            logger=None
        )
        if pipeline_logger:
            pipeline_logger.update('step5')
        gc.collect()
    
    # Step 6: 5년 예측
    if start_step <= 6 <= end_step:
        from src.step6_forecast import run_step6
        
        if 'step6' not in output_dirs:
            output_dirs['step6'] = os.path.join(output_dirs['base'], 'step6')
            os.makedirs(output_dirs['step6'], exist_ok=True)
        
        results['step6'] = run_step6(
            config=config,
            output_dir=output_dirs['step6'],
            logger=None
        )
        if pipeline_logger:
            pipeline_logger.update('step6')
    
    pipeline_end = datetime.now()
    elapsed = (pipeline_end - pipeline_start).total_seconds() / 60
    
    print("\n" + "="*80)
    print(f"파이프라인 완료!")
    print(f"총 소요 시간: {elapsed:.1f}분")
    print(f"출력 디렉토리: {output_dirs['base']}")
    print("="*80)
    
    return results


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='빈집 예측 파이프라인')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='설정 파일 경로')
    parser.add_argument('--step', type=int, default=None,
                       help='실행할 Step (1-6)')
    parser.add_argument('--start-step', type=int, default=1,
                       help='시작 Step')
    parser.add_argument('--end-step', type=int, default=5,
                       help='종료 Step')
    parser.add_argument('--user', type=str, default='system',
                       help='실행자 ID (로그용)')
    parser.add_argument('--env', type=str, default=None,
                       help='환경 선택 (dev/prod), 미지정시 서버 IP로 자동 감지')
    
    args = parser.parse_args()
    
    config = load_config(args.config, env=args.env)
    
    oracle_client_path = config['oracle'].get('client_path')
    if oracle_client_path:
        init_oracle_client(oracle_client_path)
    
    if args.step is not None:
        start_step = args.step
        end_step = args.step
    else:
        start_step = args.start_step
        end_step = args.end_step
    
    include_step6 = (end_step >= 6)
    output_dirs, timestamp = create_output_dirs(config['paths']['base_path'], include_step6)
    
    oracle_conn = connect_oracle(config)
    pg_conn = connect_postgresql(config)
    
    pipeline_logger = None

    if pg_conn and config.get('pipeline_log', {}).get('enable', False):
        try:
            from src.pipeline_logger import PipelineLogger
            pipeline_logger = PipelineLogger(pg_conn, user_id=args.user, config=config)
            pipeline_logger.start()
        except Exception as e:
            print(f"파이프라인 로거 초기화 실패: {e}")
            pipeline_logger = None

    run_id = pipeline_logger.get_run_id() if pipeline_logger else None
       
    try:
        results = run_pipeline(
            config=config,
            output_dirs=output_dirs,
            oracle_conn=oracle_conn,
            pg_conn=pg_conn,
            timestamp=timestamp,
            run_id=run_id,
            start_step=start_step,
            end_step=end_step,
            pipeline_logger=pipeline_logger
        )
        
        if pipeline_logger:
            pipeline_logger.complete()
    
    except Exception as e:
        if pipeline_logger:
            pipeline_logger.fail(str(e))
        raise
    
    finally:
        if oracle_conn:
            oracle_conn.close()
            print("Oracle DB 연결 종료")
        
        if pg_conn:
            pg_conn.close()
            print("PostgreSQL DB 연결 종료")


if __name__ == "__main__":
    main()
