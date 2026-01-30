from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os 

# 기본 설정
default_args = {
    'owner': 'binzib',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# DAG 정의
dag = DAG(
    'binzib_pipeline',
    default_args=default_args,
    description='Vacant Housing Prediction Pipeline',
    schedule = None,
    start_date=datetime(2024,1,1),
    catchup=False,
    tags=['binzib', 'ml', 'pipeline'],
    max_active_runs=1,
    max_active_tasks=1,
)

# 파이프라인 경로 (실제 서버 경로)
PIPELINE_PATH = '/home/was/project/binzib-deploy/binzib_pipeline'
PYTHON_PATH = '/home/was/project/.venv/bin/python'
# PYTHON_PATH = '/usr/bin/python3'
CONFIG_PATH = f'{PIPELINE_PATH}/configs/config.yaml'

# 환경변수 (실제 서버 경로)
env_vars = {
    'PYTHONPATH': f'{PIPELINE_PATH}/src',
    'LD_LIBRARY_PATH': '/data/bts-ai-compose/api/embedder/instantclient_23_26',
    'ORACLE_HOME': '/data/bts-ai-compose/api/embedder/instantclient_23_26',
}

def get_merged_config():
    import yaml
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    config = raw.get('common', {})
    config.update(raw.get('prod', {}))
    return config


def init_pipeline(**context):
    import sys
    sys.path.insert(0, f'{PIPELINE_PATH}/src')

    import yaml
    import psycopg2
    from pipeline_logger import PipelineLogger, create_log_table

    config = get_merged_config()

    pg_conn = psycopg2.connect(
        host=config['postgresql']['host'],
        port=config['postgresql']['port'],
        database=config['postgresql']['database'],
        user=config['postgresql']['user'],
        password=config['postgresql']['password']
    )

    logger = PipelineLogger(pg_conn, user_id='airflow', config=config)
    run_id = logger.start()

    pg_conn.close()

    context['ti'].xcom_push(key='run_id', value=run_id)
    print(f"[DAG] Pipeline started with run_id: {run_id}")
    return run_id

def update_step_status(step_key, **context):
    import sys
    sys.path.insert(0, f'{PIPELINE_PATH}/src')

    import yaml
    import psycopg2
    from pipeline_logger import update_log

    run_id = context['ti'].xcom_pull(key='run_id', task_ids='init_pipeline')
    if run_id is None:
        print(f"[DAG] WARN run_id is None")
        return

    config = get_merged_config()

    status_values = config.get('pipeline_log', {}).get('status_values', {
        'start': '시작',
        'step1': '전처리',
        'step2': '파생변수',
        'step3': '클러스터',
        'step4': '학습',
        'step5': '예측',
        'step6': '예측(5년)',
        'complete': '완료',
        'failed': '실패'
    })

    pg_conn = psycopg2.connect(
        host=config['postgresql']['host'],
        port=config['postgresql']['port'],
        database=config['postgresql']['database'],
        user=config['postgresql']['user'],
        password=config['postgresql']['password']
    )

    status = status_values.get(step_key, step_key)
    update_log(pg_conn, run_id, status)

    pg_conn.close()
    print(f"[DAG] Updated {run_id} to {status}")

def complete_pipeline(**context):
    update_step_status('complete', **context)

def fail_pipeline(context):
    import sys
    sys.path.instert(0, f'{PIPELINE_PATH}/src')

    import yaml
    import psycopg2
    from pipeline_logger import update_log

    ti = context.get('ti')
    if ti is None:
        return

    run_id = context['ti'].xcom_pull(key='run_id', task_ids='init_pipeline')
    if run_id is None:
        return

    config = get_merged_config()

    pg_conn = psycopg2.connect(
        host=config['postgresql']['host'],
        port=config['postgresql']['port'],
        database=config['postgresql']['database'],
        user=config['postgresql']['user'],
        password=config['postgresql']['password']
    )

    exception = context.get('exception')
    error_msg = str(exception)[:4000] if exception else 'Unknown error'

    update_log(pg_conn, run_id, '실패', error_message=error_msg)
    pg_conn.close()
    print(f"[DAG] FAILED {error_msg}")

init_task = PythonOperator(
    task_id='init_pipeline',
    python_callable=init_pipeline,
    dag=dag
)

# Step 1: 데이터 통합
step1_data_integration = BashOperator(
    task_id='step1_data_integration',
    bash_command=f'cd {PIPELINE_PATH} && {PYTHON_PATH} main.py --config {CONFIG_PATH} --step 1 --env prod',
    env=env_vars,
    on_failure_callback=fail_pipeline,
    dag=dag,
)

step1_data_integration_up = PythonOperator(
    task_id='step1_data_integration_up',
    python_callable=update_step_status,
    op_kwargs={'step_key':"step1"},
    dag=dag,
)

# Step 2: 피처 엔지니어링
step2_feature_engineering = BashOperator(
    task_id='step2_feature_engineering',
    bash_command=f'cd {PIPELINE_PATH} && {PYTHON_PATH} main.py --config {CONFIG_PATH} --step 2 --env prod',
    env=env_vars,
    on_failure_callback=fail_pipeline,
    dag=dag,
)

step2_feature_engineering_up = PythonOperator(
    task_id='step2_feature_engineering_up',
    python_callable=update_step_status,
    op_kwargs={'step_key':"step2"},
    dag=dag,
)


# Step 3: 클러스터링
step3_clustering = BashOperator(
    task_id='step3_clustering',
    bash_command=f'cd {PIPELINE_PATH} && {PYTHON_PATH} main.py --config {CONFIG_PATH} --step 3 --env prod',
    env=env_vars,
    on_failure_callback=fail_pipeline,
    dag=dag,
)

step3_clustering_up = PythonOperator(
    task_id='step3_clustering_up',
    python_callable=update_step_status,
    op_kwargs={'step_key':"step3"},
    dag=dag,
)

# Step 4: 모델링
step4_modeling = BashOperator(
    task_id='step4_modeling',
    bash_command=f'cd {PIPELINE_PATH} && {PYTHON_PATH} main.py --config {CONFIG_PATH} --step 4 --env prod',
    env=env_vars,
    on_failure_callback=fail_pipeline,
    dag=dag,
)

step4_modeling_up = PythonOperator(
    task_id='step4_modeling_up',
    python_callable=update_step_status,
    op_kwargs={'step_key':"step4"},
    dag=dag,
)
# Step 5: 앙상블 및 예측
step5_ensemble = BashOperator(
    task_id='step5_ensemble',
    bash_command=f'cd {PIPELINE_PATH} && {PYTHON_PATH} main.py --config {CONFIG_PATH} --step 5 --env prod',
    env=env_vars,
    on_failure_callback=fail_pipeline,
    dag=dag,
)

step5_ensemble_up = PythonOperator(
    task_id='step5_ensemble_up',
    python_callable=update_step_status,
    op_kwargs={'step_key':"step5"},
    dag=dag,
)
# Step 6: 예측 (Forecast)
step6_forecast = BashOperator(
    task_id='step6_forecast',
    bash_command=f'cd {PIPELINE_PATH} && {PYTHON_PATH} main.py --config {CONFIG_PATH} --step 6 --env prod',
    env=env_vars,
    on_failure_callback=fail_pipeline,
    dag=dag,
)

step6_forecast_up = PythonOperator(
    task_id='step6_forecast_up',
    python_callable=update_step_status,
    op_kwargs={'step_key':"step6"},
    dag=dag,
)

complete_task = PythonOperator(
    task_id='complete_pipeline',
    python_callable=complete_pipeline,
    dag=dag,
)

# 의존성 설정
(
init_task 
>> step1_data_integration >> step1_data_integration_up 
>> step2_feature_engineering >> step2_feature_engineering_up 
>> step3_clustering >> step3_clustering_up 
>> step4_modeling >> step4_modeling_up 
>>step5_ensemble >> step5_ensemble_up 
>> step6_forecast >> step6_forecast_up 
>> complete_task
)
