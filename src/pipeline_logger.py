# -*- coding: utf-8 -*-
"""
파이프라인 로깅 및 상태 관리
"""
from datetime import datetime
from contextlib import contextmanager


CREATE_LOG_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS tb_binzib_pipeline_log(
    run_id          SERIAL PRIMARY KEY,
    id              VARCHAR(50),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status          VARCHAR(50),
    compare_yn      CHAR(1) DEFAULT 'Y',
    del_yn          CHAR(1) DEFAULT 'N',
    error_message   VARCHAR(4000),
    region_base_year VARCHAR(10)
);
"""

COMMENT_SQLS = [
    "COMMENT ON TABLE tb_binzib_pipeline_log IS '빈집 예측 파이프라인 실행 로그'",
    "COMMENT ON COLUMN tb_binzib_pipeline_log.run_id IS '실행 ID(자동 증가)'",
    "COMMENT ON COLUMN tb_binzib_pipeline_log.id IS '실행자 id'",
    "COMMENT ON COLUMN tb_binzib_pipeline_log.created_at IS '시작시간'",
    "COMMENT ON COLUMN tb_binzib_pipeline_log.modified_at IS '종료/변경 시간'",
    "COMMENT ON COLUMN tb_binzib_pipeline_log.status IS '상태 (시작/전처리/파생변수/클러스터/학습/앙상블/예측/완료/실패)'",
    "COMMENT ON COLUMN tb_binzib_pipeline_log.compare_yn IS '비교여부'",
    "COMMENT ON COLUMN tb_binzib_pipeline_log.del_yn IS '삭제여부'",
    "COMMENT ON COLUMN tb_binzib_pipeline_log.error_message IS '에러 메시지'",
    "COMMENT ON COLUMN tb_binzib_pipeline_log.region_base_year IS '집계구 기준연도'"
]


def create_log_table(connection):
    """로그 테이블 생성 (없으면)"""
    if connection is None:
        return False
    
    try:
        cursor = connection.cursor()
        cursor.execute(CREATE_LOG_TABLE_DDL)
        connection.commit()
        
        for comment_sql in COMMENT_SQLS:
            try:
                cursor.execute(comment_sql)
            except:
                pass
        connection.commit()
        
        cursor.close()
        print("[LOG] tb_binzib_pipeline_log 테이블 확인/생성 완료")
        return True
    except Exception as e:
        print(f"[LOG] 테이블 생성 실패: {e}")
        return False


def insert_log(connection, user_id, status, compare_yn='Y', del_yn='N', error_message=None, created_at = None):
    """로그 삽입"""
    if connection is None:
        return None
    
    try:
        cursor = connection.cursor()
        
        sql = """
        INSERT INTO tb_binzib_pipeline_log
            (id, created_at, modified_at, status, compare_yn, del_yn, error_message)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s)
        RETURNING run_id
        """
        
        now = datetime.now()
        if created_at is None:
            created_at = now
        cursor.execute(sql, (user_id, created_at, now, status, compare_yn, del_yn, error_message))
        
        run_id = cursor.fetchone()[0]
        connection.commit()
        cursor.close()
        
        print(f"[LOG] 로그 생성: run_id = {run_id}, status = {status}")
        return run_id
    
    except Exception as e:
        print(f"[LOG] 로그 삽입 실패: {e}")
        return None


def update_log(connection, run_id, status, error_message=None, region_base_year=None):
    """로그 업데이트"""
    if connection is None or run_id is None:
        return False
    
    try:
        cursor = connection.cursor()
        
        if region_base_year is not None:
            sql = """
            UPDATE tb_binzib_pipeline_log
            SET status = %s,
                modified_at = %s,
                error_message = COALESCE(%s, error_message),
                region_base_year = %s
            WHERE run_id = %s
            """
                    
            now = datetime.now()
            cursor.execute(sql, (status, now, error_message, region_base_year, run_id))
        
        else:
            sql = """
            UPDATE tb_binzib_pipeline_log
            SET status = %s,
                modified_at = %s,
                error_message = COALESCE(%s, error_message)
            WHERE run_id = %s
            """
            now = datetime.now()
            cursor.execute(sql, (status, now, error_message, run_id))                    

        
        connection.commit()
        cursor.close()
        
        print(f"[LOG] 로그 업데이트: run_id = {run_id}, status = {status}")
        return True
    
    except Exception as e:
        print(f"[LOG] 로그 업데이트 실패: {e}")
        return False


def get_latest_log(connection, user_id=None):
    """최신 로그 조회"""
    if connection is None:
        return None
    
    try:
        cursor = connection.cursor()
        
        if user_id:
            sql = """
            SELECT run_id, id, created_at, modified_at, status, compare_yn, del_yn, error_message
            FROM tb_binzib_pipeline_log
            WHERE id = %s AND del_yn = 'N'
            ORDER BY run_id DESC
            LIMIT 1
            """
            cursor.execute(sql, (user_id,))
        else:
            sql = """
            SELECT run_id, id, created_at, modified_at, status, compare_yn, del_yn, error_message
            FROM tb_binzib_pipeline_log
            WHERE del_yn = 'N'
            ORDER BY run_id DESC
            LIMIT 1
            """
            cursor.execute(sql)
        
        row = cursor.fetchone()
        cursor.close()
        
        if row:
            return {
                'run_id': row[0],
                'id': row[1],
                'created_at': row[2],
                'modified_at': row[3],
                'status': row[4],
                'compare_yn': row[5],
                'del_yn': row[6],
                'error_message': row[7]
            }
        return None
    
    except Exception as e:
        print(f"[LOG] 로그 조회 실패: {e}")
        return None


class PipelineLogger:
    """파이프라인 로거 클래스"""
    
    def __init__(self, connection, user_id='system', config=None):
        self.connection = connection
        self.user_id = user_id
        self.run_id = None
        
        if config and 'pipeline_log' in config:
            self.status_values = config['pipeline_log'].get('status_values', {})
        else:
            self.status_values = {
                'start': '시작',
                'step1': '전처리',
                'step2': '파생변수',
                'step3': '클러스터',
                'step4': '학습',
                'step5': '예측',
                'step6': '예측(5년)',
                'complete': '완료',
                'failed': '실패'
            }
        
        create_log_table(connection)
    
    def start(self):
        """파이프라인 시작 로그"""
        self.start_time = datetime.now()
        status = self.status_values.get('start', '시작')
        self.run_id = insert_log(self.connection, self.user_id, status, created_at = self.start_time)
        return self.run_id
    
    def update(self, step_key):
        """단계별 상태 업데이트"""
        if self.run_id is None:
            print("[LOG] start() 호출 필요")
            return False
        
        status = self.status_values.get(step_key, step_key)
        return update_log(self.connection, self.run_id, status)
    
    def complete(self):
        """파이프라인 완료"""
        if self.run_id is None:
            return False
        
        status = self.status_values.get('complete', '완료')
        return update_log(self.connection, self.run_id, status)
    
    def fail(self, error_message):
        """파이프라인 실패"""
        if self.run_id is None:
            status = self.status_values.get('failed', '실패')
            self.run_id = insert_log(
                self.connection, self.user_id, status,
                error_message=error_message[:4000] if error_message else None
            )
            return self.run_id is not None
        
        status = self.status_values.get('failed', '실패')
        return update_log(
            self.connection, self.run_id, status,
            error_message=error_message[:4000] if error_message else None
        )
    
    def get_run_id(self):
        """현재 run_id 반환"""
        return self.run_id

    def update_base_year(self, base_year):
        if self.run_id is None:
            return False
        return update_log(self.connection, self.run_id, self.status_values.get('step1', '전처리'), region_base_year=str(base_year))


@contextmanager
def pipeline_logging(connection, user_id='system', config=None):
    """파이프라인 로깅 컨텍스트 매니저"""
    logger = PipelineLogger(connection, user_id, config)
    logger.start()
    
    try:
        yield logger
        logger.complete()
    except Exception as e:
        logger.fail(str(e))
        raise
