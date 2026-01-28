# Notion TODO 자동 정리 도구

Notion API를 사용하여 TODO 리스트를 자동으로 정리하는 CLI 도구.

## 설치

```bash
pip install -r requirements.txt
```

## 설정

`.env` 파일에 Notion API 키를 설정:

```
NOTION_API_KEY=your_notion_api_key_here
```

> Notion Integration을 생성한 뒤, 사용할 페이지/데이터베이스에 해당 Integration을 연결해야 합니다.

## 사용법

### 데이터베이스 검색
```bash
python main.py search
python main.py search -q "TODO"
```

### 스키마 확인
```bash
python main.py schema <database_id>
```

### TODO 조회
```bash
python main.py list <database_id>
python main.py list <database_id> --status "In progress"
```

### TODO 추가
```bash
python main.py add <database_id> "새로운 할 일"
python main.py add <database_id> "긴급 작업" --status "In progress" --due-date 2026-02-01
```

### TODO 일괄 추가
```bash
python main.py add-bulk <database_id> "할일1, 할일2, 할일3"
```

### TODO 수정
```bash
python main.py update <page_id> --new-status "Done"
python main.py update <page_id> --new-title "변경된 제목"
```

### TODO 자동 정리 (상태별 요약)
```bash
python main.py organize <database_id>
```

### 완료 항목 아카이브
```bash
python main.py archive <database_id>
```

### 페이지에 내용 추가
```bash
python main.py append <page_id> "추가할 텍스트"
python main.py add-checklist <page_id> "항목1, 항목2, 항목3"
```

## 모듈로 사용

```python
from notion_todo import search_databases, list_todos, add_todo, organize_todos

# 데이터베이스 검색
dbs = search_databases()

# TODO 정리
summary = organize_todos(database_id="your-db-id")
print(summary)
```
