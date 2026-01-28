"""Notion TODO List 자동 정리 클라이언트 모듈."""

import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from notion_client import Client

load_dotenv()

notion = Client(auth=os.getenv("NOTION_API_KEY"))


# ──────────────────────────────────────────────
# 데이터베이스 탐색
# ──────────────────────────────────────────────
def search_databases(query: str = ""):
    """Notion 워크스페이스에서 데이터베이스를 검색한다."""
    results = notion.search(
        query=query,
        filter={"property": "object", "value": "database"},
    )
    databases = []
    for db in results.get("results", []):
        title_parts = db.get("title", [])
        title = "".join(t.get("plain_text", "") for t in title_parts)
        databases.append({"id": db["id"], "title": title})
    return databases


def get_database_schema(database_id: str):
    """데이터베이스의 속성(컬럼) 스키마를 가져온다."""
    db = notion.databases.retrieve(database_id=database_id)
    schema = {}
    for name, prop in db.get("properties", {}).items():
        schema[name] = prop["type"]
    return schema


# ──────────────────────────────────────────────
# TODO 항목 조회
# ──────────────────────────────────────────────
def list_todos(database_id: str, status_property: str = "Status"):
    """데이터베이스에서 TODO 항목을 조회한다.

    Args:
        database_id: Notion 데이터베이스 ID
        status_property: 상태를 나타내는 속성 이름 (기본: 'Status')
    """
    response = notion.databases.query(database_id=database_id)
    todos = []
    for page in response.get("results", []):
        props = page.get("properties", {})

        # 제목 추출
        title = _extract_title(props)

        # 상태 추출
        status = _extract_status(props, status_property)

        # 날짜 추출 (있으면)
        due_date = _extract_date(props)

        todos.append({
            "id": page["id"],
            "title": title,
            "status": status,
            "due_date": due_date,
            "created": page.get("created_time", ""),
            "last_edited": page.get("last_edited_time", ""),
        })
    return todos


def list_todos_by_status(database_id: str, status_value: str, status_property: str = "Status"):
    """특정 상태의 TODO 항목만 필터링하여 조회한다."""
    response = notion.databases.query(
        database_id=database_id,
        filter={
            "property": status_property,
            "status": {"equals": status_value},
        },
    )
    return _parse_pages(response.get("results", []), status_property)


# ──────────────────────────────────────────────
# TODO 항목 생성
# ──────────────────────────────────────────────
def add_todo(database_id: str, title: str, status: str | None = None,
             due_date: str | None = None, status_property: str = "Status"):
    """새 TODO 항목을 추가한다.

    Args:
        database_id: Notion 데이터베이스 ID
        title: 할 일 제목
        status: 상태값 (예: 'Not started', 'In progress', 'Done')
        due_date: 마감일 (YYYY-MM-DD)
        status_property: 상태 속성 이름
    """
    properties = {
        "title": {"title": [{"text": {"content": title}}]},
    }

    if status:
        properties[status_property] = {"status": {"name": status}}

    if due_date:
        # 날짜 속성 이름 탐색
        date_prop = _find_date_property(database_id)
        if date_prop:
            properties[date_prop] = {"date": {"start": due_date}}

    page = notion.pages.create(
        parent={"database_id": database_id},
        properties=properties,
    )
    return page["id"]


def add_todos_bulk(database_id: str, items: list[dict], status_property: str = "Status"):
    """여러 TODO 항목을 한 번에 추가한다.

    Args:
        items: [{"title": "...", "status": "...", "due_date": "..."}, ...]
    """
    created_ids = []
    for item in items:
        page_id = add_todo(
            database_id=database_id,
            title=item["title"],
            status=item.get("status"),
            due_date=item.get("due_date"),
            status_property=status_property,
        )
        created_ids.append(page_id)
    return created_ids


# ──────────────────────────────────────────────
# TODO 항목 수정
# ──────────────────────────────────────────────
def update_todo_status(page_id: str, new_status: str, status_property: str = "Status"):
    """TODO 항목의 상태를 변경한다."""
    notion.pages.update(
        page_id=page_id,
        properties={
            status_property: {"status": {"name": new_status}},
        },
    )


def update_todo_title(page_id: str, new_title: str):
    """TODO 항목의 제목을 변경한다."""
    notion.pages.update(
        page_id=page_id,
        properties={
            "title": {"title": [{"text": {"content": new_title}}]},
        },
    )


# ──────────────────────────────────────────────
# TODO 정리 (자동)
# ──────────────────────────────────────────────
def organize_todos(database_id: str, status_property: str = "Status"):
    """TODO 리스트를 상태별로 정리하여 요약을 반환한다."""
    todos = list_todos(database_id, status_property)

    grouped: dict[str, list] = {}
    for todo in todos:
        status = todo["status"] or "No Status"
        grouped.setdefault(status, []).append(todo)

    # 마감일 기준 정렬
    for status_group in grouped.values():
        status_group.sort(key=lambda t: t["due_date"] or "9999-12-31")

    summary = {
        "total": len(todos),
        "by_status": {k: len(v) for k, v in grouped.items()},
        "grouped": grouped,
        "overdue": _find_overdue(todos),
    }
    return summary


def archive_done(database_id: str, status_property: str = "Status", done_value: str = "Done"):
    """완료된 항목을 아카이브(삭제) 처리한다."""
    done_todos = list_todos_by_status(database_id, done_value, status_property)
    archived = []
    for todo in done_todos:
        notion.pages.update(page_id=todo["id"], archived=True)
        archived.append(todo["id"])
    return archived


# ──────────────────────────────────────────────
# 페이지 내용 추가
# ──────────────────────────────────────────────
def append_content_to_page(page_id: str, content: str):
    """기존 페이지에 텍스트 블록을 추가한다."""
    notion.blocks.children.append(
        block_id=page_id,
        children=[
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": content}}]
                },
            }
        ],
    )


def append_todo_block(page_id: str, text: str, checked: bool = False):
    """페이지에 체크리스트(to_do 블록)를 추가한다."""
    notion.blocks.children.append(
        block_id=page_id,
        children=[
            {
                "object": "block",
                "type": "to_do",
                "to_do": {
                    "rich_text": [{"type": "text", "text": {"content": text}}],
                    "checked": checked,
                },
            }
        ],
    )


# ──────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────
def _extract_title(props: dict) -> str:
    for prop in props.values():
        if prop.get("type") == "title":
            parts = prop.get("title", [])
            return "".join(p.get("plain_text", "") for p in parts)
    return ""


def _extract_status(props: dict, status_property: str) -> str | None:
    prop = props.get(status_property)
    if not prop:
        return None
    if prop["type"] == "status":
        status_obj = prop.get("status")
        return status_obj.get("name") if status_obj else None
    if prop["type"] == "select":
        select_obj = prop.get("select")
        return select_obj.get("name") if select_obj else None
    return None


def _extract_date(props: dict) -> str | None:
    for prop in props.values():
        if prop.get("type") == "date":
            date_obj = prop.get("date")
            if date_obj:
                return date_obj.get("start")
    return None


def _find_date_property(database_id: str) -> str | None:
    schema = get_database_schema(database_id)
    for name, prop_type in schema.items():
        if prop_type == "date":
            return name
    return None


def _find_overdue(todos: list[dict]) -> list[dict]:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    overdue = []
    for todo in todos:
        if (
            todo["due_date"]
            and todo["due_date"] < today
            and todo["status"] not in ("Done", "완료")
        ):
            overdue.append(todo)
    return overdue


def _parse_pages(pages: list, status_property: str) -> list[dict]:
    results = []
    for page in pages:
        props = page.get("properties", {})
        results.append({
            "id": page["id"],
            "title": _extract_title(props),
            "status": _extract_status(props, status_property),
            "due_date": _extract_date(props),
            "created": page.get("created_time", ""),
            "last_edited": page.get("last_edited_time", ""),
        })
    return results
