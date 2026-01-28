#!/usr/bin/env python3
"""Notion TODO 리스트 자동 정리 CLI."""

import argparse
import sys

from notion_todo import (
    add_todo,
    add_todos_bulk,
    append_content_to_page,
    append_todo_block,
    archive_done,
    get_database_schema,
    list_todos,
    list_todos_by_status,
    organize_todos,
    search_databases,
    update_todo_status,
    update_todo_title,
)


def cmd_search(args):
    """워크스페이스에서 데이터베이스를 검색한다."""
    dbs = search_databases(args.query or "")
    if not dbs:
        print("데이터베이스를 찾지 못했습니다.")
        print("Notion Integration이 워크스페이스 페이지에 연결되어 있는지 확인하세요.")
        return
    print(f"{'ID':<40} {'제목'}")
    print("-" * 72)
    for db in dbs:
        print(f"{db['id']:<40} {db['title']}")


def cmd_schema(args):
    """데이터베이스 스키마를 출력한다."""
    schema = get_database_schema(args.database_id)
    print(f"{'속성 이름':<30} {'타입'}")
    print("-" * 50)
    for name, prop_type in schema.items():
        print(f"{name:<30} {prop_type}")


def cmd_list(args):
    """TODO 항목을 조회한다."""
    if args.status:
        todos = list_todos_by_status(args.database_id, args.status, args.status_property)
    else:
        todos = list_todos(args.database_id, args.status_property)

    if not todos:
        print("TODO 항목이 없습니다.")
        return

    print(f"{'상태':<15} {'마감일':<12} {'제목'}")
    print("-" * 65)
    for t in todos:
        status = t["status"] or "-"
        due = t["due_date"] or "-"
        print(f"{status:<15} {due:<12} {t['title']}")


def cmd_add(args):
    """새 TODO 항목을 추가한다."""
    page_id = add_todo(
        database_id=args.database_id,
        title=args.title,
        status=args.status,
        due_date=args.due_date,
        status_property=args.status_property,
    )
    print(f"추가 완료: {page_id}")


def cmd_add_bulk(args):
    """여러 TODO 항목을 한 번에 추가한다 (쉼표로 구분)."""
    titles = [t.strip() for t in args.titles.split(",") if t.strip()]
    items = [{"title": t, "status": args.status} for t in titles]
    ids = add_todos_bulk(args.database_id, items, args.status_property)
    print(f"{len(ids)}개 항목 추가 완료")
    for page_id in ids:
        print(f"  - {page_id}")


def cmd_update(args):
    """TODO 항목을 수정한다."""
    if args.new_status:
        update_todo_status(args.page_id, args.new_status, args.status_property)
        print(f"상태 변경 완료: {args.new_status}")
    if args.new_title:
        update_todo_title(args.page_id, args.new_title)
        print(f"제목 변경 완료: {args.new_title}")


def cmd_organize(args):
    """TODO 리스트를 상태별로 정리하여 요약한다."""
    summary = organize_todos(args.database_id, args.status_property)

    print(f"\n총 {summary['total']}개 항목\n")

    print("[ 상태별 현황 ]")
    for status, count in summary["by_status"].items():
        print(f"  {status}: {count}개")

    if summary["overdue"]:
        print(f"\n⚠ 기한 초과 항목 ({len(summary['overdue'])}개):")
        for t in summary["overdue"]:
            print(f"  - [{t['due_date']}] {t['title']}")

    print("\n[ 상세 목록 ]")
    for status, items in summary["grouped"].items():
        print(f"\n── {status} ──")
        for t in items:
            due = f" ({t['due_date']})" if t["due_date"] else ""
            print(f"  • {t['title']}{due}")


def cmd_archive(args):
    """완료된 항목을 아카이브한다."""
    archived = archive_done(args.database_id, args.status_property, args.done_value)
    print(f"{len(archived)}개 항목 아카이브 완료")


def cmd_append(args):
    """페이지에 텍스트를 추가한다."""
    append_content_to_page(args.page_id, args.content)
    print("내용 추가 완료")


def cmd_add_checklist(args):
    """페이지에 체크리스트를 추가한다."""
    items = [t.strip() for t in args.items.split(",") if t.strip()]
    for item in items:
        append_todo_block(args.page_id, item, checked=False)
    print(f"{len(items)}개 체크리스트 항목 추가 완료")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Notion TODO 리스트 자동 정리 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="사용 가능한 명령어")

    # search
    p = sub.add_parser("search", help="데이터베이스 검색")
    p.add_argument("--query", "-q", default="", help="검색어")

    # schema
    p = sub.add_parser("schema", help="데이터베이스 스키마 조회")
    p.add_argument("database_id", help="데이터베이스 ID")

    # list
    p = sub.add_parser("list", help="TODO 항목 조회")
    p.add_argument("database_id", help="데이터베이스 ID")
    p.add_argument("--status", "-s", help="상태 필터")
    p.add_argument("--status-property", default="Status", help="상태 속성 이름 (기본: Status)")

    # add
    p = sub.add_parser("add", help="TODO 항목 추가")
    p.add_argument("database_id", help="데이터베이스 ID")
    p.add_argument("title", help="할 일 제목")
    p.add_argument("--status", "-s", help="상태")
    p.add_argument("--due-date", "-d", help="마감일 (YYYY-MM-DD)")
    p.add_argument("--status-property", default="Status", help="상태 속성 이름")

    # add-bulk
    p = sub.add_parser("add-bulk", help="여러 TODO 항목 한 번에 추가")
    p.add_argument("database_id", help="데이터베이스 ID")
    p.add_argument("titles", help="할 일 제목 (쉼표로 구분)")
    p.add_argument("--status", "-s", help="상태")
    p.add_argument("--status-property", default="Status", help="상태 속성 이름")

    # update
    p = sub.add_parser("update", help="TODO 항목 수정")
    p.add_argument("page_id", help="페이지 ID")
    p.add_argument("--new-status", help="새 상태")
    p.add_argument("--new-title", help="새 제목")
    p.add_argument("--status-property", default="Status", help="상태 속성 이름")

    # organize
    p = sub.add_parser("organize", help="TODO 리스트 정리 및 요약")
    p.add_argument("database_id", help="데이터베이스 ID")
    p.add_argument("--status-property", default="Status", help="상태 속성 이름")

    # archive
    p = sub.add_parser("archive", help="완료된 항목 아카이브")
    p.add_argument("database_id", help="데이터베이스 ID")
    p.add_argument("--status-property", default="Status", help="상태 속성 이름")
    p.add_argument("--done-value", default="Done", help="완료 상태값 (기본: Done)")

    # append
    p = sub.add_parser("append", help="페이지에 텍스트 추가")
    p.add_argument("page_id", help="페이지 ID")
    p.add_argument("content", help="추가할 텍스트")

    # add-checklist
    p = sub.add_parser("add-checklist", help="페이지에 체크리스트 추가")
    p.add_argument("page_id", help="페이지 ID")
    p.add_argument("items", help="체크리스트 항목 (쉼표로 구분)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "search": cmd_search,
        "schema": cmd_schema,
        "list": cmd_list,
        "add": cmd_add,
        "add-bulk": cmd_add_bulk,
        "update": cmd_update,
        "organize": cmd_organize,
        "archive": cmd_archive,
        "append": cmd_append,
        "add-checklist": cmd_add_checklist,
    }

    try:
        commands[args.command](args)
    except Exception as e:
        print(f"오류 발생: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
