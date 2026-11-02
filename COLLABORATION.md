# 공동작업 안내

저장소: https://github.com/Gyuhyeon-Eom/vacant-house-prediction

## 실행과 구성

기존 ML/Airflow 빈집 예측 파이프라인입니다. 현재 `binzib.tridatum.co` 서비스 저장소와 별개입니다. 실행 및 데이터 준비는 `README.md`를 따릅니다.

## 작업 규칙

- `main`에서 작업 브랜치를 만들고 Pull Request로 검토 후 병합합니다.
- API 키·접속 코드·SSH 키·`.env`·운영 DB는 저장소나 이슈에 올리지 않습니다. 필요한 값은 권한 있는 비밀정보 공유 수단으로 전달합니다.
- GitHub 게시와 운영 배포는 별개입니다. 이 소스 정리 작업에서 운영 배포·데이터 갱신·실험 브랜치 병합은 수행하지 않았습니다.
- 비공개 저장소는 소유자가 GitHub Settings → Collaborators에서 작업자를 초대해야 접근할 수 있습니다. 공개 저장소도 쓰기 권한은 별도 초대가 필요합니다.
- 기존 코드 및 개발 중 변경을 보존한 협업 스냅샷이며, 게시 자체가 모든 기능의 재검증을 의미하지 않습니다.
