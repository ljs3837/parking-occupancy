# 주차면 점유 관제 프로젝트

여러 CCTV에서 같은 주차면을 하나로 묶고 점유와 빈자리를 표시하는 4인 팀 프로젝트입니다.

## 담당 영역

| 담당 | 폴더 | 첫 결과 |
|---|---|---|
| 인식·추적 | src/parking/perception | 한 카메라의 점유 결과 |
| 매칭·통합 | src/parking/fusion, configs | 매핑표와 통합 상태 |
| 관제·UI | src/parking/app, src/parking/backend | PyQt 통합 UI·저장·현황 |
| 정답·검증 | evaluation | 정답과 평가 결과 |

## 개발 순서

1. 카메라·주차면 수, 점유 정의와 정답 기준 확정
2. 단일 카메라 점유 판정과 샘플 기반 관제 개발
3. 입출차·가림 처리
4. 여러 카메라 통합
5. 전체 연결과 평가

UI는 `src/parking/app`(PyQt6)에서 통합합니다. 인식·서버는 Python 기준입니다.
설정은 설명용 예시입니다. 실제 배치에 맞게 수정하세요.
`.env.example`을 참고해 로컬 `.env`를 작성합니다. 영상과 모델은 별도로 공유합니다.

요구사항과 규격은 docs/, 협업 규칙은 CONTRIBUTING.md를 참고하세요.

## 앱 실행

Python **3.11** Conda 환경 `parking` 기준입니다.

```bash
conda activate parking
python src/main.py
```

PyQt6 통합 앱입니다. 색인·(예정)관제·학습을 한 실행 파일에서 엽니다.

## 개발 환경 설치

Anaconda 또는 Miniconda 터미널에서 프로젝트 폴더로 이동한 뒤 실행합니다.
이미 `parking` 환경이 있으면 생성 명령을 생략하고 패키지 설치만 하면 됩니다.

```bash
conda create -n parking python=3.11 -y
conda activate parking
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python --version   # 3.11.x 확인
python -m pip list
```

패키지 목록은 `requirements.txt`에서 관리하며 버전은 아직 고정하지 않았습니다.
