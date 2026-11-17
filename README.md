# VPINN Burgers study

단위원의 정상상태 2차원 벡터 Burgers 방정식에 대한 VPINN 노트북과 한국어 HTML 자료입니다.

- **HTML:** https://yeongdaejeong.github.io/vpinn-burgers-study/
- **노트북:** [VPINN_sampling_study.ipynb](VPINN_sampling_study.ipynb)

## 내용

기본 Gauss 적분, 적분점 개수·배치 90회 비교, 시험공간 차원·확산계수 45회 비교를 포함합니다. 각 조건은 5개 시드의 평균과 표준오차로 요약합니다. 신경망은 약형 잔차만 학습하며, P3 FEM과 강형 잔차는 독립 검증에 사용합니다.

## 노트북 사용

저장된 출력은 GitHub에서 읽거나 노트북을 내려받아 볼 수 있습니다. 재실행하려면 저장소 전체를 내려받고 이 폴더에서 Python 환경을 준비하세요.

```sh
python -m venv .venv
# Activate the environment for your operating system.
python -m pip install numpy scipy torch matplotlib ipykernel nbformat nbclient
```

VS Code 또는 Jupyter에서 해당 환경을 커널로 선택합니다. 기본 실행은 저장된 Gauss 모델과 내장 P3 참조해를 사용합니다. `RUN_COMPARISON_EXPERIMENTS=False`를 유지하면 배치 실험을 재학습하지 않습니다. 참조해 재계산은 별도 MATLAB 설정이 필요합니다. 확산계수 확장 코드는 `degree_epsilon/`에 있습니다.

## HTML 구성

HTML의 그림·CSS·JavaScript·CSV와 노트북에 필요한 모델·참조 파일을 함께 포함합니다. `field_cases/`는 기존 201×201 시각화 데이터를 조건별로 나눈 파일입니다. 선택한 조건만 불러오며 수치값은 원본과 같습니다. 로컬에서는 `python -m http.server 8000`으로 열 수 있습니다.

## 출처와 범위

참고 노트북은 `Vector_Burgers2D_VPINN_Sine_KO.ipynb`이며 출처와 변경 내역은 [provenance.json](provenance.json)에 있습니다. 실험 사본의 수식·코드·출력을 보존했으며 공개본에서 로컬 컴퓨터 경로만 치환했습니다. 별도의 이용허락을 가정하는 라이선스는 추가하지 않았습니다.

작은 학습 손실은 연속 PDE 정확도의 보장이 아닙니다. P3 참조해도 수치해이며, 표본 최대 오차와 표준오차는 엄밀한 PDE 오차 상한이 아닙니다.
