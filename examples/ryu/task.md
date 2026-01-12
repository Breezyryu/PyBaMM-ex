분석 목표
PyBaMM 라이브러리의 핵심 구조와 동작 원리를 체계적으로 이해

소스 코드 구조
src/pybamm/
├── __init__.py              # 패키지 진입점, 모든 공개 API
├── simulation.py            # 시뮬레이션 실행 인터페이스
│
├── models/                  # 배터리 모델 정의 (185 files)
├── expression_tree/         # 수식 표현 트리 (33 files)
├── discretisations/         # 공간 이산화
├── spatial_methods/         # 유한체적/유한요소법
├── meshes/                  # 메쉬 생성
├── geometry/                # 배터리 기하학
│
├── solvers/                 # 솔버 (18 files)
├── parameters/              # 파라미터 처리
├── experiment/              # 실험 프로토콜
├── plotting/                # 시각화
└── input/                   # 파라미터 데이터
분석 순서
Phase 1: 핵심 기초 모듈
[/] 1. 패키지 진입점 (
init
.py
)

공개 API 구조
모듈 import 패턴
 2. Expression Tree (expression_tree/)

symbol.py
 - 기본 Symbol 클래스
binary_operators.py - 이항 연산자
unary_operators.py - 단항 연산자
variable.py
 - 변수 표현
state_vector.py - 상태 벡터
 3. Simulation 파이프라인 (
simulation.py
)

Simulation 클래스 구조
solve() 메서드 흐름
Phase 2: 모델 정의
 4. Base Model (models/base_model.py)

BaseModel 클래스
rhs, algebraic, variables 구조
 5. Lithium-Ion Models

base_lithium_ion_model.py
dfn.py
, 
spm.py
서브모델 시스템
Phase 3: 이산화 & 솔버
 6. Discretisation (discretisations/)

공간 이산화 과정
변수 슬라이싱
 7. Spatial Methods (
spatial_methods/
)

FiniteVolume
행렬 생성
 8. Solvers (solvers/)

BaseSolver
CasadiSolver
Phase 4: 파라미터 & 실험
 9. Parameters (
parameters/
)
 10. Experiment (
experiment/
)
현재 진행 상황
Phase 1 시작: init.py 분석