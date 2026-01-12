# PyBaMM DFN 모델 → CasADi 솔버 변환 상세 분석

이 문서는 PyBaMM의 DFN(Doyle-Fuller-Newman) 모델이 CasADi 솔버로 해석되는 과정을 상세히 분석합니다.

## 전체 흐름 요약

```mermaid
flowchart TD
    A["`model = pybamm.lithium_ion.DFN()`"] --> B["`Simulation(model)`"]
    B --> C["`simulation.solve([0, 3600])`"]
    C --> D["`build()` - 모델 빌드"]
    D --> E["`Discretisation.process_model()`"]
    E --> F["`BaseSolver.set_up()`"]
    F --> G["CasADi 표현식 변환"]
    G --> H["`CasadiSolver.create_integrator()`"]
    H --> I["`casadi.integrator()` 호출"]
    I --> J["적분 수행 및 결과 반환"]
    
    style A fill:#e1f5fe
    style G fill:#fff3e0
    style I fill:#e8f5e9
```

---

## 1단계: DFN 모델 정의

**파일**: [dfn.py](file:///c:/Users/Ryu/Python_project/data/PyBaMM-develop/src/pybamm/models/full_battery_models/lithium_ion/dfn.py)

```python
model = pybamm.lithium_ion.DFN()
```

DFN 클래스는 `BaseModel`을 상속하며, 다음 서브모델들을 설정합니다:

| 서브모델 종류 | 설명 |
|-------------|------|
| `set_intercalation_kinetics_submodel()` | Butler-Volmer 반응 속도론 |
| `set_particle_submodel()` | 입자 내 Fickian 확산 |
| `set_solid_submodel()` | 고체상 전위/전류 분포 |
| `set_electrolyte_concentration_submodel()` | 전해질 농도 분포 |
| `set_electrolyte_potential_submodel()` | 전해질 전위 분포 |

각 서브모델은 **수식(equations)**을 PyBaMM의 `Symbol` 표현식 트리로 정의합니다.

---

## 2단계: Simulation 객체 생성

**파일**: [simulation.py](file:///c:/Users/Ryu/Python_project/data/PyBaMM-develop/src/pybamm/simulation.py)

```python
simulation = pybamm.Simulation(model)
```

`Simulation.__init__()` (line 70-152)에서:
- 기본 솔버로 `CasadiSolver`가 설정됨 (`model.default_solver`)
- 기본 지오메트리, 메쉬 타입, 공간 이산화 방법 등이 설정됨

---

## 3단계: solve() 호출 → build() 과정

**파일**: [simulation.py](file:///c:/Users/Ryu/Python_project/data/PyBaMM-develop/src/pybamm/simulation.py) (line 434-612)

```python
simulation.solve([0, 3600])
```

### 3.1 build() 과정

`build()` (line 348-385):

```python
def build(self, initial_soc=None, direction=None, inputs=None):
    self._set_parameters()  # 파라미터 값 적용
    self._mesh = pybamm.Mesh(...)  # 메쉬 생성
    self._disc = pybamm.Discretisation(...)  # 이산화 객체 생성
    self._built_model = self._disc.process_model(...)  # 모델 이산화
```

---

## 4단계: 이산화 (Discretisation)

**파일**: [discretisation.py](file:///c:/Users/Ryu/Python_project/data/PyBaMM-develop/src/pybamm/discretisations/discretisation.py)

`Discretisation.process_model()` (line 117-299)에서:

1. **변수 슬라이스 설정**: 각 변수가 상태 벡터 `y`의 어느 부분에 해당하는지 매핑
2. **경계 조건 처리**: 모든 경계 조건을 이산화
3. **RHS/Algebraic 처리**: 미분 방정식과 대수 방정식을 이산화
4. **초기 조건 처리**: 초기 상태 벡터 계산

> [!IMPORTANT]
> 이산화 후, 모든 수식은 **공간 연산자가 행렬로 대체**되고, **변수가 StateVector로 대체**됩니다.

### 이산화 전 (연속적 수식)
```
∂c/∂t = D * ∇²c  (Fick의 확산 법칙)
```

### 이산화 후 (이산 형태)
```
dc/dt = D * M * c  (M은 2차 미분 행렬)
```

여기서 `c`는 이제 `StateVector(slice(0, n_points))`로 표현됩니다.

---

## 5단계: BaseSolver.set_up() - CasADi 변환의 핵심

**파일**: [base_solver.py](file:///c:/Users/Ryu/Python_project/data/PyBaMM-develop/src/pybamm/solvers/base_solver.py) (line 151-318)

이 단계가 **CasADi 변환의 핵심**입니다.

### 5.1 CasADi 심볼 변수 생성

`_get_vars_for_processing()` (line 424-463):

```python
# CasADi 심볼릭 변수 생성
t_casadi = casadi.MX.sym("t")                    # 시간 변수
y_diff = casadi.MX.sym("y_diff", model.len_rhs)  # 미분 상태 변수
y_alg = casadi.MX.sym("y_alg", model.len_alg)    # 대수 상태 변수
y_casadi = casadi.vertcat(y_diff, y_alg)         # 전체 상태 벡터
p_casadi_stacked = casadi.vertcat(...)           # 입력 파라미터
```

### 5.2 process() 함수로 PyBaMM → CasADi 변환

`set_up()` 내에서 `process()` 함수가 호출됩니다:

```python
# RHS 처리 (미분 방정식 우변)
rhs, jac_rhs, jacp_rhs, jac_rhs_action = process(
    model.concatenated_rhs, "RHS", vars_for_processing
)

# Algebraic 처리 (대수 방정식)
algebraic, jac_algebraic, jacp_algebraic, jac_algebraic_action = process(
    model.concatenated_algebraic, "algebraic", vars_for_processing
)
```

### 5.3 CasADi 함수 생성

line 255-277에서 CasADi 솔버용 함수가 생성됩니다:

```python
# Mass matrix 역행렬 적용하여 explicit RHS 생성
mass_matrix_inv = casadi.MX(model.mass_matrix_inv.entries)
explicit_rhs = mass_matrix_inv @ rhs(t_casadi, y_casadi, p_casadi_stacked)

# CasADi 함수로 래핑
model.casadi_rhs = casadi.Function(
    "rhs", [t_casadi, y_casadi, p_casadi_stacked], [explicit_rhs]
)
model.casadi_algebraic = algebraic  # 대수 방정식도 CasADi 함수
```

---

## 6단계: CasadiConverter - 표현식 트리 변환

**파일**: [convert_to_casadi.py](file:///c:/Users/Ryu/Python_project/data/PyBaMM-develop/src/pybamm/expression_tree/operations/convert_to_casadi.py)

`CasadiConverter.convert()` (line 19-57)와 `_convert()` (line 59-303)가 PyBaMM 표현식 트리를 재귀적으로 탐색하며 CasADi 표현식으로 변환합니다.

### 변환 규칙

| PyBaMM 타입 | CasADi 변환 |
|-------------|------------|
| `Scalar`, `Array` | `casadi.MX(value)` |
| `StateVector` | `y[slice]` (상태 벡터 슬라이스) |
| `BinaryOperator` (+, -, *, /) | CasADi 연산자 |
| `UnaryOperator` (abs, exp, ...) | `casadi.fabs()`, `casadi.exp()` 등 |
| `Function` (sin, cos, exp, ...) | `casadi.sin()`, `casadi.cos()` 등 |
| `Interpolant` | `casadi.interpolant()` 또는 bspline |

### 예시: StateVector 변환

```python
elif isinstance(symbol, pybamm.StateVector):
    if y is None:
        raise ValueError("Must provide a 'y' for converting state vectors")
    return casadi.vertcat(*[y[y_slice] for y_slice in symbol.y_slices])
```

### 예시: BinaryOperator 변환

```python
elif isinstance(symbol, pybamm.BinaryOperator):
    left, right = symbol.children
    converted_left = self.convert(left, t, y, y_dot, inputs)
    converted_right = self.convert(right, t, y, y_dot, inputs)
    return symbol._binary_evaluate(converted_left, converted_right)
```

---

## 7단계: CasadiSolver.create_integrator()

**파일**: [casadi_solver.py](file:///c:/Users/Ryu/Python_project/data/PyBaMM-develop/src/pybamm/solvers/casadi_solver.py) (line 496-598)

이 함수가 실제로 `casadi.integrator()`를 호출합니다.

### 7.1 Problem 정의

```python
# CasADi 심볼릭 변수 설정
t = casadi.MX.sym("t")
p = casadi.MX.sym("p", inputs.shape[0])
y_diff = casadi.MX.sym("y_diff", rhs(0, y0, p).shape[0])
y_alg = casadi.MX.sym("y_alg", algebraic(0, y0, p).shape[0])
y_full = casadi.vertcat(y_diff, y_alg)
```

### 7.2 DAE 문제 정의

```python
problem = {
    "t": t,                    # 시간 변수
    "x": y_diff,              # 미분 상태 변수
    "ode": rhs(t_scaled, y_full, p),  # 미분 방정식 (dy/dt = f(t,y))
    "p": p_with_tlims,        # 파라미터
}

# DAE인 경우 대수 방정식 추가
if algebraic(0, y0, p).is_not_empty():
    method = "idas"  # DAE 솔버
    problem.update({
        "z": y_alg,           # 대수 상태 변수
        "alg": algebraic(...),  # 대수 방정식 (0 = g(t,y,z))
    })
else:
    method = "cvodes"  # ODE 솔버
```

### 7.3 Integrator 생성

```python
options = {
    "reltol": self.rtol,  # 상대 허용오차 (기본 1e-6)
    "abstol": self.atol,  # 절대 허용오차 (기본 1e-6)
}

# CasADi integrator 생성
integrator = casadi.integrator("F", method, problem, *time_args, options)
```

> [!NOTE]
> - **ODE 모델**: CVODES (SUNDIALS의 BDF/Adams 솔버) 사용
> - **DAE 모델**: IDAS (SUNDIALS의 BDF DAE 솔버) 사용

---

## 8단계: 적분 실행

**파일**: [casadi_solver.py](file:///c:/Users/Ryu/Python_project/data/PyBaMM-develop/src/pybamm/solvers/casadi_solver.py) (line 600-726)

`_run_integrator()` 에서 실제 적분이 수행됩니다:

```python
casadi_sol = integrator(
    x0=y0_diff,           # 초기 미분 변수
    z0=y0_alg,            # 초기 대수 변수
    p=inputs_with_tmin,   # 파라미터
)
```

CasADi의 integrator가 반환하는 결과:
- `casadi_sol["xf"]`: 각 시점에서의 미분 상태 벡터
- `casadi_sol["zf"]`: 각 시점에서의 대수 상태 벡터

---

## 핵심 수식 흐름 예시

### DFN 모델의 대표 수식: 전해질 확산

**연속 형태 (물리 수식)**:
$$\varepsilon \frac{\partial c_e}{\partial t} = \nabla \cdot (D_e^{eff} \nabla c_e) + \frac{1-t_+}{F} j$$

**이산화 후 (PyBaMM 표현식)**:
```python
dcdt = (D_eff @ grad_matrix @ c_e) + source_term
# grad_matrix는 유한차분/유한체적 행렬
```

**CasADi 변환 후**:
```python
# t, y, p는 casadi.MX 심볼
dcdt_casadi = D_eff_mx @ grad_matrix_mx @ y[c_e_slice] + source_mx
```

**최종 integrator 호출**:
```python
# problem["ode"]에 dcdt_casadi가 포함됨
casadi.integrator("F", "idas", problem, ...)
```

---

## 디버깅 팁

### 1. 이산화된 모델의 상태 확인
```python
model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model)
sim.build()

# 상태 벡터 크기
print(f"RHS size: {sim._built_model.len_rhs}")
print(f"Algebraic size: {sim._built_model.len_alg}")

# 변수 슬라이스 확인
print(sim._disc.y_slices)
```

### 2. CasADi 함수 확인
```python
sim.solve([0, 3600])
built_model = sim._built_model

# CasADi RHS 함수
print(built_model.casadi_rhs)

# CasADi Algebraic 함수
print(built_model.casadi_algebraic)
```

### 3. Integrator 확인
```python
solver = sim._solver
# integrator specs 확인
print(solver.integrator_specs)
```

---

## 주요 클래스/함수 참조

| 위치 | 역할 |
|-----|-----|
| [DFN](file:///c:/Users/Ryu/Python_project/data/PyBaMM-develop/src/pybamm/models/full_battery_models/lithium_ion/dfn.py) | 모델 정의 |
| [Simulation.solve()](file:///c:/Users/Ryu/Python_project/data/PyBaMM-develop/src/pybamm/simulation.py#L434-L612) | 시뮬레이션 실행 진입점 |
| [Discretisation.process_model()](file:///c:/Users/Ryu/Python_project/data/PyBaMM-develop/src/pybamm/discretisations/discretisation.py#L117-L299) | 모델 이산화 |
| [BaseSolver.set_up()](file:///c:/Users/Ryu/Python_project/data/PyBaMM-develop/src/pybamm/solvers/base_solver.py#L151-L318) | 솔버 셋업 및 CasADi 변환 |
| [CasadiConverter](file:///c:/Users/Ryu/Python_project/data/PyBaMM-develop/src/pybamm/expression_tree/operations/convert_to_casadi.py) | 표현식 트리 → CasADi 변환 |
| [CasadiSolver.create_integrator()](file:///c:/Users/Ryu/Python_project/data/PyBaMM-develop/src/pybamm/solvers/casadi_solver.py#L496-L598) | casadi.integrator() 생성 |
| [CasadiSolver._run_integrator()](file:///c:/Users/Ryu/Python_project/data/PyBaMM-develop/src/pybamm/solvers/casadi_solver.py#L600-L726) | 적분 실행 |
