当前主求解器 `ding_nsch` 的核心实现按职责拆成了这些文件：

- `solver.hpp`
  只保留求解器状态和接口声明；`Config`、`Grid`、`Field2D`、`Diagnostics` 已转移到 `src/core/base/`。
- `solver.cpp`
  状态构造、初始化、时间步推进总控和 `run()` 主循环。
- `operators.cpp`
  边界条件、差分/插值算子、密度黏度面值、耦合残差等基础几何与离散工具。
- `phase_field.cpp`
  Cahn-Hilliard 子问题、化学势、表面张力、WENO 相场通量和 CH Krylov 线性求解。
- `momentum.cpp`
  动量对流、全黏性张量离散和单体块 BiCGSTAB 速度预测器。
- `pressure.cpp`
  压力修正求解器与速度修正，包括 Jacobi、ICCPCG、ILDLTPCG、Liu split、PETSc、HyDEA。
- `diagnostics.cpp`
  质量、自由能、散度、速度等诊断量计算与终端打印。
- `io.cpp`
  VTK/PVD、summary/history CSV、最终场输出。
- `internal.hpp`
  仅供这些实现文件共享的小型内部工具函数。
