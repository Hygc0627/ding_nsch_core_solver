核心求解器代码统一整理在这个目录下。

- `base/`
  配置、均匀网格、场变量和诊断结构等基础对象，供各个子求解器共享。
- `coupled/`
  当前主程序 `ding_nsch` 使用的 CHNS 主求解器，包括时间推进、CH/NS 耦合与 I/O。现在已进一步拆成 `solver.cpp`、`operators.cpp`、`phase_field.cpp`、`momentum.cpp`、`pressure.cpp`、`diagnostics.cpp`、`io.cpp`。
- `linear_algebra/`
  主求解器和 phase2/phase3 共用的稀疏矩阵与 Krylov 线性代数组件。
