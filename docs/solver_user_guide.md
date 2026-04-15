# Ding NSCH Core Solver 使用说明

这份文档以当前仓库代码的真实行为为准，目标是回答三个问题：

1. 这个求解器现在能做什么。
2. 一个 case 文件该怎么写。
3. 相分数、速度、边界条件这些必须条件应该怎么设置。

## 1. 当前求解器范围

当前仓库是一个二维、均匀笛卡尔网格、MAC 交错网格上的 CHNS 求解器。

当前主流程固定为：

1. 用当前速度场解一次 Cahn-Hilliard 方程。
2. 用新旧相场更新物性并计算表面张力。
3. 解一次动量预测器。
4. 解压力泊松方程。
5. 做一次压力修正，得到新的速度和压力。

也就是说，当前主程序是单次顺序耦合，不是步内多次外迭代的强耦合。

## 2. 构建与运行

构建：

```bash
cmake -S . -B build
cmake --build build -j
```

运行：

```bash
./build/ding_nsch testcases/07_static_droplet_dr100.cfg
```

命令行接口很简单，只接受一个参数：

```text
ding_nsch <case.cfg>
```

退出码：

- `0`：算例完成并通过 `check_*` 验收阈值。
- `2`：算例完成，但未通过 `check_*` 阈值。
- `1`：配置或运行时错误，例如文件打不开、线性求解 breakdown、出现异常。

## 3. case 文件格式

case 文件是最简单的 `key = value` 文本格式。

规则：

- 每行一个配置项。
- `#` 后面的内容会被当作注释去掉。
- 空行会被忽略。
- 没写的项用程序默认值。

示例：

```ini
name = static_droplet_dr100
nx = 64
ny = 64
dt = 1e-6
steps = 100
pressure_scheme = icpcg
periodic_x = true
periodic_y = true
```

## 4. 最常用的配置项

### 4.1 基本控制

- `name`
  case 名字。输出目录会写到 `output_dir/name/`。
- `output_dir`
  输出根目录，默认是 `output`。
- `steps`
  时间步数。
- `dt`
  时间步长。
- `output_every`
  终端输出频率。
- `write_every`
  VTK 写出频率。
- `verbose`
  是否把逐步日志打印到终端。无论这个值是什么，`run.log` 都会写到 case 目录。
- `print_step_log`
  是否按 `output_every` 把时间步日志打印到终端；适合想看 step report，但不想打开其它详细求解器输出的情况。
- `write_vtk`
  是否写 VTK/PVD 可视化文件。
- `write_restart`
  是否写 restart 快照。默认会在 case 目录里维护一个 `restart_latest.bin`。
- `restart_every`
  restart 快照写出频率。`<= 0` 时自动退化为 `write_every`。
- `restart`
  是否从 restart 快照恢复并继续运行。
- `restart_file`
  restart 快照路径。留空时默认读 `<output_dir>/<name>/restart_latest.bin`。

### 4.2 网格与几何

- `nx`, `ny`
  单元数。
- `lx`, `ly`
  物理域长度。
- `ghost`
  ghost cell 层数。

必须注意：

- 当前 WENO5 相场对流需要至少 3 层 ghost cell。
- 除非你明确重写了相场重构部分，否则不要把 `ghost` 设成小于 `3`。

### 4.3 物理参数

- `re`
  Reynolds 数。
- `ca`
  Capillary 数。
- `pe`
  Peclet 数。
- `cn`
  Cahn 数，也控制界面厚度。
- `density_ratio`
  低相相对于参考相的密度比。
- `viscosity_ratio`
  低相相对于参考相的黏度比。
- `body_force_x`, `body_force_y`
  体力项。

当前代码里的物性插值是：

```text
rho(c) = c + (1 - c) * density_ratio
eta(c) = c + (1 - c) * viscosity_ratio
```

所以：

- `c = 1` 对应参考相，密度和黏度都是 `1`。
- `c = 0` 对应另一相，密度和黏度分别是 `density_ratio` 和 `viscosity_ratio`。

## 5. 相分数怎么初始化

当前主程序不支持从外部文件读入任意初始 `c` 场，但现在支持从 restart 快照恢复。

- 不启用 restart 时，只支持下面两种内置初始化方式。
- 启用 restart 时，会跳过内置初始化，直接从快照恢复相场、速度、压力和多步历史量。

### 5.1 圆形液滴初始化

当

```ini
interface_radius > 0
```

时，程序会生成一个以 `(interface_center_x, interface_center_y)` 为圆心、半径为 `interface_radius` 的圆形液滴。

相分数是平滑 `tanh` 过渡：

```text
c(x,y) = 0.5 - 0.5 * tanh((r - R) / (2 * sqrt(2) * cn))
```

其中 `r` 是到圆心的距离，`R` 是 `interface_radius`。

这意味着：

- 圆内 `c` 接近 `1`。
- 圆外 `c` 接近 `0`。
- 界面厚度由 `cn` 控制。

常用配置：

```ini
interface_center_x = 0.5
interface_center_y = 0.5
interface_radius = 0.16
```

### 5.2 正弦扰动平界面初始化

当

```ini
interface_radius <= 0
```

时，程序不再生成圆滴，而是生成一条随 `x` 正弦起伏的水平界面：

```text
y0(x) = interface_center_y
      + interface_amplitude * cos(2*pi*interface_wavenumber*x/lx)
```

相分数是：

```text
c(x,y) = 0.5 - 0.5 * tanh((y - y0(x)) / (2 * sqrt(2) * cn))
```

这意味着：

- 界面下方 `c` 接近 `1`。
- 界面上方 `c` 接近 `0`。

常用配置：

```ini
interface_radius = 0.0
interface_center_y = 0.5
interface_amplitude = 0.02
interface_wavenumber = 1.0
```

### 5.3 初始化相场时的注意事项

- 圆滴最好完整落在计算域内，否则会直接和边界相互作用。
- `cn` 不能大到把整个液滴都抹平，也不能小到界面只有 1 到 2 个网格点。
- 实际上，`cn / dx` 和 `cn / dy` 必须能被网格解析。

上面最后一条是数值建议，不是程序硬检查。

## 6. 速度场怎么初始化

当前速度初始化有两种路径。

### 6.1 均匀平移速度

默认使用：

```ini
advect_u = ...
advect_v = ...
```

这样会把整个初始速度场设成常数：

```text
u = advect_u
v = advect_v
```

如果你想做静止液滴，就设成：

```ini
advect_u = 0.0
advect_v = 0.0
```

### 6.2 Couette 型剪切初始化

如果同时满足下面两个条件：

```ini
periodic_y = false
top_wall_velocity_x != 0 或 bottom_wall_velocity_x != 0
```

程序会自动把初始 `u` 场设成线性 Couette 分布：

```text
u(y) = bottom_wall_velocity_x
     + (top_wall_velocity_x - bottom_wall_velocity_x) * y / ly
```

而 `v` 还是由 `advect_v` 给定，通常设为 `0`。

剪切液滴算例的典型配置：

```ini
periodic_x = true
periodic_y = false
top_wall_velocity_x = 1.0
bottom_wall_velocity_x = -1.0
advect_v = 0.0
```

## 7. 边界条件怎么设置

当前边界条件分成两层：

- `periodic_x` 和 `periodic_y` 仍然负责控制某个方向是否周期。
- 对于非周期方向，现在可以分别给 `u`、`v`、`pressure` 的每一条边设置 `Dirichlet` 或 `Neumann`。

边界类型配置格式是：

```ini
u_bc_left_type = dirichlet
u_bc_left_value = 0.0

v_bc_top_type = neumann
v_bc_top_value = 0.0

pressure_bc_right_type = dirichlet
pressure_bc_right_value = 1.0
```

可用类型只有：

- `dirichlet`
- `neumann`
- `unset`

其中：

- `unset` 表示使用程序默认边界。
- `neumann` 的 `value` 表示外法向导数 `∂/∂n` 的值，不是坐标方向导数。
- 如果某个方向已经是周期边界，则该方向对应的显式边界配置不能再设置，程序会直接报错。

## 7.1 标量场边界条件

标量场包括：

- 相分数 `c`
- 化学势 `mu`
- 压力 `pressure`
- 其他单元中心标量

边界条件规则：

- 如果某个方向 `periodic = true`，就做周期边界。
- 如果某个方向 `periodic = false`，则：
  - `pressure` 可以按边指定 `Dirichlet` 或 `Neumann`
  - 其他标量场当前仍然固定使用零法向梯度边界

也就是说：

- `c`、`mu`、`rho`、`eta` 在非周期边界上当前仍然是 Neumann 型边界。
- `pressure` 默认也是零法向梯度，但你现在可以覆盖成任意按边的 `Dirichlet/Neumann`。
- 当前仍然没有接触角模型，也没有指定 Dirichlet 相场边界。

## 7.2 速度边界条件

速度使用 MAC 交错网格：

- `u` 在 x-face。
- `v` 在 y-face。

对于速度，现在可以在四条边分别设置：

- `u_bc_left/right/bottom/top_type`
- `u_bc_left/right/bottom/top_value`
- `v_bc_left/right/bottom/top_type`
- `v_bc_left/right/bottom/top_value`

如果你不显式设置，默认行为是：

- `u`
  - 左右边界默认 `Dirichlet 0`
  - 上下边界默认 `Dirichlet bottom_wall_velocity_x / top_wall_velocity_x`
- `v`
  - 四条边默认 `Dirichlet 0`

也就是说：

- 旧配置里的 `top_wall_velocity_x` 和 `bottom_wall_velocity_x` 仍然有效，但现在只是 `u` 在上下边界的默认值来源。
- 如果你显式写了 `u_bc_bottom_*` 或 `u_bc_top_*`，它们会覆盖 `bottom_wall_velocity_x/top_wall_velocity_x`。

一个简单例子：

```ini
periodic_x = false
periodic_y = false

u_bc_left_type = dirichlet
u_bc_left_value = 0.0
u_bc_right_type = dirichlet
u_bc_right_value = 0.0
u_bc_bottom_type = dirichlet
u_bc_bottom_value = -1.0
u_bc_top_type = dirichlet
u_bc_top_value = 1.0

v_bc_left_type = dirichlet
v_bc_left_value = 0.0
v_bc_right_type = dirichlet
v_bc_right_value = 0.0
v_bc_bottom_type = dirichlet
v_bc_bottom_value = 0.0
v_bc_top_type = dirichlet
v_bc_top_value = 0.0
```

所以当前最常见的几类设置仍然是：

1. 周期盒子里的静滴

```ini
periodic_x = true
periodic_y = true
```

2. x 周期、y 非周期的平板剪切

```ini
periodic_x = true
periodic_y = false
u_bc_bottom_type = dirichlet
u_bc_bottom_value = ...
u_bc_top_type = dirichlet
u_bc_top_value = ...
v_bc_bottom_type = dirichlet
v_bc_bottom_value = 0.0
v_bc_top_type = dirichlet
v_bc_top_value = 0.0
```

## 7.3 压力边界条件

压力现在也可以按边设置：

- `pressure_bc_left/right/bottom/top_type`
- `pressure_bc_left/right/bottom/top_value`

默认行为是：

- 周期方向：周期边界
- 非周期方向：`Neumann 0`

也就是说：

- 现在已经支持显式压力 `Dirichlet` 和 `Neumann`
- 但仍然没有通用开边界模型；如果你要做 outflow，需要你自己把它等价成合适的 `pressure` 和 `velocity` 边界组合

## 8. 求解器相关配置

### 8.1 CH 子问题

当前 CH 使用半隐式离散和稀疏 PCG。

相关配置：

- `ch_inner_iterations`
  CH 线性迭代最大次数。
- `ch_tolerance`
  CH 线性求解容限。

注意：

- CH 线性求解默认优先使用稀疏预条件 PCG。
- 当前代码里 CH 仍保留了对角预条件 fallback。
- CH 稳定化系数 `a1`、`a2` 由 `cn` 在代码内部自动推导，不再作为 case 配置项暴露。
- 若 case 文件仍写入 `stabilization_a1` 或 `stabilization_a2`，程序会直接报错，避免误以为它们仍然生效。

### 8.2 动量预测器

当前动量预测器是：

- 对流项显式。
- 整个黏性张量全隐式。
- `u-v` 联立块系统。
- `BiCGSTAB + 对角预条件`。

相关配置：

- `momentum_iterations`
  动量隐式求解最大迭代数。
- `momentum_tolerance`
  动量隐式求解的绝对残差容限。

这里要特别说明：

- 当前 `momentum_tolerance` 的语义已经是绝对残差。
- `run.log` 里会写成 `momentum_abs_tol`。

### 8.3 压力泊松方程

当前内置可用的 `pressure_scheme`：

- `jacobi`
- `icpcg`
- `ildlt_pcg`
- `liu_split_icpcg`
- `liu_split_ildlt_pcg`
- `petsc_pcg`
- `hydea`

其中：

- `petsc_pcg` 现在已经随仓库提供了 Python 驱动脚本：
  - `python/petsc_pressure_solver.py`
  - `python/petsc_pressure_options.py`
- `hydea` 仍然依赖仓库外部脚本和模型文件

如果你要用 `petsc_pcg`，通常需要在配置里补这项：

```ini
petsc_python_executable = /path/to/python/with/petsc4py
```

如果你想用自定义 KSP/PC 选项，可以在配置里改：

```ini
petsc_solver_script = python/petsc_pressure_solver.py
petsc_solver_config = python/petsc_pressure_options.py
```

然后在 `petsc_pressure_options.py` 里指定，例如：

```python
PETSCPRESSURE_OPTIONS = {
    "ksp_type": "cg",
    "pc_type": "icc",
    "norm_type": "unpreconditioned",
    "rtol": 0.0,
    "atol": 1.0e-6,
    "max_it": 1000,
    "monitor": True,
    "monitor_stdout": False,
}
```

程序会自动根据当前压力边界类型决定是否给 PETSc 设置常数零空间：

- 全 Neumann / 周期压力边界：使用常数零空间
- 只要存在压力 Dirichlet：不使用常数零空间

关于残差含义：

- `jacobi`：`pressure_tolerance` 对应的是 Jacobi 迭代中的绝对残差。
- `icpcg` / `ildlt_pcg` / `liu_split_*`：`pressure_tolerance` 对应的是 Krylov 线性求解返回的相对残差。
- `petsc_pcg`：最终记录的是 PETSc 报告文件里的残差范数，具体意义由 Python 配置中的 `norm_type` 决定。

## 9. 验收阈值与失败条件

这些参数不会改变方程，只用于最终 PASS/FAIL 判断：

- `check_mass_drift_max`
- `check_divergence_max`
- `check_mu_max`
- `check_velocity_max`

程序结束时会检查最终一步的诊断量：

- 如果都在阈值内，返回 `PASS`。
- 如果算完了但超阈值，返回 `FAIL`，进程退出码是 `2`。

真正会让程序中途停掉的是：

- 出现非有限数。
- 配置文件错误。
- 线性求解 breakdown。
- 其他抛出的异常。

## 10. 输出文件说明

每个 case 默认输出到：

```text
<output_dir>/<name>/
```

常见文件：

- `run.log`
  完整逐步日志。包含每一步的时间、时间步长、CH/动量/压力残差、迭代次数、求解器名字、散度误差等。
- `summary.csv`
  最终一步汇总。
- `history.csv`
  全部时间步的结构化历史数据。现在按步持续落盘，方便 crash 后继续保留已完成历史。
- `final_cell_fields.csv`
  最终单元中心场，包括 `c`、`rho`、`eta`、`pressure`、`mu`、速度和散度。
- `restart_latest.bin`
  最近一次 restart 快照。包含继续时间推进所需的当前场和多步历史量。
- `*.vtk`
  VTK 可视化文件。
- `*.pvd`
  ParaView 时间序列索引。

如果启用了外部压力求解器，还会用到：

- `pressure_solver/`

## 11. Restart 用法

最简单的继续运行方式是保持同一个 `name` 和 `output_dir`，然后把 `steps` 改大：

```ini
name = my_case
output_dir = output
steps = 200000
write_restart = true
restart_every = 500
```

第一次正常起算：

```ini
restart = false
```

中断后继续：

```ini
restart = true
restart_file =
```

这时程序会默认读取：

```text
<output_dir>/<name>/restart_latest.bin
```

如果你想从别的位置恢复，也可以显式指定：

```ini
restart = true
restart_file = /abs/path/to/restart_latest.bin
```

restart 时允许你修改：

- `steps`
- `output_every`
- `write_every`
- `restart_every`
- `verbose`
- `write_vtk`
- `pressure_scheme`

restart 时不应该修改：

- `nx`, `ny`, `ghost`
- `dt`
- `lx`, `ly`
- `re`, `ca`, `pe`, `cn`
- `density_ratio`, `viscosity_ratio`
- `periodic_x`, `periodic_y`
- `body_force_x`, `body_force_y`
- `top_wall_velocity_x`, `bottom_wall_velocity_x`
- 所有 `u_bc_*`、`v_bc_*`、`pressure_bc_*` 的 `type/value`

这些量如果和快照不一致，程序会直接拒绝加载。

## 12. 推荐的最小 case 模板

### 12.1 周期静止液滴

```ini
name = my_static_droplet
mode = coupled
nx = 64
ny = 64
ghost = 3
steps = 100
dt = 1e-6
lx = 1.0
ly = 1.0

re = 50.0
ca = 1.0
pe = 150.0
cn = 0.02
density_ratio = 0.1
viscosity_ratio = 0.1

interface_center_x = 0.5
interface_center_y = 0.5
interface_radius = 0.16

periodic_x = true
periodic_y = true
advect_u = 0.0
advect_v = 0.0

pressure_scheme = icpcg
poisson_iterations = 1000
pressure_tolerance = 1e-10
momentum_iterations = 100
momentum_tolerance = 1e-8
ch_inner_iterations = 30
ch_tolerance = 1e-8

write_vtk = false
verbose = false
output_dir = output
```

### 12.2 剪切流下液滴变形

```ini
name = my_shear_droplet
mode = coupled
nx = 64
ny = 64
ghost = 3
steps = 10000
dt = 2e-5
lx = 1.0
ly = 1.0

re = 0.1
ca = 0.45
pe = 150.0
cn = 0.02
density_ratio = 1.0
viscosity_ratio = 1.0

interface_center_x = 0.5
interface_center_y = 0.5
interface_radius = 0.25

periodic_x = true
periodic_y = false
top_wall_velocity_x = 1.0
bottom_wall_velocity_x = -1.0

pressure_scheme = jacobi
poisson_iterations = 200
pressure_tolerance = 1e-10

write_vtk = true
verbose = false
output_dir = output
```

### 12.3 单相流基线

```ini
name = my_single_phase
mode = single_phase
nx = 64
ny = 64
ghost = 3
steps = 100
dt = 1e-4
lx = 1.0
ly = 1.0

re = 100.0
ca = 1.0
pe = 150.0
cn = 0.02

# single_phase 模式下相场保持常数，不再求解 CH，也不施加界面张力。
# 默认使用 c=1 的参考流体；若想使用另一侧常物性流体，可设 invert_phase = true。
density_ratio = 0.1
viscosity_ratio = 0.1
invert_phase = false

periodic_x = false
periodic_y = false
top_wall_velocity_x = 1.0
bottom_wall_velocity_x = 0.0

pressure_scheme = icpcg
poisson_iterations = 1000
pressure_tolerance = 1e-10
momentum_iterations = 100
momentum_tolerance = 1e-8

write_vtk = false
verbose = false
output_dir = output
```

也可以用更通用的冻结相场写法：

```ini
mode = coupled
freeze_ch = true
```

这会停止 CH 时间推进，但仍然保留当前相场对 `rho`、`eta`、`mu` 和表面张力项的影响。若目标只是单相流，继续使用均匀初值即可；此时它与 `single_phase` 的结果一致，但更方便后续扩展到“冻结界面”的流动算例。

## 12. 当前限制

这部分很重要，避免把仓库当前能力想得比实际更完整。

- 只支持二维均匀笛卡尔网格。
- 只支持内置初始化，不支持从文件读取初值，不支持重启。
- 当前没有接触角、润湿、开边界、入口出口、左右移动壁面。
- 当前没有 3D、AMR、非结构网格。
- `mode` 字段目前只参与日志和相场初始化分支，不会切换成真正独立的 `NS-only` 或 `CH-only` 主流程。
- `coupling_iterations` 和 `coupling_tolerance` 当前只记录在日志里，主程序仍然只做一次外层顺序耦合。
- `momentum_advection_scheme` 这个配置项目前会被解析和记录，但主动量对流主路径仍然使用 centered finite difference。

## 13. 建议的使用顺序

如果你第一次用这个求解器，建议按这个顺序：

1. 先跑 `testcases/04_coupled_capillary_smoke.cfg`，确认编译和运行链路是通的。
2. 再跑 `05/06/07_static_droplet_*.cfg`，确认静滴在你机器上稳定。
3. 最后再改成你自己的几何、密度比和边界条件。

这样排查最快。
