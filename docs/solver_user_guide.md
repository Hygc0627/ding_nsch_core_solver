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
- `write_vtk`
  是否写 VTK/PVD 可视化文件。

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

当前主程序不支持从外部文件读入初始 `c` 场，也不支持 restart。
它只支持两种内置初始化方式。

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

当前边界条件是由 `periodic_x` 和 `periodic_y` 这两个开关控制的。

## 7.1 标量场边界条件

标量场包括：

- 相分数 `c`
- 化学势 `mu`
- 压力 `pressure`
- 其他单元中心标量

边界条件规则：

- 如果某个方向 `periodic = true`，就做周期边界。
- 如果某个方向 `periodic = false`，就对标量场使用零法向梯度边界。

也就是说：

- `c`、`mu`、`pressure` 在非周期边界上当前都是 Neumann 型边界。
- 当前没有接触角模型，也没有指定 Dirichlet 相场边界。

## 7.2 速度边界条件

速度使用 MAC 交错网格：

- `u` 在 x-face。
- `v` 在 y-face。

### x 方向非周期边界

如果

```ini
periodic_x = false
```

则左右边界当前是固壁边界：

- `u = 0`
- `v = 0`

也就是没有穿透，也没有沿壁滑移。

当前不支持设置左右壁面的移动速度。

### y 方向非周期边界

如果

```ini
periodic_y = false
```

则上下边界总是没有法向穿透：

- `v = 0`

而切向速度 `u` 由下面两种情况决定：

- 如果 `top_wall_velocity_x` 和 `bottom_wall_velocity_x` 都是 `0`，则上下壁面是静止 no-slip，`u = 0`。
- 如果它们有非零值，则上下壁面 `u` 被强制成给定的壁面速度。

所以当前最常见的两类设置是：

1. 周期盒子里的静滴

```ini
periodic_x = true
periodic_y = true
```

2. x 周期、y 非周期的平板剪切

```ini
periodic_x = true
periodic_y = false
top_wall_velocity_x = ...
bottom_wall_velocity_x = ...
```

## 7.3 压力边界条件

压力修正方程当前是：

- 周期方向：周期边界。
- 非周期方向：零法向梯度。

也就是说，当前没有显式压力 Dirichlet 边界，也没有开边界条件。

## 8. 求解器相关配置

### 8.1 CH 子问题

当前 CH 使用半隐式离散和稀疏 PCG。

相关配置：

- `ch_inner_iterations`
  CH 线性迭代最大次数。
- `ch_tolerance`
  CH 线性求解容限。
- `stabilization_a1`, `stabilization_a2`
  CH 稳定化参数。

注意：

- CH 线性求解默认优先使用稀疏预条件 PCG。
- 当前代码里 CH 仍保留了对角预条件 fallback。

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

另外还有：

- `petsc_pcg`
- `hydea`

但这两个依赖仓库外部 Python 脚本和外部环境，没有额外准备时不要直接选。

关于残差含义：

- `jacobi`：`pressure_tolerance` 对应的是 Jacobi 迭代中的绝对残差。
- `icpcg` / `ildlt_pcg` / `liu_split_*`：`pressure_tolerance` 对应的是 Krylov 线性求解返回的相对残差。

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
  全部时间步的结构化历史数据。
- `final_cell_fields.csv`
  最终单元中心场，包括 `c`、`rho`、`eta`、`pressure`、`mu`、速度和散度。
- `*.vtk`
  VTK 可视化文件。
- `*.pvd`
  ParaView 时间序列索引。

如果启用了外部压力求解器，还会用到：

- `pressure_solver/`

## 11. 推荐的最小 case 模板

### 11.1 周期静止液滴

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

### 11.2 剪切流下液滴变形

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
