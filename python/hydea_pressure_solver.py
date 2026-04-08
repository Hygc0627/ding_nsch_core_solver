#!/usr/bin/env python3

import argparse
import contextlib
import importlib
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import torch
from petsc4py import PETSc


def load_options(path: Path):
    spec = importlib.util.spec_from_file_location("hydea_pressure_options", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load HyDEA options file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    options = getattr(module, "HYDEA_OPTIONS", None)
    if options is None:
        raise RuntimeError("HYDEA_OPTIONS dict missing in HyDEA options file")
    return options


def read_triplets(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().split()
        nrows, ncols, nnz = map(int, header)
        rows = []
        cols = []
        vals = []
        for line in handle:
            row, col, value = line.split()
            rows.append(int(row))
            cols.append(int(col))
            vals.append(float(value))
    if len(rows) != nnz:
        raise RuntimeError(f"triplet count mismatch: expected {nnz}, got {len(rows)}")
    return nrows, ncols, rows, cols, vals


def read_vector(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        n = int(handle.readline().strip())
        values = [float(line.strip()) for line in handle if line.strip()]
    if len(values) != n:
        raise RuntimeError(f"vector length mismatch: expected {n}, got {len(values)}")
    return np.asarray(values, dtype=np.float64)


def resolve_norm_type(name: str):
    mapping = {
        "default": PETSc.KSP.NormType.DEFAULT,
        "none": PETSc.KSP.NormType.NONE,
        "preconditioned": PETSc.KSP.NormType.PRECONDITIONED,
        "unpreconditioned": PETSc.KSP.NormType.UNPRECONDITIONED,
        "natural": PETSc.KSP.NormType.NATURAL,
    }
    key = name.strip().lower()
    if key not in mapping:
        raise RuntimeError(f"unsupported PETSc norm type: {name}")
    return mapping[key]


def build_petsc_matrix(nrows, ncols, rows, cols, vals):
    mat = PETSc.Mat().createAIJ([nrows, ncols], comm=PETSc.COMM_SELF)
    mat.setUp()
    for row, col, value in zip(rows, cols, vals):
        mat.setValue(row, col, value, addv=True)
    mat.assemblyBegin()
    mat.assemblyEnd()
    return mat


def build_petsc_vector(values):
    vec = PETSc.Vec().createSeq(len(values), comm=PETSc.COMM_SELF)
    vec.setValues(range(len(values)), values)
    vec.assemblyBegin()
    vec.assemblyEnd()
    return vec


def load_hydea_model(example_dir: Path, model_path: Path, grid_nx: int, grid_ny: int, use_cuda: bool):
    if grid_nx != 192 or grid_ny != 192:
        raise RuntimeError(
            f"HyDEA local model is hard-wired for 192x192, got {grid_nx}x{grid_ny}"
        )

    if not example_dir.exists():
        raise RuntimeError(f"HyDEA example directory not found: {example_dir}")
    if not model_path.exists():
        raise RuntimeError(f"HyDEA model weights not found: {model_path}")

    original_argv = sys.argv[:]
    sys.path.insert(0, str(example_dir))
    sys.argv = [original_argv[0]]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            get_params = importlib.import_module("get_params")
            fluidmodel = importlib.import_module("fluidmodel")
        params = get_params.params
        params.cuda = bool(use_cuda and torch.cuda.is_available())
        params.imagexsize = grid_nx
        params.imageysize = grid_ny

        model = fluidmodel.get_Net(params)
        device = torch.device("cuda" if params.cuda else "cpu")
        model = model.to(device)
        state = torch.load(model_path, map_location=device)
        if "model0" not in state:
            raise RuntimeError(f"HyDEA state file missing model0: {model_path}")
        model.load_state_dict(state["model0"])
        model.eval()

        xs = np.linspace(0.0, 1.0, grid_nx, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, grid_ny, dtype=np.float32)
        location = np.vstack([xx.ravel() for xx in np.meshgrid(xs, ys)]).T
        trunk = torch.from_numpy(location).to(device)
        return model, trunk, device
    finally:
        sys.argv = original_argv
        try:
            sys.path.remove(str(example_dir))
        except ValueError:
            pass


def write_line(handle, prefix: str, stage: str, event_index: int, residual: float, extra: str = ""):
    if handle is None:
        return
    tag = prefix if prefix else "[hydea pressure]"
    suffix = f" {extra}" if extra else ""
    handle.write(f"{tag} {stage} iter={event_index} residual={residual:.6e}{suffix}\n")


def set_extra_petsc_options(ksp, options_dict, handled_keys):
    prefix = "hydea_"
    ksp.setOptionsPrefix(prefix)
    petsc_options = PETSc.Options()
    for key, value in options_dict.items():
        if key in handled_keys:
            continue
        petsc_options[f"{prefix}{key}"] = str(value).lower() if isinstance(value, bool) else str(value)


def try_ml_update(model, trunkin, a_torch, rhs_torch, x_array, grid_nx, grid_ny, device):
    x_tensor = torch.from_numpy(x_array.astype(np.float32)).to(device)
    r_t = rhs_torch - torch.sparse.mm(a_torch, x_tensor)
    residual_old = float(torch.linalg.norm(r_t).item())
    if not np.isfinite(residual_old):
        raise RuntimeError("HyDEA ML stage produced non-finite residual")
    if residual_old <= 1.0e-30:
        return x_array, residual_old, residual_old, 0.0, True

    r_input = (r_t / residual_old).reshape(1, 1, grid_ny, grid_nx)
    q = model(r_input.float(), trunkin)
    q_vec = q.reshape(a_torch.shape[0], 1)
    aq = torch.sparse.mm(a_torch, q_vec)
    denom = torch.matmul(q_vec.t(), aq).item()
    numer = torch.matmul(q_vec.t(), r_t).item()
    if not np.isfinite(denom) or abs(denom) < 1.0e-30:
        raise RuntimeError("HyDEA ML stage encountered invalid q^T A q")

    alpha = numer / denom
    candidate_x = x_tensor + alpha * q_vec
    candidate_r = rhs_torch - torch.sparse.mm(a_torch, candidate_x)
    residual_new = float(torch.linalg.norm(candidate_r).item())
    if not np.isfinite(residual_new):
        raise RuntimeError("HyDEA ML stage produced non-finite candidate residual")
    return candidate_x.detach().cpu().numpy().astype(np.float64), residual_old, residual_new, alpha, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--rhs", required=True)
    parser.add_argument("--solution", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--grid-nx", required=True, type=int)
    parser.add_argument("--grid-ny", required=True, type=int)
    parser.add_argument("--log-prefix", default="")
    parser.add_argument("--monitor-log", default="")
    args = parser.parse_args()

    options = load_options(Path(args.config))
    example_dir = Path(options["example_dir"])
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = (Path.cwd() / model_path).resolve()

    nrows, ncols, rows, cols, vals = read_triplets(Path(args.matrix))
    rhs_values = read_vector(Path(args.rhs))
    if len(rhs_values) != nrows:
        raise RuntimeError("rhs length does not match matrix size")

    model, trunkin, device = load_hydea_model(
        example_dir, model_path, args.grid_nx, args.grid_ny, bool(options.get("cuda", False))
    )

    mat = build_petsc_matrix(nrows, ncols, rows, cols, vals)
    rhs_vec = build_petsc_vector(rhs_values)
    x_vec = rhs_vec.duplicate()
    x_vec.set(0.0)

    ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    ksp.setOperators(mat)
    ksp.setType(options.get("ksp_type", "cg"))
    ksp.setNormType(resolve_norm_type(options.get("norm_type", "unpreconditioned")))
    pc = ksp.getPC()
    pc.setType(options.get("pc_type", "icc"))

    handled_keys = {
        "example_dir",
        "schedule",
        "ksp_type",
        "pc_type",
        "norm_type",
        "rtol",
        "atol",
        "max_it",
        "num_cg_iterations",
        "num_ml_iterations",
        "cuda",
        "monitor",
        "monitor_stdout",
    }
    set_extra_petsc_options(ksp, options, handled_keys)

    num_cg = int(options.get("num_cg_iterations", 3))
    num_ml = int(options.get("num_ml_iterations", 2))
    schedule = str(options.get("schedule", "warmstart_once")).strip().lower()
    atol = float(options.get("atol", 1.0e-6))
    max_it = int(options.get("max_it", 800))
    ksp.setInitialGuessNonzero(True)
    ksp.setTolerances(
        rtol=float(options.get("rtol", 0.0)),
        atol=atol,
        max_it=num_cg,
    )
    ksp.setFromOptions()

    monitor_log = None
    prefix = args.log_prefix.strip()
    if args.monitor_log:
        monitor_path = Path(args.monitor_log)
        monitor_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if monitor_path.exists() else "w"
        monitor_log = monitor_path.open(mode, encoding="utf-8")
        if prefix:
            monitor_log.write(f"{prefix} begin\n")

    event_index = 0
    algorithm_iterations = 0
    cg_events = 0
    ml_events = 0
    ml_accepted = 0
    ml_rejected = 0

    def log_cg(_ksp, its, rnorm):
        nonlocal event_index, cg_events
        write_line(monitor_log, prefix, "cg", event_index, float(rnorm), f"petsc_its={its}")
        event_index += 1
        cg_events += 1

    if options.get("monitor", False):
        ksp.setMonitor(log_cg)

    row_indices = torch.tensor(np.vstack((rows, cols)), dtype=torch.long, device=device)
    values = torch.tensor(vals, dtype=torch.float32, device=device)
    a_torch = torch.sparse_coo_tensor(row_indices, values, (nrows, ncols), device=device).coalesce()
    rhs_torch = torch.from_numpy(rhs_values.astype(np.float32)).to(device).reshape(nrows, 1)

    x_array = np.zeros((nrows, 1), dtype=np.float64)
    final_residual = np.inf

    with torch.no_grad():
        if schedule == "warmstart_once":
            x_array, residual_old, residual_new, alpha, accepted = try_ml_update(
                model, trunkin, a_torch, rhs_torch, x_array, args.grid_nx, args.grid_ny, device
            )
            final_residual = residual_new
            write_line(
                monitor_log,
                prefix,
                "ml_init",
                event_index,
                final_residual,
                f"alpha={alpha:.6e} previous={residual_old:.6e}",
            )
            ml_accepted += 1
            event_index += 1
            ml_events += 1
            algorithm_iterations += 1

            if final_residual > atol:
                ksp.setTolerances(
                    rtol=float(options.get("rtol", 0.0)),
                    atol=atol,
                    max_it=max(1, max_it - algorithm_iterations),
                )
                ksp.setFromOptions()
                x_vec.setValues(range(nrows), x_array.ravel())
                x_vec.assemblyBegin()
                x_vec.assemblyEnd()
                ksp.solve(rhs_vec, x_vec)
                x_array = x_vec.getArray(readonly=True).copy().reshape(nrows, 1)
                final_residual = float(ksp.getResidualNorm())
                algorithm_iterations += max(0, ksp.getIterationNumber())
        elif schedule == "alternating":
            ksp.solve(rhs_vec, x_vec)
            x_array = x_vec.getArray(readonly=True).copy().reshape(nrows, 1)
            final_residual = float(ksp.getResidualNorm())
            algorithm_iterations = num_cg

            while final_residual > atol and algorithm_iterations < max_it:
                if (algorithm_iterations + num_ml) % (num_cg + num_ml) == 0:
                    for _ in range(num_ml):
                        x_array, residual_old, residual_new, alpha, accepted = try_ml_update(
                            model, trunkin, a_torch, rhs_torch, x_array, args.grid_nx, args.grid_ny, device
                        )
                        final_residual = residual_new
                        write_line(
                            monitor_log,
                            prefix,
                            "ml",
                            event_index,
                            final_residual,
                            f"alpha={alpha:.6e} previous={residual_old:.6e}",
                        )
                        ml_accepted += 1
                        event_index += 1
                        ml_events += 1
                        algorithm_iterations += 1
                        if final_residual <= atol or algorithm_iterations >= max_it:
                            break
                else:
                    x_vec.setValues(range(nrows), x_array.ravel())
                    x_vec.assemblyBegin()
                    x_vec.assemblyEnd()
                    ksp.solve(rhs_vec, x_vec)
                    x_array = x_vec.getArray(readonly=True).copy().reshape(nrows, 1)
                    final_residual = float(ksp.getResidualNorm())
                    algorithm_iterations += num_cg
        else:
            raise RuntimeError(f"unsupported HyDEA schedule: {schedule}")

    with Path(args.solution).open("w", encoding="utf-8") as handle:
        handle.write(f"{nrows}\n")
        for value in x_array.ravel():
            handle.write(f"{value:.17e}\n")

    with Path(args.report).open("w", encoding="utf-8") as handle:
        handle.write(f"iterations {algorithm_iterations}\n")
        handle.write(f"residual_norm {final_residual:.17e}\n")
        handle.write(f"residual_events {event_index}\n")
        handle.write(f"cg_events {cg_events}\n")
        handle.write(f"ml_events {ml_events}\n")
        handle.write(f"ml_accepted {ml_accepted}\n")
        handle.write(f"ml_rejected {ml_rejected}\n")
        handle.write(f"ksp_type {ksp.getType()}\n")
        handle.write(f"pc_type {pc.getType()}\n")

    if monitor_log is not None:
        if prefix:
            monitor_log.write(
                f"{prefix} end iterations={algorithm_iterations} residual={final_residual:.6e} cg_events={cg_events} ml_events={ml_events}\n"
            )
        else:
            monitor_log.write(
                f"[hydea pressure] end iterations={algorithm_iterations} residual={final_residual:.6e} cg_events={cg_events} ml_events={ml_events}\n"
            )
        monitor_log.close()

    if not np.isfinite(final_residual) or final_residual > atol:
        raise RuntimeError(f"HyDEA pressure solve did not converge to atol={atol}, residual={final_residual}")


if __name__ == "__main__":
    main()
