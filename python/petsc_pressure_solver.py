#!/usr/bin/env python3

import argparse
import importlib.util
from pathlib import Path

from petsc4py import PETSc


def load_options(path: Path):
    spec = importlib.util.spec_from_file_location("petsc_pressure_options", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load PETSc options file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    options = getattr(module, "PETSCPRESSURE_OPTIONS", None)
    if options is None:
        raise RuntimeError("PETSCPRESSURE_OPTIONS dict missing in PETSc options file")
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
    return values


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


def parse_bool(value: str):
    key = value.strip().lower()
    if key in {"1", "true", "yes", "on"}:
        return True
    if key in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"unsupported boolean value: {value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--rhs", default="")
    parser.add_argument("--solution", default="")
    parser.add_argument("--report", default="")
    parser.add_argument("--initial-guess", default="")
    parser.add_argument("--log-prefix", default="")
    parser.add_argument("--monitor-log", default="")
    parser.add_argument("--use-constant-nullspace", default="true")
    parser.add_argument("--server", action="store_true")
    args = parser.parse_args()

    options = load_options(Path(args.config))
    nrows, ncols, rows, cols, vals = read_triplets(Path(args.matrix))

    mat = PETSc.Mat().createAIJ([nrows, ncols], comm=PETSc.COMM_SELF)
    mat.setUp()
    for row, col, value in zip(rows, cols, vals):
        mat.setValue(row, col, value, addv=True)
    mat.assemblyBegin()
    mat.assemblyEnd()

    use_constant_nullspace = parse_bool(args.use_constant_nullspace)
    nullspace = None
    if use_constant_nullspace:
        nullspace = PETSc.NullSpace().create(constant=True, comm=PETSc.COMM_SELF)
        mat.setNullSpace(nullspace)

    rhs = PETSc.Vec().createSeq(nrows, comm=PETSc.COMM_SELF)

    x = rhs.duplicate()
    x.set(0.0)

    ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    ksp.setOperators(mat)
    ksp.setType(options.get("ksp_type", "cg"))
    ksp.setNormType(resolve_norm_type(options.get("norm_type", "unpreconditioned")))
    pc = ksp.getPC()
    pc.setType(options.get("pc_type", "jacobi"))
    ksp.setTolerances(
        rtol=options.get("rtol", 1.0e-8),
        atol=options.get("atol", 1.0e-50),
        max_it=options.get("max_it", 1000),
    )

    # Allow the Python config file to pass through extra PETSc options such as
    # ICC factor shifting without hard-coding every knob in this driver.
    handled_keys = {
        "ksp_type",
        "pc_type",
        "norm_type",
        "rtol",
        "atol",
        "max_it",
        "monitor",
        "monitor_stdout",
    }
    options_prefix = "pressure_"
    ksp.setOptionsPrefix(options_prefix)
    petsc_options = PETSc.Options()
    for key, value in options.items():
        if key in handled_keys:
            continue
        petsc_options[f"{options_prefix}{key}"] = str(value).lower() if isinstance(value, bool) else str(value)

    ksp.setFromOptions()

    monitor_state = {"handle": None, "prefix": "", "stdout": False}

    def monitor(_ksp, its, rnorm):
        if monitor_state["prefix"]:
            line = f"{monitor_state['prefix']} petsc iter {its} residual={rnorm:.6e}"
        else:
            line = f"[petsc pressure] iter {its} residual={rnorm:.6e}"
        if monitor_state["handle"] is not None:
            monitor_state["handle"].write(line + "\n")
        elif monitor_state["stdout"]:
            print(line)

    if options.get("monitor", False):
        ksp.setMonitor(monitor)

    def load_rhs(rhs_path: str):
        rhs_values = read_vector(Path(rhs_path))
        if len(rhs_values) != nrows:
            raise RuntimeError("rhs length does not match matrix size")
        rhs.setValues(range(nrows), rhs_values)
        rhs.assemblyBegin()
        rhs.assemblyEnd()
        if nullspace is not None:
            nullspace.remove(rhs)

    def maybe_load_initial_guess(initial_guess_path: str):
        x.set(0.0)
        if not initial_guess_path:
            ksp.setInitialGuessNonzero(False)
            return
        guess_values = read_vector(Path(initial_guess_path))
        if len(guess_values) != nrows:
            raise RuntimeError("initial guess length does not match matrix size")
        x.setValues(range(nrows), guess_values)
        x.assemblyBegin()
        x.assemblyEnd()
        ksp.setInitialGuessNonzero(True)

    def solve_once(rhs_path: str, solution_path: str, report_path: str, initial_guess_path: str, log_prefix: str, monitor_log: str):
        load_rhs(rhs_path)
        maybe_load_initial_guess(initial_guess_path)

        if monitor_state["handle"] is not None:
            monitor_state["handle"].close()
            monitor_state["handle"] = None
        monitor_state["prefix"] = log_prefix.strip()
        monitor_state["stdout"] = bool(options.get("monitor_stdout", False)) and not monitor_log
        if options.get("monitor", False) and monitor_log:
            monitor_path = Path(monitor_log)
            monitor_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if monitor_path.exists() else "w"
            monitor_state["handle"] = monitor_path.open(mode, encoding="utf-8")
            if monitor_state["prefix"]:
                monitor_state["handle"].write(f"{monitor_state['prefix']} begin\n")

        ksp.solve(rhs, x)

        solution_out = Path(solution_path)
        with solution_out.open("w", encoding="utf-8") as handle:
            handle.write(f"{nrows}\n")
            for value in x.getArray(readonly=True):
                handle.write(f"{value:.17e}\n")

        report_out = Path(report_path)
        with report_out.open("w", encoding="utf-8") as handle:
            handle.write(f"converged_reason {ksp.getConvergedReason()}\n")
            handle.write(f"iterations {ksp.getIterationNumber()}\n")
            handle.write(f"residual_norm {ksp.getResidualNorm():.17e}\n")
            handle.write(f"norm_type {options.get('norm_type', 'unpreconditioned')}\n")
            handle.write(f"ksp_type {ksp.getType()}\n")
            handle.write(f"pc_type {pc.getType()}\n")
            handle.write(f"use_constant_nullspace {str(use_constant_nullspace).lower()}\n")

        if monitor_state["handle"] is not None:
            if monitor_state["prefix"]:
                monitor_state["handle"].write(
                    f"{monitor_state['prefix']} end iterations={ksp.getIterationNumber()} residual={ksp.getResidualNorm():.6e}\n"
                )
            else:
                monitor_state["handle"].write(
                    f"[petsc pressure] end iterations={ksp.getIterationNumber()} residual={ksp.getResidualNorm():.6e}\n"
                )
            monitor_state["handle"].close()
            monitor_state["handle"] = None

        if ksp.getConvergedReason() <= 0:
            raise RuntimeError(f"PETSc KSP did not converge, reason={ksp.getConvergedReason()}")

    if args.server:
        try:
            for raw in iter(input, ""):
                command = raw.rstrip("\n")
                if command == "EXIT":
                    print("OK", flush=True)
                    return
                if command != "SOLVE":
                    raise RuntimeError(f"unknown server command: {command}")
                rhs_path = input().rstrip("\n")
                solution_path = input().rstrip("\n")
                report_path = input().rstrip("\n")
                monitor_log = input().rstrip("\n")
                log_prefix = input().rstrip("\n")
                solve_once(rhs_path, solution_path, report_path, "", log_prefix, monitor_log)
                print("OK", flush=True)
        except Exception as exc:
            print(f"ERROR {exc}", flush=True)
            raise
    else:
        if not args.rhs or not args.solution or not args.report:
            raise RuntimeError("--rhs, --solution, and --report are required unless --server is used")
        solve_once(args.rhs, args.solution, args.report, args.initial_guess, args.log_prefix, args.monitor_log)


if __name__ == "__main__":
    main()
