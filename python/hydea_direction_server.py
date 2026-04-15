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


def load_options(path: Path):
    spec = importlib.util.spec_from_file_location("hydea_direction_options", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load HyDEA options file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    options = getattr(module, "HYDEA_OPTIONS", None)
    if options is None:
        raise RuntimeError("HYDEA_OPTIONS dict missing in options file")
    return options


def read_vector(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        n = int(handle.readline().strip())
        values = [float(line.strip()) for line in handle if line.strip()]
    if len(values) != n:
        raise RuntimeError(f"vector length mismatch: expected {n}, got {len(values)}")
    return np.asarray(values, dtype=np.float64)


def write_vector(path: Path, values):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{array.size}\n")
        for value in array:
            handle.write(f"{value:.17e}\n")


def load_hydea_model(example_dir: Path, model_path: Path, grid_nx: int, grid_ny: int, use_cuda: bool):
    if grid_nx != 192 or grid_ny != 192:
        raise RuntimeError(f"HyDEA local model is hard-wired for 192x192, got {grid_nx}x{grid_ny}")

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


def predict_ml_direction(model, trunkin, residual, grid_nx: int, grid_ny: int, device):
    residual_array = np.asarray(residual, dtype=np.float64).reshape(-1)
    if residual_array.size != grid_nx * grid_ny:
        raise RuntimeError(
            f"direction input has length {residual_array.size}, expected {grid_nx * grid_ny}"
        )
    residual_norm = float(np.linalg.norm(residual_array))
    if not np.isfinite(residual_norm):
        raise RuntimeError("HyDEA ML direction received non-finite residual")
    if residual_norm <= 1.0e-30:
        return np.zeros_like(residual_array)

    residual_input = torch.from_numpy(
        (residual_array / residual_norm).astype(np.float32).reshape(1, 1, grid_ny, grid_nx)
    ).to(device)
    with torch.no_grad():
        q = model(residual_input.float(), trunkin).reshape(-1)
    q_array = q.detach().cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(q_array)):
        raise RuntimeError("HyDEA ML direction produced non-finite values")
    return q_array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--grid-nx", required=True, type=int)
    parser.add_argument("--grid-ny", required=True, type=int)
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    options = load_options(Path(args.config))
    example_dir = Path(options["example_dir"])
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = (Path.cwd() / model_path).resolve()
    model, trunkin, device = load_hydea_model(
        example_dir, model_path, args.grid_nx, args.grid_ny, bool(options.get("cuda", False))
    )

    def process(input_path: str, output_path: str):
        residual = read_vector(Path(input_path))
        direction = predict_ml_direction(model, trunkin, residual, args.grid_nx, args.grid_ny, device)
        write_vector(Path(output_path), direction)

    if args.server:
        for line in sys.stdin:
            command = line.strip()
            if not command:
                continue
            if command == "EXIT":
                print("OK", flush=True)
                return
            if command != "DIRECTION":
                print(f"ERROR unsupported command: {command}", flush=True)
                continue
            input_path = sys.stdin.readline().strip()
            output_path = sys.stdin.readline().strip()
            try:
                process(input_path, output_path)
                print("OK", flush=True)
            except Exception as exc:  # pragma: no cover - exercised via parent process
                print(f"ERROR {exc}", flush=True)
        return

    if not args.input or not args.output:
        raise RuntimeError("--input and --output are required unless --server is used")
    process(args.input, args.output)


if __name__ == "__main__":
    main()
