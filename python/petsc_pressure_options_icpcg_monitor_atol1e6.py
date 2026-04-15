PETSCPRESSURE_OPTIONS = {
    "ksp_type": "cg",
    "pc_type": "icc",
    "pc_factor_shift_type": "positive_definite",
    "pc_factor_shift_amount": 1.0e-12,
    "norm_type": "unpreconditioned",
    "rtol": 0.0,
    "atol": 1.0e-6,
    "max_it": 1500,
    "monitor": True,
    "monitor_stdout": False,
}
