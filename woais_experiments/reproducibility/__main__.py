from woais_experiments.reproducibility.validate import run_validate_artifact

if __name__ == "__main__":
    raise SystemExit(0 if run_validate_artifact()["ok"] else 1)
