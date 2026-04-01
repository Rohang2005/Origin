import subprocess
import sys
import time

STEPS = [
    ("Step 1/6: Downloading datasets",     "download_data.py"),
    ("Step 2/6: Preparing data",            "prepare_data.py"),
    ("Step 3/6: Training model",            "train.py"),
    ("Step 4/6: Running inference",         "inference.py"),
    ("Step 5/6: Evaluating results",        "evaluate.py"),
    ("Step 6/6: Generating visualizations", "visualize.py"),
]


def run_step(description: str, script: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {description}")
    print(f"  Script: {script}")
    print("=" * 60 + "\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  FAILED: {script} exited with code {result.returncode}")
        print(f"  Pipeline aborted.")
        sys.exit(result.returncode)

    print(f"\n  Completed {script} in {elapsed:.1f}s")


def main() -> None:
    print("\n" + "#" * 60)
    print("#" + " " * 58 + "#")
    print("#   TEXT-CONDITIONED IMAGE SEGMENTATION PIPELINE" + " " * 10 + "#")
    print("#   Drywall QA — CLIPSeg Fine-Tuning" + " " * 22 + "#")
    print("#" + " " * 58 + "#")
    print("#" * 60)

    pipeline_start = time.time()

    for description, script in STEPS:
        try:
            run_step(description, script)
        except Exception as e:
            print(f"\n  FATAL ERROR in {script}: {e}")
            sys.exit(1)

    total_time = time.time() - pipeline_start
    total_min = total_time / 60.0

    print("\n" + "#" * 60)
    print(f"#  PIPELINE COMPLETE — Total time: {total_min:.1f} minutes")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
