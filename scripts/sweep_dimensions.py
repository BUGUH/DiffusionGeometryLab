"""多维度批量实验编排。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dimensions", type=int, nargs="+", default=[4096, 16384])
    parser.add_argument("--schedulers", type=str, nargs="+", default=["vp"])
    parser.add_argument("--extra_overrides", nargs="*", default=[])
    args = parser.parse_args()

    experiments = []
    for d in args.dimensions:
        res = int(d ** 0.5)
        for sched in args.schedulers:
            exp_name = f"sweep_d{d}_{sched}"
            overrides = [
                f"data.target_dim={d}",
                f"data.target_resolution={res}",
                f"scheduler.type={sched}",
                f"experiment.name={exp_name}",
            ] + args.extra_overrides
            experiments.append((exp_name, overrides))

    print(f"计划运行 {len(experiments)} 个实验:")
    for name, ov in experiments:
        print(f"  {name}: {ov}")

    for name, overrides in experiments:
        print(f"\n{'=' * 60}")
        print(f"开始实验: {name}")
        print(f"{'=' * 60}")

        cmd = [
            sys.executable, str(ROOT / "scripts" / "train.py"),
            "--config", args.config,
            "--overrides",
        ] + overrides

        try:
            subprocess.run(cmd, check=True)
            print(f"✅ {name} 完成")
        except subprocess.CalledProcessError as e:
            print(f"❌ {name} 失败: {e}")
            continue

    print(f"\n{'=' * 60}")
    print("所有实验完成，运行评估:")
    print(f"  python scripts/evaluate.py --config {args.config} --ckpt_dir outputs/<exp>/checkpoints")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()