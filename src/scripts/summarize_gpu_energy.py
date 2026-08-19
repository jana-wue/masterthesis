from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S.%f"


def parse_timestamp(value: str) -> datetime:
    """Parse nvidia-smi timestamps with or without fractional seconds."""
    value = value.strip()
    try:
        return datetime.strptime(value, TIMESTAMP_FORMAT)
    except ValueError:
        return datetime.strptime(value, "%Y/%m/%d %H:%M:%S")


def read_samples(path: Path) -> list[tuple[datetime, float, float, float]]:
    """Read timestamp, power, utilization and memory from a raw CSV."""
    samples = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 4:
                continue
            try:
                samples.append(
                    (
                        parse_timestamp(row[0]),
                        float(row[1]),
                        float(row[2]),
                        float(row[3]),
                    )
                )
            except ValueError:
                continue
    return samples


def calculate_energy_wh(samples: list[tuple[datetime, float, float, float]]) -> float:
    """Integrate power over time with the trapezoidal rule."""
    if len(samples) < 2:
        raise ValueError("At least two valid GPU samples are required.")

    watt_seconds = 0.0
    for previous, current in zip(samples, samples[1:]):
        seconds = (current[0] - previous[0]).total_seconds()
        if seconds <= 0:
            continue
        average_power_watts = (previous[1] + current[1]) / 2
        watt_seconds += average_power_watts * seconds
    return watt_seconds / 3600


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples_csv", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--run_label", required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    args = parser.parse_args()

    samples = read_samples(args.samples_csv)
    energy_wh = calculate_energy_wh(samples)
    duration_seconds = (samples[-1][0] - samples[0][0]).total_seconds()

    summary = {
        "method": args.method,
        "run_label": args.run_label,
        "n_samples": len(samples),
        "duration_seconds": round(duration_seconds, 3),
        "gpu_energy_wh": round(energy_wh, 4),
        "mean_gpu_power_w": round(sum(sample[1] for sample in samples) / len(samples), 3),
        "peak_gpu_power_w": round(max(sample[1] for sample in samples), 3),
        "mean_gpu_utilization_pct": round(sum(sample[2] for sample in samples) / len(samples), 3),
        "peak_gpu_memory_mib": round(max(sample[3] for sample in samples), 3),
    }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)

    print("GPU measurement summary")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
