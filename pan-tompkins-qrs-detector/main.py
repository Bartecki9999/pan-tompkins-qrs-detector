"""
Run the Pan-Tompkins detector on the two MIT-BIH records in ./data and print
how well it does.

Examples
--------
    python main.py                      # run, print results, open plot windows
    python main.py --no-show            # just print results
    python main.py --save-figures images
    python main.py --window-experiment  # also try different integration windows
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from pan_tompkins import detect_r_peaks, evaluate, load_annotations, load_signal
from plots import plot_detection, plot_filter_response

FS = 360  # sampling frequency of the MIT-BIH Arrhythmia Database [Hz]

# record name -> (signal file, annotation file)
RECORDS = {
    "102": ("102_V5.dat", "102_attr.dat"),
    "228": ("228_MLII.dat", "228_attr.dat"),
}

WINDOWS_MS = [50, 100, 150, 200, 300]


def parse_args():
    parser = argparse.ArgumentParser(description="Pan-Tompkins QRS detection on MIT-BIH records.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data",
                        help="folder with the .dat files (default: ./data)")
    parser.add_argument("--save-figures", type=Path, metavar="DIR",
                        help="save the figures as PNG files into DIR")
    parser.add_argument("--no-show", action="store_true",
                        help="do not open plot windows")
    parser.add_argument("--window-experiment", action="store_true",
                        help="repeat the evaluation for several integration window lengths")
    return parser.parse_args()


def print_metrics(m):
    print(f"  MAE = {m.mae_ms:.2f} ms   STD = {m.std_ms:.2f} ms   "
          f"TP = {m.tp}   FP = {m.fp}   FN = {m.fn}   "
          f"(sensitivity {100 * m.sensitivity:.2f}%, precision {100 * m.precision:.2f}%)")


def main():
    args = parse_args()

    if args.save_figures:
        args.save_figures.mkdir(parents=True, exist_ok=True)

    records = {}
    for name, (signal_file, annotation_file) in RECORDS.items():
        records[name] = (load_signal(args.data_dir / signal_file),
                         load_annotations(args.data_dir / annotation_file))

    figures = {"filter_response": plot_filter_response(FS)}

    for name, (ecg, reference) in records.items():
        detection = detect_r_peaks(ecg, FS)
        metrics = evaluate(detection.r_peaks, reference, FS)

        print(f"Record {name}: detected {len(detection.r_peaks)}, reference {len(reference)}")
        print_metrics(metrics)
        print()

        figures[f"record_{name}"] = plot_detection(
            ecg, detection, reference, FS, f"Record {name} - detected QRS complexes")

    if args.window_experiment:
        print("Integration window experiment")
        for name, (ecg, reference) in records.items():
            for window_ms in WINDOWS_MS:
                detection = detect_r_peaks(ecg, FS, integration_ms=window_ms)
                m = evaluate(detection.r_peaks, reference, FS)
                print(f"  record {name}, {window_ms:3d} ms:  MAE = {m.mae_ms:.2f}  "
                      f"STD = {m.std_ms:.2f}  FP = {m.fp}  FN = {m.fn}")
        print()

    if args.save_figures:
        for key, fig in figures.items():
            path = args.save_figures / f"{key}.png"
            fig.savefig(path, dpi=130)
            print(f"saved {path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
