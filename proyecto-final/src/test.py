import glob
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

from .dummy import dummy_method
from .segment import ExCuSe

plt.style.use('ggplot')

def plot_cumulative_error(
    algorithm_errors: list[np.ndarray], titles: list[str], plot_path: str, plot_title: str = "Cumulative Error Distribution of Each Algorithm"
) -> None:
    total_samples = len(algorithm_errors[0])
    errors = []
    for err in algorithm_errors:
        errors.extend(err)
    x_values = sorted(set(errors))

    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)

    # Plot cumulative error distribution
    for errors, title in zip(algorithm_errors, titles):
        alg_err_dist = [
            sum(1 for error in errors if error <= x) / total_samples
            for x in x_values
        ]
        ax.plot(x_values, alg_err_dist, label=title)

    ax.set_ylabel("Percentage of Images")
    ax.set_xlabel("Detection Error")
    ax.set_title(plot_title)

    # Set tick labels with units using formatters
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}px"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y*100:.0f}%"))

    ax.legend()
    plt.savefig(plot_path)


def error_fn(x: float, y: float) -> float:
    return np.sqrt(np.sum((x - y) ** 2))

def process_subfolder(
    subfolder_path: str, algorithms: list[Callable[[str], tuple[int, int]]]
) -> list[float]:
    
    results = []
    for annot_path in Path(subfolder_path).glob("*.txt"):
        print(f"Processing {annot_path}")
        ground_truth = pd.read_csv(annot_path, sep=" ", header=None)
        ground_truth.columns = ["x", "y"]

        video_number = annot_path.stem

        # uso 1999 frames nomas porque sub 1 video 4 solo tiene 1999 frames
        frames_errs = []
        for frame in range(1, 1501):
            img = cv2.imread(
                f"{subfolder_path}/{video_number}_frames/frame_{frame}.png", 0
            )
            print(f"Processing frame {frame} of {video_number}")

            x, y = ground_truth.iloc[frame - 1].values
            alg_errs = []
            for alg in algorithms:
                c_x, c_y = alg(img)
                error = error_fn(c_x - x, c_y - y)
                alg_errs.append(error)

            frames_errs.append(alg_errs)
        results.extend(frames_errs)
            
    # results_subfolder = [
            # video1
            # [ err_alg1_frame1, err_alg2_frame1, ... xN algorithms ],
            # [ err_alg1_frame2, err_alg2_frame2, ... xN algorithms ],
            # ...
            # [ err_alg1_frame2000, err_alg2_frame2000, ... xN algorithms ],

            # videoM
            # [ err_alg1_frame1, err_alg2_frame1, ... xN algorithms ],
            # [ err_alg1_frame2, err_alg2_frame2, ... xN algorithms ],
            # ...
            # [ err_alg1_frame2000, err_alg2_frame2000, ... xN algorithms ],

    # ]
    return results


def test_folder(csv_dir: str, plots_dir: str, folder_path: str) -> None:
    folder_name = Path(folder_path).stem
    algorithms = {
        "ExCuSe": ExCuSe,
        "Dummy": dummy_method
    }
    results = process_subfolder(folder_path, list(algorithms.values()))
    results = np.array(results)
    
    df = pd.DataFrame(results, columns=algorithms.keys())
    df.to_csv(f"{csv_dir}/{Path(folder_path).stem}.csv", index=False)
    
    rand_errs = np.random.randint(0, 300, size=results.shape[0])
    plot_cumulative_error(
        [results[:,0], results[:,1], rand_errs],
        list(algorithms.keys()) + ["Random"],
        plot_path=f"{plots_dir}/Cum_Err_folder{folder_name}.png",
        plot_title=f"Folder {folder_name}"
    )


def test_dataset(csv_dir: str, plots_dir: str, dataset_path: str) -> None:
    algorithms = {
        "ExCuSe": ExCuSe,
        "Dummy": dummy_method
    }
    results = []
    with ProcessPoolExecutor() as executor:
        MAX_FOLDER = 2  # <23
        for subfolder_idx in range(1, MAX_FOLDER):
            subfolder = f"{dataset_path}/{subfolder_idx}"
            future = executor.submit(
                process_subfolder, subfolder, list(algorithms.values())
            )
            results.append(future)

    # Gather results from all the processes
    final_results = []
    for future in results:
        final_results.extend(future.result())
    final_results = np.array(final_results)
    
    df = pd.DataFrame(final_results, columns=algorithms.keys())
    df.to_csv(f"{csv_dir}/final_results.csv", index=False)
    
    rand_errs = np.random.randint(0, 300, size=final_results.shape[0])
    plot_cumulative_error(
        [final_results[:,0], final_results[:,1], rand_errs],
        list(algorithms.keys()) + ["Random"],
        f"{plots_dir}/cumulative_error.png",
    )
