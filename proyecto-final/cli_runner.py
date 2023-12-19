from argparse import ArgumentParser
from src import segment
from src import test

# black -t py39 -l 80 src/*.py cli_runner.py

DATASET_DIR = "dataset/LPW_frames/LPW"
PLOTS_DIR = "plots"
CSV_DIR = "csv"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--img_path",
        type=str,
        nargs="?",
        const=f"{DATASET_DIR}/1/1_frames/frame_1.png",
    )
    parser.add_argument("--folder_path", type=str, nargs="?", const=f"{DATASET_DIR}/1")
    parser.add_argument(
        "--dataset_path", type=str, nargs="?", const=DATASET_DIR
    )
    args = parser.parse_args()

    if args.img_path is not None:
        # guardar la imagen con la pupila detectada marcada
        segment.process_image(PLOTS_DIR, args.img_path)
    elif args.folder_path is not None:
        test.test_folder(CSV_DIR, PLOTS_DIR, args.folder_path)
    elif args.dataset_path is not None:
        # guardar cumulative plot en la carpeta plot
        test.test_dataset(CSV_DIR, PLOTS_DIR, args.dataset_path)
    else:
        print("No arguments provided. Please use --img_path or --dataset_path")
        exit(1)
