"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
from tqdm.auto import tqdm
import tyro

REPO_NAME = "your_hf_username/libero"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]  # For simplicity we will combine multiple Libero datasets into one training dataset


def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Load the datasets first so the progress bar can report a total across all
    # Libero subsets. Each RLDS example is one episode.
    raw_datasets = []
    total_episodes = 0
    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_dataset, dataset_info = tfds.load(
            raw_dataset_name, data_dir=data_dir, split="train", with_info=True
        )
        raw_datasets.append((raw_dataset_name, raw_dataset))
        total_episodes += dataset_info.splits["train"].num_examples

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset.
    # You can modify this for your own data format.
    with tqdm(total=total_episodes, desc="Converting LIBERO", unit="episode", dynamic_ncols=True) as progress:
        for raw_dataset_name, raw_dataset in raw_datasets:
            progress.set_postfix(dataset=raw_dataset_name)
            for episode in raw_dataset:
                for step in episode["steps"].as_numpy_iterator():
                    dataset.add_frame(
                        {
                            "image": step["observation"]["image"],
                            "wrist_image": step["observation"]["wrist_image"],
                            "state": step["observation"]["state"],
                            "actions": step["action"],
                            "task": step["language_instruction"].decode(),
                        }
                    )
                dataset.save_episode()
                progress.update()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
