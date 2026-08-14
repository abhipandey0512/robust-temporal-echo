import cv2
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = Path(r"C:\Research\EchoNet-Dynamic")

VIDEO_DIR = DATASET_ROOT / "Videos"
FILE_LIST = DATASET_ROOT / "FileList.csv"

NUM_FRAMES = 10
IMAGE_SIZE = (32, 32)


# ============================================================
# EchoNet Dataset
# ============================================================

class EchoDataset(Dataset):

    def __init__(
        self,
        csv_file,
        video_dir,
        num_frames=10,
        image_size=(32, 32),
    ):

        self.data = pd.read_csv(csv_file)

        self.video_dir = Path(video_dir)

        self.num_frames = num_frames

        self.image_size = image_size


    # --------------------------------------------------------
    # Number of samples
    # --------------------------------------------------------

    def __len__(self):

        return len(self.data)


    # --------------------------------------------------------
    # Load one sample
    # --------------------------------------------------------

    def __getitem__(self, index):

        row = self.data.iloc[index]

        # ----------------------------------------------
        # Video path
        # ----------------------------------------------

        video_name = row["FileName"]

        video_path = (
            self.video_dir /
            f"{video_name}.mp4"
        )


        # ----------------------------------------------
        # EF label
        # ----------------------------------------------

        ef = float(row["EF"])


        # ----------------------------------------------
        # Binary classification label
        #
        # EF > 50  -> 0 (Normal)
        # EF <= 50 -> 1 (Reduced)
        # ----------------------------------------------

        label = 0 if ef > 50 else 1


        # ----------------------------------------------
        # Open video
        # ----------------------------------------------

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():

            raise RuntimeError(
                f"Could not open video: {video_path}"
            )


        # ----------------------------------------------
        # Number of frames
        # ----------------------------------------------

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )


        if total_frames <= 0:

            cap.release()

            raise RuntimeError(
                f"Invalid video: {video_path}"
            )


        # ----------------------------------------------
        # Uniform temporal sampling
        # ----------------------------------------------

        frame_indices = np.linspace(
            0,
            total_frames - 1,
            self.num_frames,
            dtype=int,
        )


        frames = []


        # ----------------------------------------------
        # Read sampled frames
        # ----------------------------------------------

        for frame_idx in frame_indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_idx),
            )

            success, frame = cap.read()


            if not success:

                cap.release()

                raise RuntimeError(
                    f"Could not read frame "
                    f"{frame_idx} from {video_path}"
                )


            # BGR -> grayscale

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )


            # 112 x 112 -> 32 x 32

            frame = cv2.resize(
                frame,
                self.image_size,
                interpolation=cv2.INTER_AREA,
            )


            # 0-255 -> 0-1

            frame = (
                frame.astype(np.float32)
                / 255.0
            )


            frames.append(frame)


        cap.release()


        # ----------------------------------------------
        # Convert to NumPy
        #
        # [T, H, W]
        # ----------------------------------------------

        frames = np.stack(frames)


        # ----------------------------------------------
        # Add channel dimension
        #
        # [T, H, W]
        #       ↓
        # [T, C, H, W]
        # ----------------------------------------------

        frames = frames[:, np.newaxis, :, :]


        # ----------------------------------------------
        # NumPy -> PyTorch tensor
        # ----------------------------------------------

        video = torch.from_numpy(frames)


        # ----------------------------------------------
        # Label tensor
        # ----------------------------------------------

        label = torch.tensor(
            label,
            dtype=torch.long,
        )


        return video, label


# ============================================================
# Test Dataset
# ============================================================

if __name__ == "__main__":

    dataset = EchoDataset(
        csv_file=FILE_LIST,
        video_dir=VIDEO_DIR,
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE,
    )


    print("=" * 60)
    print("DATASET TEST")
    print("=" * 60)

    print("Number of samples:", len(dataset))


    # Load first sample

    video, label = dataset[0]


    print("\nSample 0")

    print("Video shape:", video.shape)

    print("Video dtype:", video.dtype)

    print("Label:", label.item())

    print(
        "Pixel range:",
        video.min().item(),
        "to",
        video.max().item(),
    )
    # ============================================================
# Test Dataset
# ============================================================

if __name__ == "__main__":

    dataset = EchoDataset(
        csv_file=FILE_LIST,
        video_dir=VIDEO_DIR,
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE,
    )

    print("=" * 60)
    print("DATASET TEST")
    print("=" * 60)

    print("Number of samples:", len(dataset))

    # --------------------------------------------------------
    # Test one sample
    # --------------------------------------------------------

    video, label = dataset[0]

    print("\nSample 0")

    print("Video shape:", video.shape)
    print("Video dtype:", video.dtype)
    print("Label:", label.item())

    print(
        "Pixel range:",
        video.min().item(),
        "to",
        video.max().item(),
    )

    # ========================================================
    # DataLoader test
    # ========================================================

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    videos, labels = next(iter(dataloader))

    print("\n" + "=" * 60)
    print("DATALOADER TEST")
    print("=" * 60)

    print("Batch video shape:", videos.shape)
    print("Batch label shape:", labels.shape)

    print("Batch video dtype:", videos.dtype)
    print("Batch labels:", labels)

    print(
        "Batch pixel range:",
        videos.min().item(),
        "to",
        videos.max().item(),
    )# ============================================================
# Test Dataset
# ============================================================

if __name__ == "__main__":

    dataset = EchoDataset(
        csv_file=FILE_LIST,
        video_dir=VIDEO_DIR,
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE,
    )

    print("=" * 60)
    print("DATASET TEST")
    print("=" * 60)

    print("Number of samples:", len(dataset))

    # --------------------------------------------------------
    # Test one sample
    # --------------------------------------------------------

    video, label = dataset[0]

    print("\nSample 0")

    print("Video shape:", video.shape)
    print("Video dtype:", video.dtype)
    print("Label:", label.item())

    print(
        "Pixel range:",
        video.min().item(),
        "to",
        video.max().item(),
    )

    # ========================================================
    # DataLoader test
    # ========================================================

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    videos, labels = next(iter(dataloader))

    print("\n" + "=" * 60)
    print("DATALOADER TEST")
    print("=" * 60)

    print("Batch video shape:", videos.shape)
    print("Batch label shape:", labels.shape)

    print("Batch video dtype:", videos.dtype)
    print("Batch labels:", labels)

    print(
        "Batch pixel range:",
        videos.min().item(),
        "to",
        videos.max().item(),
    )
        # ========================================================
    # DataLoader test
    # ========================================================

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    videos, labels = next(iter(dataloader))

    print("\n" + "=" * 60)
    print("DATALOADER TEST")
    print("=" * 60)

    print("Batch video shape:", videos.shape)
    print("Batch label shape:", labels.shape)

    print("Batch video dtype:", videos.dtype)
    print("Batch labels:", labels)

    print(
        "Batch pixel range:",
        videos.min().item(),
        "to",
        videos.max().item(),
    )
        # ========================================================
    # DataLoader test
    # ========================================================

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    videos, labels = next(iter(dataloader))

    print("\n" + "=" * 60)
    print("DATALOADER TEST")
    print("=" * 60)

    print("Batch video shape:", videos.shape)
    print("Batch label shape:", labels.shape)

    print("Batch video dtype:", videos.dtype)
    print("Batch labels:", labels)

    print(
        "Batch pixel range:",
        videos.min().item(),
        "to",
        videos.max().item(),
    )
        # ========================================================
    # DataLoader test
    # ========================================================

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    videos, labels = next(iter(dataloader))

    print("\n" + "=" * 60)
    print("DATALOADER TEST")
    print("=" * 60)

    print("Batch video shape:", videos.shape)
    print("Batch label shape:", labels.shape)

    print("Batch video dtype:", videos.dtype)
    print("Batch labels:", labels)

    print(
        "Batch pixel range:",
        videos.min().item(),
        "to",
        videos.max().item(),
    )


        # ========================================================
    # DataLoader test
    # ========================================================

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    videos, labels = next(iter(dataloader))

    print("\n" + "=" * 60)
    print("DATALOADER TEST")
    print("=" * 60)

    print("Batch video shape:", videos.shape)
    print("Batch label shape:", labels.shape)

    print("Batch video dtype:", videos.dtype)
    print("Batch labels:", labels)

    print(
        "Batch pixel range:",
        videos.min().item(),
        "to",
        videos.max().item(),
    )
    # ========================================================
    # DataLoader test
    # ========================================================

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    videos, labels = next(iter(dataloader))

    print("\n" + "=" * 60)
    print("DATALOADER TEST")
    print("=" * 60)

    print("Batch video shape:", videos.shape)
    print("Batch label shape:", labels.shape)

    print("Batch video dtype:", videos.dtype)
    print("Batch labels:", labels)

    print(
        "Batch pixel range:",
        videos.min().item(),
        "to",
        videos.max().item(),
    )

print("\n" + "=" * 60)
print("DATASET SPLIT CHECK")
print("=" * 60)

print(dataset.data["Split"].value_counts())


    )

print("\n" + "=" * 60)
print("DATASET SPLIT CHECK")
print("=" * 60)

print(dataset.data["Split"].value_counts())