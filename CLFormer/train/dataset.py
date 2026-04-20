import os
import glob
import pickle
import _pickle
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class PSFDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        num_images: int = 4,
        num_coefficients: int = 25,
    ):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if not self.files:
            raise FileNotFoundError(f"No .npy files found in: {data_dir}")

        self.num_images = num_images
        self.num_coefficients = num_coefficients

        sample = np.load(self.files[0], allow_pickle=True)
        if isinstance(sample[1], dict):
            self.label_shape = sample[1]["gt_a"].shape
        else:
            self.label_shape = sample[1].shape

        print(f"Dataset: {data_dir}")
        print(f"  Samples: {len(self.files)}")
        print(f"  Mode: {num_images} images -> {num_coefficients} coefficients")
        print(f"  Input shape: {sample[0].shape}, Label shape: {self.label_shape}")

        self._validate_dataset()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        max_retries = 5
        for retry in range(max_retries):
            try:
                current_idx = (idx + retry) % len(self.files)
                data = np.load(self.files[current_idx], allow_pickle=True)

                if len(data) < 2:
                    raise ValueError("Incomplete data")

                inputs = torch.from_numpy(
                    data[0][: self.num_images].astype(np.float32)
                )

                if isinstance(data[1], dict):
                    labels_matrix = data[1]["gt_a"].astype(np.float32)
                else:
                    labels_matrix = data[1].astype(np.float32)

                labels = self._extract_coefficients(labels_matrix)
                return inputs, labels

            except (
                EOFError, ValueError, OSError, IndexError,
                pickle.UnpicklingError, _pickle.UnpicklingError,
            ):
                if retry == max_retries - 1:
                    print(
                        f"Error: All retries failed for index {idx}, "
                        "returning zero data"
                    )
                    inputs = torch.zeros(
                        self.num_images, 112, 112, dtype=torch.float32
                    )
                    labels = torch.zeros(
                        self.num_coefficients, dtype=torch.float32
                    )
                    return inputs, labels
                continue

    def _validate_dataset(self) -> None:
        print("Validating dataset integrity...")
        corrupted = []
        for i in range(min(100, len(self.files))):
            try:
                data = np.load(self.files[i], allow_pickle=True)
                if len(data) < 2:
                    corrupted.append(self.files[i])
            except (
                EOFError, ValueError, OSError,
                pickle.UnpicklingError, _pickle.UnpicklingError,
            ):
                corrupted.append(self.files[i])

        if corrupted:
            print(f"Found {len(corrupted)} corrupted files in sample check")
            print("Note: Corrupted files will be skipped during training")
        else:
            print("Dataset validation passed")

    def _extract_coefficients(
        self, labels_matrix: np.ndarray,
    ) -> torch.Tensor:
        if self.num_coefficients == 25:
            coeffs = np.zeros(25, dtype=np.float32)
            coeffs[0] = labels_matrix[0, 3]
            coeffs[1:] = labels_matrix[1:7, :4].reshape(-1)
            return torch.from_numpy(coeffs)

        if self.num_coefficients == 77:
            matrix = labels_matrix[:7, :11]
            return torch.from_numpy(matrix.reshape(-1))

        raise ValueError(
            f"Unsupported num_coefficients: {self.num_coefficients}"
        )
