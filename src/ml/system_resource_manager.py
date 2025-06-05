import torch
import os
import platform
import psutil
import numpy as np
from typing import Dict, Any


class SystemResourceManager:
    """Class for managing and optimizing system resources."""

    @staticmethod
    def get_system_resources() -> Dict[str, Any]:
        """
        Get information about available system resources.

        Returns:
            Dictionary with system resource information
        """
        resources = {
            "cpu_count": os.cpu_count(),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_percent": psutil.cpu_percent(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "platform": platform.system(),
            "cuda_available": torch.cuda.is_available(),
        }

        if resources["cuda_available"]:
            resources.update(
                {
                    "cuda_device_count": torch.cuda.device_count(),
                    "cuda_device_name": torch.cuda.get_device_name(0),
                    "cuda_memory_total": torch.cuda.get_device_properties(
                        0
                    ).total_memory,
                    "cuda_memory_reserved": torch.cuda.memory_reserved(0),
                    "cuda_memory_allocated": torch.cuda.memory_allocated(0),
                }
            )

        return resources

    @staticmethod
    def calculate_optimal_workers(total_cores: int) -> int:
        """
        Calculate the optimal number of worker processes for data loading.

        Args:
            total_cores: Total number of CPU cores available

        Returns:
            Optimal number of worker processes
        """
        if total_cores <= 2:
            return 0  # Disable multiprocessing for systems with few cores
        elif total_cores <= 4:
            return max(1, total_cores - 1)  # Reserve 1 core
        else:
            # For systems with many cores, use 75% of cores, rounding down
            return max(1, int(total_cores * 0.75))

    @staticmethod
    def calculate_optimal_batch_size(
        input_dim: int,
        model_params: int,
        available_memory: int,
        precision: str = "mixed",
    ) -> int:
        """
        Calculate optimal batch size based on available memory.

        Args:
            input_dim: Dimension of input data
            model_params: Number of model parameters
            available_memory: Available memory in bytes
            precision: Precision mode ('full' or 'mixed')

        Returns:
            Optimal batch size
        """
        # Estimate bytes per sample based on precision
        bytes_per_float = 2 if precision == "mixed" else 4

        # Memory for input, output, gradients, optimizer states, etc.
        bytes_per_sample = input_dim * 4 * bytes_per_float

        # Model memory (parameters, gradients, optimizer states)
        model_memory = model_params * 4 * bytes_per_float * 3

        # Use 70% of available memory, accounting for model memory and other overhead
        usable_memory = (available_memory * 0.7) - model_memory

        # Calculate batch size
        batch_size = max(1, int(usable_memory / bytes_per_sample))

        # Cap at reasonable values and ensure it's a power of 2 for GPU efficiency
        batch_size = min(batch_size, 8192)

        # Round down to the nearest power of 2 for better GPU utilization
        batch_size = 2 ** int(np.log2(batch_size))

        return batch_size
