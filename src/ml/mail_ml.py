import torch
import time
import logging
from typing import Tuple, Optional
from data.data_acorn import AcornData
from ml_data_processor import DataProcessor
from trainer import AutoencoderTrainer

# Configure backend for performance
torch.backends.cudnn.benchmark = True


# =============== Pipeline Class ===============


class AutoencoderPipeline:
    """End-to-end pipeline for autoencoder workflow."""

    def __init__(
        self,
        acorn_group: str = "Comfortable",
        selected_years: Tuple[int, int] = (2011, 2012),
        encoding_dim: int = 2,
        hidden_dim: int = 8,
        learning_rate: float = 0.001,
        auto_resource_adjustment: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the autoencoder pipeline.

        Args:
            acorn_group: Acorn group to use for data
            selected_years: Year range to select data from
            encoding_dim: Dimension of the encoded representation
            hidden_dim: Dimension of hidden layers
            learning_rate: Learning rate for optimizer
            auto_resource_adjustment: Whether to automatically adjust resources
            device: Device to use (auto-detected if None)
        """
        self.acorn_group = acorn_group
        self.selected_years = selected_years
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.auto_resource_adjustment = auto_resource_adjustment

        # Set device if not provided
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Metadata for model identification
        self.metadata = {
            "acorn_group": acorn_group,
            "selected_years": selected_years,
            "data_source": "acorn",
            "feature": "hh_consumption",
        }

        # Will be set when data is loaded
        self.input_dim = None
        self.original_data = None
        self.processed_data = None

        # Initialize components
        self.data_processor = DataProcessor()

    def load_data(self):
        """Load and preprocess data from the Acorn dataset."""
        start_time = time.time()
        acorn_data = AcornData(
            acorn_group=self.acorn_group, selected_years=self.selected_years
        ).get_data()
        self.original_data = acorn_data.select("hh_consumption").to_numpy()
        # self.metadata = acorn_data.select(
        #     ["day_of_week", "is_weekend", "avg_temperature", "avg_humidity", "season"]
        # )

        logging.info(f"Data loading took {time.time() - start_time:.2f} seconds")

        # Process data
        self.processed_data = self.data_processor.preprocess_data(self.original_data)
        self.input_dim = (
            self.processed_data.shape[1]
            if self.processed_data.ndim > 1
            else self.processed_data.shape[0]
        )

        # Initialize the trainer with the correct input dimension
        self.trainer = AutoencoderTrainer(
            input_dim=self.input_dim,
            encoding_dim=self.encoding_dim,
            hidden_dim=self.hidden_dim,
            learning_rate=self.learning_rate,
            use_amp=True,
            auto_resource_adjustment=self.auto_resource_adjustment,
            device=self.device,
        )

        return self.processed_data

    def train_or_load_model(self, force_retrain: bool = False, epochs: int = 20):
        """Train a new model or load a pre-trained one."""
        if not hasattr(self, "trainer"):
            raise ValueError(
                "Data must be loaded before training or loading a model. Call load_data() first."
            )

        if not force_retrain:
            model_loaded = self.trainer.load_model(self.metadata)
        else:
            model_loaded = False

        if not model_loaded:
            logging.info("Training new model...")
            self.trainer.train(
                self.original_data,
                epochs=epochs,
                batch_size=None,
                num_workers=None,
                log_interval=1,
                validation_split=0.1,
                early_stopping_patience=5,
            )

            self.trainer.save_model(self.metadata)
            return False
        else:
            logging.info("Using pre-trained model")
            return True

    def evaluate_model(self):
        """Evaluate the model on the loaded data."""
        if not hasattr(self, "trainer"):
            raise ValueError(
                "Model must be trained or loaded before evaluation. Call train_or_load_model() first."
            )

        start_time = time.time()
        reconstructed_data, mse = self.trainer.evaluate(
            self.original_data, batch_size=None
        )
        logging.info(f"Evaluation took {time.time() - start_time:.2f} seconds")

        return reconstructed_data, mse

    def visualize_results(self, reconstructed_data, num_examples=5):
        """Visualize the reconstruction results."""
        if not hasattr(self, "trainer"):
            raise ValueError(
                "Model must be evaluated before visualization. Call evaluate_model() first."
            )

        self.trainer.visualize_reconstruction(
            self.processed_data, reconstructed_data, num_examples=num_examples
        )

    def run_pipeline(
        self, force_retrain: bool = False, epochs: int = 20, num_examples: int = 5
    ):
        """Run the full pipeline: load data, train/load model, evaluate, and visualize."""
        # Load and preprocess data
        self.load_data()

        # Train or load model
        self.train_or_load_model(force_retrain=force_retrain, epochs=epochs)

        # Evaluate model
        reconstructed_data, mse = self.evaluate_model()

        # Visualize results
        self.visualize_results(reconstructed_data, num_examples=num_examples)

        return reconstructed_data, mse


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the full pipeline
    pipeline = AutoencoderPipeline(
        acorn_group="Comfortable",
        selected_years=(2011, 2012),
        encoding_dim=2,
        hidden_dim=8,
        auto_resource_adjustment=True,
    )

    pipeline.run_pipeline(force_retrain=False, epochs=20, num_examples=5)

