import yaml
from pathlib import Path


class Config:
    def __init__(self, config_path: str = "config.yaml"):
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, "r") as f:
            self._config = yaml.safe_load(f)

        self._validate_config()

    def _validate_config(self):
        required_sections = ["data", "preprocessing", "model", "training", "clustering"]
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required config section: {section}")

    @property
    def data_dir(self) -> str:
        return self._config["data"]["data_dir"]

    @property
    def counts_file(self) -> str:
        return self._config["data"]["counts_file"]

    @property
    def min_spots_per_gene(self) -> int:
        return self._config["preprocessing"]["min_spots_per_gene"]

    @property
    def top_genes_count(self) -> int:
        return self._config["preprocessing"]["top_genes_count"]

    @property
    def pca_components(self) -> int:
        return self._config["preprocessing"]["pca_components"]

    @property
    def spatial_neighbors(self) -> int:
        return self._config["preprocessing"]["spatial_neighbors"]

    @property
    def latent_dimensions(self) -> int:
        return self._config["model"]["latent_dimensions"]

    @property
    def hidden_layer_dimensions(self) -> tuple:
        return tuple(self._config["model"]["hidden_layer_dimensions"])

    @property
    def training_epochs(self) -> int:
        return self._config["training"]["epochs"]

    @property
    def batch_size(self) -> int:
        return self._config["training"]["batch_size"]

    @property
    def learning_rate(self) -> float:
        return self._config["training"]["learning_rate"]

    @property
    def spatial_regularization_weight(self) -> float:
        return self._config["training"]["spatial_regularization_weight"]

    @property
    def cluster_neighbors(self) -> int:
        return self._config["clustering"]["n_neighbors"]

    def __repr__(self) -> str:
        return f"Config({self._config})"


def load_config(config_path: str = "config.yaml") -> Config:
    return Config(config_path)
