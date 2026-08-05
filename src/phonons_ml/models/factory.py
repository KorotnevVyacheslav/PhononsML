from dataclasses import asdict, dataclass

from phonons_ml.models.base import ModelBase, ModelConfigBase


@dataclass
class ModelETLFactoryConfig:
    """Configuration container for model ETL factory.

    Attributes:
        model_name: Name of the model implementation ('mattersim' or 'mace').
        model_config: Configuration object for the specific model.
    """

    model_name: str
    model_config: ModelConfigBase

    def to_dict(self) -> dict[str, object]:
        """Convert the configuration to a dictionary.

        Returns:
            dict[str, object]: Dictionary representation of the configuration.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ModelETLFactoryConfig":
        """Create a factory configuration from a dictionary.

        Args:
            data: Dictionary containing 'model_name' and 'model_config' keys.

        Returns:
            ModelETLFactoryConfig: Configured factory instance.

        Raises:
            ValueError: If the model name is unknown.
        """
        model_name = str(data.get("model_name"))
        raw_config = data.get("model_config")

        match model_name:
            case "mattersim":
                from phonons_ml.models.implementations.mattersim import (
                    ModelConfigMatterSim,
                )

                model_config = ModelConfigMatterSim.from_dict(raw_config)  # type: ignore[arg-type]
            case "mace":
                from phonons_ml.models.implementations.mace import ModelConfigMACE

                model_config = ModelConfigMACE.from_dict(raw_config)  # type: ignore[arg-type]
            case "upet":
                from phonons_ml.models.implementations.upet import ModelConfigUPET

                model_config = ModelConfigUPET.from_dict(raw_config)  # type: ignore[arg-type]
            case _:
                raise ValueError(f"Unknown model name: {model_name}")

        return cls(model_name=model_name, model_config=model_config)


class ModelETLFactory:
    """Factory class for creating model instances (ETL calculators).

    This factory creates concrete model implementations based on the provided
    configuration.
    """

    @classmethod
    def create_etl(cls, config: ModelETLFactoryConfig) -> ModelBase:
        """Create a model instance based on the factory configuration.

        Args:
            config: Factory configuration containing model name and its config.

        Returns:
            ModelBase: Instance of the requested model.

        Raises:
            ValueError: If the model name is unknown.
        """
        match config.model_name:
            case "mattersim":
                from phonons_ml.models.implementations.mattersim import ModelMatterSim

                return ModelMatterSim(config.model_config)
            case "mace":
                from phonons_ml.models.implementations.mace import ModelMACE

                return ModelMACE(config.model_config)
            case "upet":
                from phonons_ml.models.implementations.upet import ModelUPET

                return ModelUPET(config.model_config)
            case _:
                raise ValueError(f"Unknown model name: {config.model_name}")
