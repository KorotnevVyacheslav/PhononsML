from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class ModelConfigBase:
    """Base configuration class for machine learning models.

    Provides serialization methods to convert to/from dictionaries.
    """

    def to_dict(self: ModelConfigBase) -> dict[str, object]:
        """Convert the configuration to a dictionary.

        Returns:
            dict[str, object]: Dictionary representation of the configuration.
        """
        return asdict(self)

    @classmethod
    def from_dict(
        cls: type[ModelConfigBase], data: dict[str, object]
    ) -> ModelConfigBase:
        """Create a configuration instance from a dictionary.

        Args:
            data: Dictionary containing configuration parameters.

        Returns:
            ModelConfigBase: Instance of the configuration class.
        """
        return cls(**data)


class ModelBase:
    """Base class for phonon calculation models.

    All model implementations should inherit from this class and implement
    the `calculate` method.
    """

    def __init__(self: ModelBase, config: ModelConfigBase) -> None:
        """Initialize the model with a configuration.

        Args:
            config: Configuration object for the model.
        """
        self.config = config

    def calculate(self: ModelBase, structure: object) -> dict[str, object]:
        """Calculate phonon properties for a given structure.

        Args:
            structure: Input structure object (e.g., ASE Atoms).

        Returns:
            dict[str, object]: Dictionary containing calculated phonon properties.

        Raises:
            NotImplementedError: If the method is not overridden by a subclass.
        """
        raise NotImplementedError
