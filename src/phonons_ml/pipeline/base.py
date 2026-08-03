from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class PipelineConfigBase:
    """Base configuration class for pipelines.

    Provides serialization methods to convert to/from dictionaries.
    """

    def to_dict(self) -> dict[str, Any]:
        """Convert the configuration to a dictionary.

        Returns:
            dict[str, Any]: Dictionary representation of the configuration.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfigBase":
        """Create a configuration instance from a dictionary.

        Args:
            data: Dictionary containing configuration parameters.

        Returns:
            PipelineConfigBase: Instance of the configuration class.
        """
        return cls(**data)


class PipelineBase:
    """Base class for phonon calculation pipelines.

    All pipeline implementations should inherit from this class and implement
    the `run` method.
    """

    def __init__(self, config: PipelineConfigBase) -> None:
        """Initialize the pipeline with a configuration.

        Args:
            config: Configuration object for the pipeline.
        """
        self.config = config

    def run(self, structure: Any) -> dict[str, Any]:
        """Run the pipeline on a given structure.

        Args:
            structure: Input structure object (e.g., ASE Atoms, pymatgen Structure).

        Returns:
            dict[str, Any]: Dictionary containing pipeline results.

        Raises:
            NotImplementedError: If the method is not overridden by a subclass.
        """
        raise NotImplementedError
