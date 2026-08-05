from dataclasses import asdict, dataclass
from typing import Any

from phonons_ml.calculator.phonon import PhononCalculator, PhononConfig
from phonons_ml.evaluation.properties import (
    ThermodynamicAnalyzer,
    ThermodynamicConfig,
)
from phonons_ml.evaluation.visualizer import (
    PhononVisualizationConfig,
    PhononVisualizer,
)
from phonons_ml.models.factory import ModelETLFactory, ModelETLFactoryConfig
from phonons_ml.pipeline.base import PipelineBase, PipelineConfigBase


@dataclass(slots=True)
class PipelineConfigSingle(PipelineConfigBase):
    """Configuration for a single-structure phonon pipeline.

    Attributes:
        material_id: Unique identifier for the material.
        calculator_config: Configuration for the ML calculator factory.
        phonon_config: Configuration for the phonon calculator.
        visualizer_config: Configuration for the band structure visualizer.
        properties_config: Configuration for thermodynamic analysis.
    """

    material_id: str
    calculator_config: ModelETLFactoryConfig
    phonon_config: PhononConfig
    visualizer_config: PhononVisualizationConfig
    properties_config: ThermodynamicConfig

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfigSingle":
        """Create configuration from a dictionary."""
        return cls(
            material_id=data.get("material_id"),
            calculator_config=ModelETLFactoryConfig.from_dict(
                data.get("calculator_config")
            ),
            phonon_config=PhononConfig.from_dict(data.get("phonon_config")),
            visualizer_config=PhononVisualizationConfig.from_dict(
                data.get("visualizer_config")
            ),
            properties_config=ThermodynamicConfig.from_dict(
                data.get("properties_config")
            ),
        )


class PipelineSingle(PipelineBase):
    """Pipeline for performing full phonon analysis on a single structure."""

    def __init__(self, config: PipelineConfigSingle) -> None:
        super().__init__(config)
        self.config: PipelineConfigSingle = config

        self.calculator = ModelETLFactory.create_etl(
            config=self.config.calculator_config
        )
        self.phonon_calculator = PhononCalculator(config=self.config.phonon_config)
        self.visualizer = PhononVisualizer(config=self.config.visualizer_config)
        self.thermo_analyzer = ThermodynamicAnalyzer(
            config=self.config.properties_config
        )

    def run(self, structure: Any = None) -> dict[str, Any]:
        """Execute the full phonon analysis pipeline.

        Args:
            structure: Input structure (ignored, structure taken from phonon_config).

        Returns:
            dict[str, Any]: Summary of thermodynamic properties and stability.
        """
        self.phonon_calculator.calculate_force_constants(self.calculator)

        self.visualizer.plot_band_structure(
            calculator=self.phonon_calculator,
            filename="phonon_bands.pdf",
        )

        self.visualizer.plot_band_structure_with_dos(
            calculator=self.phonon_calculator,
            filename="phonon_bands_dos.pdf",
        )

        self.thermo_analyzer.initialize(
            self.phonon_calculator,
            self.calculator,
        )

        summary = self.thermo_analyzer.get_summary()

        # self.thermo_analyzer.config.output_dir = (
        #     f"data/results_{self.config.model_config.model_name}/{self.config.material_id}"
        # )

        self.thermo_analyzer.save_all()

        return summary
