import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl

from phonons_ml.calculator.phonon import PhononCalculator
from phonons_ml.models.base import ModelBase

THZ_TO_CM = 33.35641
H_BAR_EV_THZ = 0.004135667696
KB_EV = 8.617333262145e-5

MAX_OMEGA = 1.5
LOW_SCORE = 0.4
HIGH_SCORE = 0.7
DELTA_E = 0.1  # 100 meV heuristic barrier


@dataclass(slots=True)
class ThermodynamicConfig:
    """Configuration for thermodynamic property analysis.

    Attributes:
        material_id: Unique identifier for the material.
        t_min: Minimum temperature in Kelvin.
        t_max: Maximum temperature in Kelvin.
        t_step: Temperature step in Kelvin.
        mesh_density: Density of k-point mesh for phonon calculations.
        reference_temperature: Reference temperature for free energy (unused, kept for compatibility).
        stability_temperature: Temperature at which stability is assessed.
        imaginary_threshold_thz: Frequency threshold (THz) for classifying high instability.
        imaginary_threshold_cm: Frequency threshold (cm⁻¹) – kept for reference.
        output_dir: Directory to save output files.
    """

    material_id: str
    t_min: float = 0
    t_max: float = 1000
    t_step: float = 10
    mesh_density: int = 40
    reference_temperature: float = 300
    stability_temperature: float = 300
    imaginary_threshold_thz: float = 5.0
    imaginary_threshold_cm: float = 160.0
    output_dir: str = "."

    def __post_init__(self) -> None:
        """Create output directory if it does not exist."""
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThermodynamicConfig":
        """Create configuration from dictionary."""
        return cls(**data)


class ThermodynamicAnalyzer:
    """Analyzer for thermodynamic properties and stability from phonon calculations.

    This class computes thermal properties (free energy, entropy, heat capacity)
    and evaluates dynamic stability based on imaginary modes. It provides
    dataframes and summary reports.
    """

    def __init__(self, config: ThermodynamicConfig) -> None:
        """Initialize the analyzer with configuration.

        Args:
            config: Thermodynamic configuration.
        """
        self.config = config
        self._phonon = None
        self._model_calculator: ModelBase | None = None
        self.max_imag_freq_thz = 0.0
        self.max_imag_freq_cm = 0.0
        self.n_imag_modes = 0
        self.has_imaginary = False

    def initialize(
        self, calculator: PhononCalculator, model_calculator: ModelBase
    ) -> None:
        """Initialize with phonon data and compute mesh and thermal properties.

        Args:
            calculator: PhononCalculator instance containing the structure and force constants.
            model_calculator: ML model calculator used for force constants (required for get_phonon).
        """
        self._model_calculator = model_calculator
        self._phonon = calculator.get_phonon(model_calculator)
        self._run_mesh()
        self._run_thermal_properties()
        self._analyze_imaginary_modes()

    def _run_mesh(self) -> None:
        """Run phonon mesh integration."""
        if self._phonon is None:
            raise RuntimeError(
                "Phonon object not initialized. Call initialize() first."
            )
        self._phonon.run_mesh(
            [self.config.mesh_density] * 3,
            is_gamma_center=True,
            is_mesh_symmetry=False,
        )

    def _run_thermal_properties(self) -> None:
        """Compute thermal properties over the temperature range."""
        if self._phonon is None:
            raise RuntimeError(
                "Phonon object not initialized. Call initialize() first."
            )
        self._phonon.run_thermal_properties(
            t_min=self.config.t_min,
            t_max=self.config.t_max,
            t_step=self.config.t_step,
        )

    def _analyze_imaginary_modes(self) -> None:
        """Analyze imaginary frequencies from the phonon mesh."""
        if self._phonon is None:
            raise RuntimeError(
                "Phonon object not initialized. Call initialize() first."
            )
        mesh = self._phonon.mesh
        if mesh is None:
            raise RuntimeError("Mesh not computed. Call _run_mesh() first.")
        frequencies = mesh.frequencies

        imaginary_freqs = frequencies[frequencies < 0]

        if len(imaginary_freqs) > 0:
            self.max_imag_freq_thz = abs(np.min(imaginary_freqs))
            self.max_imag_freq_cm = self.max_imag_freq_thz * THZ_TO_CM
            self.n_imag_modes = len(imaginary_freqs)
            self.has_imaginary = True
        else:
            self.max_imag_freq_thz = 0.0
            self.max_imag_freq_cm = 0.0
            self.n_imag_modes = 0
            self.has_imaginary = False

    def _calculate_distortion_energy(self) -> float | None:
        """Estimate distortion energy from the largest imaginary mode.

        Returns:
            float | None: Distortion energy in eV, or None if no imaginary modes.
        """
        if not self.has_imaginary:
            return None
        # Zero-point energy of an oscillator with imaginary frequency (abs value)
        return 0.5 * H_BAR_EV_THZ * self.max_imag_freq_thz

    def _get_stability_criteria(
        self, temperature: float | None = None
    ) -> dict[str, Any]:
        """Compute stability criteria based on imaginary modes and temperature.

        Args:
            temperature: Temperature in K (if None, uses stability_temperature).

        Returns:
            dict: Stability metrics including status, score, and detailed criteria.
        """
        criteria: dict[str, Any] = {}

        if not self.has_imaginary:
            criteria["stability_status"] = "stable"
            criteria["stability_score"] = 1.0
            criteria["thermal_stabilization"] = "N/A"
            criteria["frequency_regime"] = "stable"
            criteria["stability_potential"] = "stable"
            criteria["distortion_energy_meV"] = 0.0
            criteria["distortion_criterion"] = "stable"
            return criteria

        T = (
            temperature
            if temperature is not None
            else self.config.stability_temperature
        )
        omega_thz = self.max_imag_freq_thz

        E_imag_ev = H_BAR_EV_THZ * omega_thz
        kT_ev = KB_EV * T

        thermal_ratio = E_imag_ev / kT_ev if kT_ev > 0 else float("inf")
        criteria["thermal_ratio"] = thermal_ratio
        criteria["thermal_stabilization"] = (
            "likely" if thermal_ratio < 1.0 else "unlikely"
        )

        # Frequency regime classification
        if omega_thz < MAX_OMEGA:
            criteria["frequency_regime"] = "low"
            criteria["stability_potential"] = "high"
        elif omega_thz < self.config.imaginary_threshold_thz:
            criteria["frequency_regime"] = "medium"
            criteria["stability_potential"] = "moderate"
        else:
            criteria["frequency_regime"] = "high"
            criteria["stability_potential"] = "low"

        delta_E = self._calculate_distortion_energy()
        if delta_E is not None:
            criteria["distortion_energy_meV"] = delta_E * 1000.0
            if delta_E < kT_ev:
                criteria["distortion_criterion"] = "thermal_accessible"
            elif delta_E < DELTA_E:  # 100 meV heuristic barrier
                criteria["distortion_criterion"] = "metastable"
            else:
                criteria["distortion_criterion"] = "unstable"

        # Stability score (0 to 1)
        score = 0.0
        if criteria.get("thermal_ratio", float("inf")) < 1.0:
            score += 0.4
        regime = criteria.get("frequency_regime", "")
        if regime == "low":
            score += 0.3
        elif regime == "medium":
            score += 0.15
        dist_crit = criteria.get("distortion_criterion", "")
        if dist_crit == "thermal_accessible":
            score += 0.3
        elif dist_crit == "metastable":
            score += 0.15

        criteria["stability_score"] = min(score, 1.0)

        if score > HIGH_SCORE:
            criteria["stability_status"] = "likely_stable"
        elif score > LOW_SCORE:
            criteria["stability_status"] = "metastable"
        else:
            criteria["stability_status"] = "unstable"

        return criteria

    def get_dataframe(self) -> pl.DataFrame:
        """Get a DataFrame with thermal properties and stability metrics for each temperature.

        Returns:
            pl.DataFrame: Table with columns for temperature, thermodynamic quantities,
                and stability indicators.

        Raises:
            RuntimeError: If analyzer is not initialized.
        """
        if self._phonon is None:
            raise RuntimeError("Analyzer is not initialized with phonon data.")

        thermal = self._phonon.thermal_properties
        if thermal is None:
            raise RuntimeError(
                "Thermal properties not computed. Call initialize() first."
            )

        temperatures = thermal.temperatures
        free_energy = thermal.free_energy
        entropy = thermal.entropy
        heat_capacity = thermal.heat_capacity
        zero_point_energy = thermal.zero_point_energy

        data = []
        for i, T in enumerate(temperatures):
            criteria = self._get_stability_criteria(T)
            row = {
                "material_id": self.config.material_id,
                "temperature_K": T,
                "free_energy_eV": free_energy[i],
                "entropy_eV_K": entropy[i],
                "heat_capacity_eV_K": heat_capacity[i],
                "zero_point_energy_eV": zero_point_energy,
                "has_imaginary_modes": self.has_imaginary,
                "max_imag_freq_THz": self.max_imag_freq_thz,
                "max_imag_freq_cm": self.max_imag_freq_cm,
                "n_imaginary_modes": self.n_imag_modes,
                "thermal_ratio_Eimag_kT": criteria.get("thermal_ratio", float("inf")),
                "thermal_stabilization": criteria.get(
                    "thermal_stabilization", "unknown"
                ),
                "frequency_regime": criteria.get("frequency_regime", "stable"),
                "stability_potential": criteria.get("stability_potential", "stable"),
                "distortion_energy_meV": criteria.get("distortion_energy_meV", 0.0),
                "distortion_criterion": criteria.get("distortion_criterion", "stable"),
                "stability_status": criteria.get("stability_status", "stable"),
                "stability_score": criteria.get("stability_score", 1.0),
            }
            data.append(row)

        return pl.DataFrame(data)

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of stability analysis at the configured stability temperature.

        Returns:
            dict: Summary dictionary with key stability indicators and recommendation.
        """
        criteria = self._get_stability_criteria(self.config.stability_temperature)
        return {
            "material_id": self.config.material_id,
            "has_imaginary_modes": self.has_imaginary,
            "max_imag_freq_THz": self.max_imag_freq_thz,
            "max_imag_freq_cm": self.max_imag_freq_cm,
            "n_imaginary_modes": self.n_imag_modes,
            "stability_status": criteria.get("stability_status", "stable"),
            "stability_score": criteria.get("stability_score", 1.0),
            "thermal_stabilization": criteria.get("thermal_stabilization", "N/A"),
            "frequency_regime": criteria.get("frequency_regime", "stable"),
            "stability_potential": criteria.get("stability_potential", "stable"),
            "distortion_energy_meV": criteria.get("distortion_energy_meV", 0.0),
            "distortion_criterion": criteria.get("distortion_criterion", "stable"),
            "recommendation": self._get_recommendation(criteria),
        }

    def get_summary_dataframe(self) -> pl.DataFrame:
        """Get a DataFrame with the summary (single row)."""
        return pl.DataFrame([self.get_summary()])

    @staticmethod
    def _get_recommendation(criteria: dict[str, Any]) -> str:
        """Generate a textual recommendation based on stability criteria."""
        if criteria.get("stability_status") == "stable":
            return "Structure is dynamically stable at 0 K."

        status = criteria.get("stability_status", "unknown")
        if status == "likely_stable":
            return "Structure likely exists. Thermal effects should stabilize it."
        if status == "metastable":
            return "Structure is metastable. Consider using SSCHA or SCAILD for accurate Tc."
        return "Structure is highly unstable. Unlikely to exist in real conditions."

    def save_thermal_properties_parquet(
        self, filename: str = "thermal_properties.parquet"
    ) -> str:
        """Save thermal properties DataFrame to Parquet.

        Args:
            filename: Output filename (saved in config.output_dir).

        Returns:
            str: Full path to the saved file.
        """
        df = self.get_dataframe()
        full_path = os.path.join(self.config.output_dir, filename)
        df.write_parquet(full_path)
        return full_path

    def save_summary_parquet(self, filename: str = "summary.parquet") -> str:
        """Save summary DataFrame to Parquet.

        Args:
            filename: Output filename (saved in config.output_dir).

        Returns:
            str: Full path to the saved file.
        """
        df = self.get_summary_dataframe()
        full_path = os.path.join(self.config.output_dir, filename)
        df.write_parquet(full_path)
        return full_path

    def save_all(self, prefix: str = "") -> None:
        """Save both thermal properties and summary with an optional prefix.

        Args:
            prefix: Prefix to prepend to filenames.
        """
        if prefix:
            self.save_thermal_properties_parquet(f"{prefix}_thermal_properties.parquet")
            self.save_summary_parquet(f"{prefix}_summary.parquet")
        else:
            self.save_thermal_properties_parquet("thermal_properties.parquet")
            self.save_summary_parquet("summary.parquet")
