import os
from dataclasses import asdict, dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from jarvis.core.kpoints import Kpoints3D
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from pymatgen.io.jarvis import JarvisAtomsAdaptor

from phonons_ml.calculator.phonon import PhononCalculator
from phonons_ml.models.base import ModelBase

THZ_TO_CM = 33.35641
TOLERANCE = -1e-3


@dataclass(slots=True)
class PhononVisualizationConfig:
    """Configuration for phonon band structure visualization.

    Attributes:
        units: Frequency units ('THz' or 'cm-1').
        line_density: Number of k-points per segment in the band path.
        stability_threshold: Frequency threshold (THz) below which modes are considered imaginary.
        color: Color for real modes.
        imag_color: Color for imaginary modes.
        title: Plot title (None for no title).
        line_width: Line width for band curves.
        figure_width: Width of the figure in inches.
        figure_height: Height of the figure in inches.
        dpi: Resolution for saved figures.
        show_zero_line: Whether to draw a horizontal line at zero frequency.
        show_vertical_lines: Whether to draw vertical lines at high-symmetry points.
        shade_imaginary: Whether to shade the region below zero for imaginary modes.
        font_family: Base font family for the plot.
        font_size: Base font size.
        output_format: File format for saved figures (e.g., 'pdf', 'png').
        output_dir: Directory where figures will be saved.
    """

    units: str = "THz"
    line_density: int = 30
    stability_threshold: float = -0.1
    color: str = "#2A6F97"
    imag_color: str = "#C1121F"
    title: str | None = None
    line_width: float = 1.1
    figure_width: float = 3.4
    figure_height: float = 2.8
    dpi: int = 600
    show_zero_line: bool = True
    show_vertical_lines: bool = True
    shade_imaginary: bool = True
    font_family: str = "sans-serif"
    font_size: int = 8
    output_format: str = "pdf"
    output_dir: str = "."

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary.

        Returns:
            dict: Dictionary representation.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhononVisualizationConfig":
        """Create configuration from a dictionary.

        Args:
            data: Dictionary with configuration parameters.

        Returns:
            PhononVisualizationConfig: Reconstructed configuration.
        """
        return cls(**data)


class PhononVisualizer:
    """Visualizer for phonon band structures and density of states.

    This class provides methods to plot phonon band structures along a k-path
    and optionally overlay the total density of states. It supports customizable
    styles and output formats.
    """

    def __init__(self, config: PhononVisualizationConfig) -> None:
        """Initialize the visualizer with configuration.

        Args:
            config: Visualization configuration object.
        """
        self.config = config
        self._setup_style()
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        """Create the output directory if it does not exist."""
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir, exist_ok=True)

    def _setup_style(self) -> None:
        """Apply matplotlib style settings from configuration."""
        rcParams.update(
            {
                "font.family": self.config.font_family,
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                "font.size": self.config.font_size,
                "axes.linewidth": 0.8,
                "xtick.major.width": 0.8,
                "ytick.major.width": 0.8,
                "xtick.direction": "in",
                "ytick.direction": "in",
                "mathtext.fontset": "dejavusans",
            }
        )

    def _get_kpoints_and_labels(
        self, calculator: PhononCalculator
    ) -> tuple[np.ndarray, list[list[float]], list[float], list[str]]:
        """Generate k-point path and high-symmetry labels.

        Args:
            calculator: PhononCalculator instance containing the structure.

        Returns:
            tuple: (distances, kpoints, label_positions, label_names)
                - distances: cumulative distances along the path.
                - kpoints: list of k-point coordinates.
                - label_positions: positions of high-symmetry labels.
                - label_names: formatted labels (Gamma replaced with symbol).
        """
        structure = calculator.config.structure
        jarvis_atoms = JarvisAtomsAdaptor.get_atoms(structure)
        kpoints = Kpoints3D().kpath(jarvis_atoms, line_density=self.config.line_density)

        labels = kpoints.labels
        kpts = kpoints.kpts

        distances = []
        current_dist = 0.0
        for i, kpt in enumerate(kpts):
            if i == 0:
                distances.append(0.0)
            else:
                diff = np.array(kpt) - np.array(kpts[i - 1])
                current_dist += np.linalg.norm(diff)
                distances.append(current_dist)

        label_positions = []
        label_names = []

        for i, label in enumerate(labels):
            if label:
                if i == 0:
                    label_positions.append(0.0)
                else:
                    label_positions.append(distances[i])

                if "Gamma" in label or "GAMMA" in label.upper():
                    label_names.append(r"$\Gamma$")
                else:
                    label_names.append(label)

        return np.array(distances), kpts, label_positions, label_names

    def _calculate_band_data(
        self, calculator: PhononCalculator, phonon: Any
    ) -> tuple[np.ndarray, np.ndarray, list[float], list[str]]:
        """Compute phonon frequencies along the k-path.

        Args:
            calculator: PhononCalculator instance.
            phonon: Phonopy object with force constants.

        Returns:
            tuple: (distances, frequencies, label_positions, label_names)
                where frequencies are in the requested units.
        """
        distances, kpts, label_positions, label_names = self._get_kpoints_and_labels(
            calculator
        )

        frequencies = []
        for k in kpts:
            qpoint_result = phonon.run_qpoints([k])
            freqs = qpoint_result.frequencies[0]  # shape (nbands,)
            frequencies.append(freqs)

        frequencies = np.array(frequencies)  # shape (nkpoints, nbands)

        if self.config.units == "cm-1":
            frequencies = frequencies * THZ_TO_CM

        return distances, frequencies, label_positions, label_names

    def _get_output_path(self, filename: str) -> str:
        """Get the full output path for a filename.

        Args:
            filename: Base filename.

        Returns:
            str: Full path in the output directory.
        """
        return os.path.join(self.config.output_dir, filename)

    def _save_figure(self, fig: plt.Figure, filename: str) -> None:
        """Save figure in the configured format and also as PNG.

        Args:
            fig: Matplotlib figure.
            filename: Base filename (extension may be added).
        """
        base, ext = os.path.splitext(filename)
        if not ext:
            ext = f".{self.config.output_format}"
            filename = base + ext

        full_path = self._get_output_path(filename)
        fig.savefig(full_path, dpi=self.config.dpi, bbox_inches="tight")

        png_name = base + ".png"
        full_png = self._get_output_path(png_name)
        fig.savefig(full_png, dpi=self.config.dpi, bbox_inches="tight")

    def _ensure_force_constants(
        self, calculator: PhononCalculator, model_calculator: ModelBase | None
    ) -> None:
        """Ensure force constants are available; raise error if not.

        Args:
            calculator: PhononCalculator instance.
            model_calculator: ML model calculator (required if force constants not yet computed).

        Raises:
            ValueError: If force constants are missing and no model_calculator is provided.
        """
        if calculator.force_constants is None and model_calculator is None:
            raise ValueError(
                "Force constants not computed. Provide a model_calculator to compute them."
            )

    def plot_band_structure(
        self,
        calculator: PhononCalculator,
        model_calculator: ModelBase | None = None,
        filename: str | None = None,
        show: bool = False,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the phonon band structure.

        Args:
            calculator: PhononCalculator instance with the structure.
            model_calculator: ML model for force calculations (required if force constants missing).
            filename: Optional filename to save the figure (without extension uses configured format).
            show: If True, display the figure interactively.

        Returns:
            tuple: (figure, axes) of the plot.

        Raises:
            ValueError: If force constants are not available and no model_calculator is provided.
        """
        self._ensure_force_constants(calculator, model_calculator)
        phonon = calculator.get_phonon(model_calculator)  # type: ignore[arg-type]
        distances, frequencies, label_positions, label_names = (
            self._calculate_band_data(calculator, phonon)
        )

        unit_label = (
            "Frequency (THz)"
            if self.config.units == "THz"
            else r"Frequency (cm$^{-1}$)"
        )

        fig, ax = plt.subplots(
            figsize=(self.config.figure_width, self.config.figure_height)
        )

        nband = frequencies.shape[1]
        for b in range(nband):
            y = frequencies[:, b]
            is_imag = np.any(y < TOLERANCE)
            color = self.config.imag_color if is_imag else self.config.color
            ax.plot(
                distances,
                y,
                color=color,
                lw=self.config.line_width,
                solid_capstyle="round",
                zorder=3 if is_imag else 2,
            )

        if self.config.show_zero_line:
            ax.axhline(0.0, color="0.55", lw=0.7, ls=(0, (4, 3)), zorder=1)

        if self.config.show_vertical_lines:
            for t in label_positions[1:-1]:
                ax.axvline(t, color="0.75", lw=0.7, zorder=1)

        ax.set_xticks(label_positions)
        ax.set_xticklabels(label_names)
        ax.set_xlim(distances[0], distances[-1])

        ymin = min(-1.5, frequencies.min() * 1.15)
        ymax = frequencies.max() * 1.05
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(unit_label)

        if self.config.title:
            ax.set_title(self.config.title, fontsize=9, pad=4)

        if self.config.shade_imaginary and frequencies.min() < TOLERANCE:
            ax.axhspan(ymin, 0, color=self.config.imag_color, alpha=0.05, zorder=0)

        fig.tight_layout(pad=0.4)

        if filename:
            self._save_figure(fig, filename)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax

    def plot_band_structure_with_dos(
        self,
        calculator: PhononCalculator,
        model_calculator: ModelBase | None = None,
        filename: str | None = None,
        show: bool = False,
        mesh_density: int = 40,
        dos_color: tuple[float, float, float, float] = (0.2, 0.4, 0.6, 0.6),
    ) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
        """Plot phonon band structure with total density of states.

        Args:
            calculator: PhononCalculator instance.
            model_calculator: ML model for force calculations (required if force constants missing).
            filename: Optional filename to save the figure.
            show: If True, display the figure interactively.
            mesh_density: Mesh density for DOS calculation.
            dos_color: RGBA color for the DOS fill.

        Returns:
            tuple: (figure, (band_axes, dos_axes)).

        Raises:
            ValueError: If force constants are not available and no model_calculator is provided.
        """
        self._ensure_force_constants(calculator, model_calculator)
        phonon = calculator.get_phonon(model_calculator)  # type: ignore[arg-type]
        distances, frequencies, label_positions, label_names = (
            self._calculate_band_data(calculator, phonon)
        )

        phonon.run_mesh(
            [mesh_density, mesh_density, mesh_density],
            is_gamma_center=True,
            is_mesh_symmetry=False,
        )
        dos_result = phonon.run_total_dos()
        freqs_dos = dos_result.frequency_points
        dos = dos_result.dos

        if self.config.units == "cm-1":
            freqs_dos = freqs_dos * THZ_TO_CM

        unit_label = (
            "Frequency (THz)"
            if self.config.units == "THz"
            else r"Frequency (cm$^{-1}$)"
        )

        fig = plt.figure(
            figsize=(self.config.figure_width * 1.5, self.config.figure_height)
        )
        gs = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.0)

        ax_band = plt.subplot(gs[0])
        ax_dos = plt.subplot(gs[1])

        nband = frequencies.shape[1]
        for b in range(nband):
            y = frequencies[:, b]
            is_imag = np.any(y < TOLERANCE)
            color = self.config.imag_color if is_imag else self.config.color
            ax_band.plot(
                distances,
                y,
                color=color,
                lw=self.config.line_width,
                solid_capstyle="round",
                zorder=3 if is_imag else 2,
            )

        if self.config.show_zero_line:
            ax_band.axhline(0.0, color="0.55", lw=0.7, ls=(0, (4, 3)), zorder=1)

        if self.config.show_vertical_lines:
            for t in label_positions[1:-1]:
                ax_band.axvline(t, color="0.75", lw=0.7, zorder=1)

        ax_band.set_xticks(label_positions)
        ax_band.set_xticklabels(label_names)
        ax_band.set_xlim(distances[0], distances[-1])

        ymin = min(-1.5, frequencies.min() * 1.15, freqs_dos.min() * 1.15)
        ymax = max(frequencies.max() * 1.05, freqs_dos.max() * 1.05)
        ax_band.set_ylim(ymin, ymax)
        ax_band.set_ylabel(unit_label)

        if self.config.title:
            ax_band.set_title(self.config.title, fontsize=9, pad=4)

        if self.config.shade_imaginary and frequencies.min() < TOLERANCE:
            ax_band.axhspan(ymin, 0, color=self.config.imag_color, alpha=0.05, zorder=0)

        ax_dos.fill_between(
            dos,
            freqs_dos,
            color=dos_color,
            edgecolor="k",
            lw=1,
            y2=0,
        )
        ax_dos.set_xlabel("DOS")
        ax_dos.set_yticks([])
        ax_dos.set_xticks([])
        ax_dos.set_ylim(ymin, ymax)
        ax_dos.set_xlim(0, max(dos) * 1.05)

        fig.tight_layout(pad=0.4)

        if filename:
            self._save_figure(fig, filename)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, (ax_band, ax_dos)

    def save(
        self,
        calculator: PhononCalculator,
        model_calculator: ModelBase | None = None,
        filename: str = "phonon_plot.pdf",
    ) -> None:
        """Convenience method to save a band structure plot.

        Args:
            calculator: PhononCalculator instance.
            model_calculator: ML model for force calculations.
            filename: Output filename (extension may be overridden by config format).
        """
        self.plot_band_structure(
            calculator, model_calculator=model_calculator, filename=filename
        )
