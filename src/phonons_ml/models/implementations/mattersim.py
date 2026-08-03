from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms as AseAtoms

from phonons_ml.models.base import ModelBase, ModelConfigBase

try:
    from mattersim.forcefield import MatterSimCalculator
except ImportError as e:
    msg = "MatterSim package is required. Install with: pip install mattersim"
    raise ImportError(msg) from e

FORCE_DRIFT_THRESHOLD = 1e-6


@dataclass(slots=True)
class ModelConfigMatterSim(ModelConfigBase):
    """Configuration class for MatterSim model.

    Attributes:
        device: Computing device ('cpu' or 'cuda').
        default_dtype: Data type for calculations ('float64' or 'float32').
        checkpoint_path: Optional path to a custom checkpoint file.
        compute_stress: Whether to compute stress tensor.
        compute_virials: Whether to compute virials.
        verbose: Enable verbose output.
        enable_energy: Whether to enable energy calculation.
        enable_forces: Whether to enable forces calculation.
        enable_stress: Whether to enable stress calculation (requires compute_stress).
        enable_virials: Whether to enable virials calculation (requires compute_virials).
        model_size: Model size ('1M' or '5M').
        neighbor_cutoff: Cutoff radius for neighbor list in Angstroms.
    """

    device: str = "cpu"
    default_dtype: str = "float64"
    checkpoint_path: str | None = None
    compute_stress: bool = False
    compute_virials: bool = False
    verbose: bool = False

    enable_energy: bool = True
    enable_forces: bool = True
    enable_stress: bool = False
    enable_virials: bool = False

    model_size: str = "1M"
    neighbor_cutoff: float = 8.0

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        if self.device not in ("cpu", "cuda"):
            msg = "device must be 'cpu' or 'cuda'"
            raise ValueError(msg)

        if self.default_dtype not in ("float64", "float32"):
            msg = "default_dtype must be 'float64' or 'float32'"
            raise ValueError(msg)

        if self.model_size not in ("1M", "5M"):
            msg = "model_size must be '1M' or '5M'"
            raise ValueError(msg)

        if self.checkpoint_path is not None and not Path(self.checkpoint_path).exists():
            msg = f"Checkpoint file not found: {self.checkpoint_path}"
            raise FileNotFoundError(msg)

        if self.device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    msg = "CUDA is not available on this system"
                    raise ValueError(msg)
            except ImportError as e:
                msg = "PyTorch is required for GPU support"
                raise ImportError(msg) from e

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ModelConfigMatterSim:  # type: ignore[misc]
        """Create configuration instance from a dictionary.

        Args:
            data: Dictionary with configuration parameters.

        Returns:
            ModelConfigMatterSim: Configured instance.
        """
        return cls(**data)


class ModelMatterSim(ModelBase):
    """MatterSim-based phonon calculator model.

    This class wraps the MatterSim force field to compute energies, forces,
    stresses, and virials for atomic structures. It includes caching and
    batch processing capabilities.
    """

    def __init__(self, config: ModelConfigMatterSim) -> None:
        """Initialize the MatterSim model with given configuration.

        Args:
            config: Configuration object for MatterSim.
        """
        super().__init__(config)
        self.config = config
        self._calculator: MatterSimCalculator | None = None
        self._is_initialized: bool = False
        self._cache: dict[str, dict] = {}
        self._cache_size: int = 100
        self._initialize_calculator()

    def _initialize_calculator(self) -> None:
        """Initialize the MatterSim calculator with current configuration."""
        if self.config.verbose:
            print(f"Initializing MatterSim calculator on {self.config.device}")

        device_str = "cuda" if self.config.device == "cuda" else "cpu"

        if self.config.checkpoint_path is not None:
            load_path = self.config.checkpoint_path
            if self.config.verbose:
                print(f"Loading checkpoint from: {load_path}")
        else:
            load_path = (
                "MatterSim-v1.0.0-1M.pth"
                if self.config.model_size == "1M"
                else "MatterSim-v1.0.0-5M.pth"
            )
            if self.config.verbose:
                print(f"Using pretrained MatterSim model: {load_path}")

        self._calculator = MatterSimCalculator(load_path=load_path, device=device_str)
        self._is_initialized = True
        if self.config.verbose:
            print(f"MatterSim calculator initialized on {self.config.device}")

    def set_device(self, device: str) -> None:
        """Switch the computing device.

        Args:
            device: New device ('cpu' or 'cuda').

        Raises:
            ValueError: If device is invalid or CUDA is not available.
            ImportError: If PyTorch is required but not installed.
        """
        if device not in ("cpu", "cuda"):
            msg = "device must be 'cpu' or 'cuda'"
            raise ValueError(msg)
        if device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    msg = "CUDA is not available on this system"
                    raise ValueError(msg)
            except ImportError as e:
                msg = "PyTorch is required for GPU support"
                raise ImportError(msg) from e

        self.config.device = device
        self._is_initialized = False
        self.clear_cache()
        self._initialize_calculator()
        if self.config.verbose:
            print(f"Switched to device: {device}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a custom checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        if not Path(checkpoint_path).exists():
            msg = f"Checkpoint file not found: {checkpoint_path}"
            raise FileNotFoundError(msg)

        self.config.checkpoint_path = checkpoint_path
        self._is_initialized = False
        self.clear_cache()
        self._initialize_calculator()
        if self.config.verbose:
            print(f"Loaded checkpoint: {checkpoint_path}")

    def calculate(
        self,
        structure: AseAtoms,  # type: ignore
        return_energy: bool = True,
        return_forces: bool = True,
        return_stress: bool = False,
        return_virials: bool = False,
        use_cache: bool = True,
    ) -> dict[str, object]:
        """Calculate phonon properties for a single structure.

        Args:
            structure: Input structure (ASE Atoms, Phonopy Atoms, or pymatgen Structure).
            return_energy: Whether to compute and return energy.
            return_forces: Whether to compute and return forces.
            return_stress: Whether to compute and return stress.
            return_virials: Whether to compute and return virials.
            use_cache: Whether to use cached results.

        Returns:
            dict: Dictionary containing computed properties and metadata.
        """
        ase_atoms = self._convert_to_ase(structure)
        structure_hash = self._get_structure_hash(ase_atoms)

        if use_cache and structure_hash in self._cache:
            if self.config.verbose:
                print(f"Using cached results for structure {structure_hash}")
            return self._cache[structure_hash]

        if not self._is_initialized:
            self._initialize_calculator()

        ase_atoms.calc = self._calculator

        results: dict[str, object] = {
            "structure_hash": structure_hash,
            "n_atoms": len(ase_atoms),
        }

        # Helper to safely compute properties
        self._compute_property(
            ase_atoms,
            results,
            return_energy,
            self.config.enable_energy,
            "energy",
            "get_potential_energy",
            lambda atoms, val: {"energy": val, "energy_per_atom": val / len(atoms)},
        )

        if return_forces or self.config.enable_forces:
            try:
                forces = ase_atoms.get_forces()
                drift_force = forces.sum(axis=0)
                if np.any(np.abs(drift_force) > FORCE_DRIFT_THRESHOLD):
                    forces -= drift_force / forces.shape[0]
                    if self.config.verbose:
                        print(f"Corrected force drift: {drift_force}")
                results["forces"] = forces
            except Exception as e:
                if self.config.verbose:
                    print(f"Error calculating forces: {e}")
                results["forces"] = None

        if (return_stress or self.config.enable_stress) and self.config.compute_stress:
            try:
                stress = ase_atoms.get_stress()
                results["stress"] = stress
            except Exception as e:
                if self.config.verbose:
                    print(f"Error calculating stress: {e}")
                results["stress"] = None

        if (
            return_virials or self.config.enable_virials
        ) and self.config.compute_virials:
            try:
                virials = ase_atoms.get_virial()
                results["virials"] = virials
            except Exception as e:
                if self.config.verbose:
                    print(f"Error calculating virials: {e}")
                results["virials"] = None

        if use_cache:
            self._add_to_cache(structure_hash, results)

        return results

    def _compute_property(
        self,
        atoms: AseAtoms,
        results: dict[str, object],
        requested: bool,
        enabled: bool,
        name: str,
        method_name: str,
        transform_func,
    ) -> None:
        """Safely compute a property if requested and enabled."""
        if requested or enabled:
            try:
                method = getattr(atoms, method_name)
                value = method()
                if transform_func:
                    transformed = transform_func(atoms, value)
                    results.update(transformed)
                else:
                    results[name] = value
            except Exception as e:
                if self.config.verbose:
                    print(f"Error calculating {name}: {e}")
                results[name] = None

    def calculate_batch(
        self,
        structures: list[AseAtoms],  # type: ignore
        return_energy: bool = True,
        return_forces: bool = True,
        return_stress: bool = False,
        return_virials: bool = False,
        use_cache: bool = True,
    ) -> list[dict[str, object]]:
        """Calculate properties for a batch of structures without progress bar.

        Args:
            structures: List of input structures.
            return_energy: Whether to compute energies.
            return_forces: Whether to compute forces.
            return_stress: Whether to compute stress.
            return_virials: Whether to compute virials.
            use_cache: Whether to use cached results.

        Returns:
            list[dict]: List of result dictionaries for each structure.
        """
        results: list[dict[str, object]] = []
        for i, structure in enumerate(structures):
            if self.config.verbose and i % 10 == 0:
                print(f"Processing structure {i + 1}/{len(structures)}")

            result = self.calculate(
                structure=structure,
                return_energy=return_energy,
                return_forces=return_forces,
                return_stress=return_stress,
                return_virials=return_virials,
                use_cache=use_cache,
            )
            results.append(result)
        return results

    def calculate_batch_with_progress(
        self,
        structures: list[AseAtoms],  # type: ignore
        return_energy: bool = True,
        return_forces: bool = True,
        return_stress: bool = False,
        return_virials: bool = False,
        use_cache: bool = True,
        show_progress: bool = True,
    ) -> list[dict[str, object]]:
        """Calculate properties for a batch with optional progress bar.

        Args:
            structures: List of input structures.
            return_energy: Whether to compute energies.
            return_forces: Whether to compute forces.
            return_stress: Whether to compute stress.
            return_virials: Whether to compute virials.
            use_cache: Whether to use cached results.
            show_progress: Whether to display a progress bar (requires tqdm).

        Returns:
            list[dict]: List of result dictionaries.
        """
        try:
            from tqdm import tqdm

            iterator = (
                tqdm(structures, desc="Calculating") if show_progress else structures
            )
        except ImportError:
            iterator = structures

        results: list[dict[str, object]] = []
        for structure in iterator:
            result = self.calculate(
                structure=structure,
                return_energy=return_energy,
                return_forces=return_forces,
                return_stress=return_stress,
                return_virials=return_virials,
                use_cache=use_cache,
            )
            results.append(result)
        return results

    def _convert_to_ase(
        self,
        structure: object,
    ) -> AseAtoms:
        """Convert various structure types to ASE Atoms.

        Args:
            structure: Input structure (ASE, Phonopy, pymatgen, or compatible).

        Returns:
            AseAtoms: Converted ASE Atoms object.

        Raises:
            ValueError: If the structure type is unsupported.
        """
        if isinstance(structure, AseAtoms):
            return structure

        if hasattr(structure, "symbols") and hasattr(structure, "positions"):
            try:
                from phonopy.structure.atoms import Atoms as PhonopyAtoms

                if isinstance(structure, PhonopyAtoms):
                    return AseAtoms(
                        symbols=structure.symbols,
                        positions=structure.positions,
                        pbc=True,
                        cell=structure.cell,
                    )
            except ImportError:
                pass

        if hasattr(structure, "sites") and hasattr(structure, "lattice"):
            try:
                from pymatgen.io.ase import AseAtomsAdaptor

                return AseAtomsAdaptor.get_atoms(structure)
            except ImportError:
                pass

        if hasattr(structure, "elements"):
            try:
                from pymatgen.io.jarvis import JarvisAtomsAdaptor

                return JarvisAtomsAdaptor.get_atoms(structure)
            except ImportError:
                pass

        try:
            if hasattr(structure, "get_positions"):
                return AseAtoms(
                    symbols=structure.get_chemical_symbols(),
                    positions=structure.get_positions(),
                    cell=structure.get_cell(),
                    pbc=True,
                )
        except Exception:
            pass

        msg = f"Unsupported structure type: {type(structure)}"
        raise ValueError(msg)

    def _get_structure_hash(self, structure: AseAtoms) -> str:
        """Generate a unique hash for a structure.

        Args:
            structure: ASE Atoms object.

        Returns:
            str: MD5 hash string.
        """
        import hashlib

        info = {
            "symbols": structure.get_chemical_symbols(),
            "positions": structure.get_positions().round(6).tolist(),
            "cell": structure.get_cell().tolist(),
        }
        info_str = str(info)
        return hashlib.md5(info_str.encode()).hexdigest()

    def _add_to_cache(self, key: str, value: dict[str, object]) -> None:
        """Add a result to the cache, evicting oldest if full.

        Args:
            key: Cache key (structure hash).
            value: Result dictionary to store.
        """
        if len(self._cache) >= self._cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value

    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._cache.clear()
        if self.config.verbose:
            print("Cache cleared")

    def get_cache_info(self) -> dict[str, object]:
        """Get information about the current cache state.

        Returns:
            dict: Cache size, max size, and first 10 keys.
        """
        return {
            "size": len(self._cache),
            "max_size": self._cache_size,
            "keys": list(self._cache.keys())[:10],
        }

    def get_energy(
        self,
        structure: AseAtoms,  # type: ignore
    ) -> float | None:
        """Convenience method to get only the energy.

        Args:
            structure: Input structure.

        Returns:
            float | None: Energy in eV, or None if calculation failed.
        """
        result = self.calculate(structure, return_energy=True, return_forces=False)
        energy = result.get("energy")
        return float(energy) if energy is not None else None

    def get_forces(
        self,
        structure: AseAtoms,  # type: ignore
    ) -> np.ndarray | None:
        """Convenience method to get only the forces.

        Args:
            structure: Input structure.

        Returns:
            np.ndarray | None: Forces in eV/Å, or None if calculation failed.
        """
        result = self.calculate(structure, return_energy=False, return_forces=True)
        forces = result.get("forces")
        return forces if isinstance(forces, np.ndarray) else None

    def get_energy_and_forces(
        self,
        structure: AseAtoms,  # type: ignore
    ) -> tuple[float | None, np.ndarray | None]:
        """Convenience method to get both energy and forces.

        Args:
            structure: Input structure.

        Returns:
            tuple: (energy in eV, forces in eV/Å). Either may be None if failed.
        """
        result = self.calculate(structure, return_energy=True, return_forces=True)
        energy = result.get("energy")
        forces = result.get("forces")
        return (
            float(energy) if energy is not None else None,
            forces if isinstance(forces, np.ndarray) else None,
        )

    def __repr__(self) -> str:
        """Return a string representation of the model."""
        return f"ModelMatterSim(device={self.config.device}, model_size={self.config.model_size})"

    def __del__(self) -> None:
        """Cleanup cache on deletion."""
        with contextlib.suppress(Exception):
            self.clear_cache()


def create_mattersim_calculator(
    device: str = "cpu",
    checkpoint_path: str | None = None,
    model_size: str = "1M",
    **kwargs: object,
) -> ModelMatterSim:
    """Factory function to create a MatterSim calculator.

    Args:
        device: Computing device ('cpu' or 'cuda').
        checkpoint_path: Optional path to a custom checkpoint.
        model_size: Model size ('1M' or '5M').
        **kwargs: Additional configuration parameters for ModelConfigMatterSim.

    Returns:
        ModelMatterSim: Configured MatterSim model instance.
    """
    config = ModelConfigMatterSim(
        device=device,
        checkpoint_path=checkpoint_path,
        model_size=model_size,
        **kwargs,
    )
    return ModelMatterSim(config)
