import os
from dataclasses import asdict, dataclass

import numpy as np
from ase import Atoms as AseAtoms
from phonopy import Phonopy
from phonopy.file_IO import write_FORCE_CONSTANTS
from phonopy.structure.atoms import Atoms as PhonopyAtoms
from pymatgen.core import Structure
from pymatgen.io.phonopy import get_phonopy_structure

from phonons_ml.models.base import ModelBase

TOLERANCE = 1e-6


def phonopy_to_ase_atoms(phonopy_atoms: PhonopyAtoms, pbc: bool = True) -> AseAtoms:
    """Convert Phonopy Atoms to ASE Atoms.

    Args:
        phonopy_atoms: Phonopy atoms object.
        pbc: Periodic boundary conditions flag.

    Returns:
        AseAtoms: Converted ASE atoms object.
    """
    return AseAtoms(
        symbols=phonopy_atoms.symbols,
        positions=phonopy_atoms.positions,
        pbc=pbc,
        cell=phonopy_atoms.cell,
    )


def ase_to_phonopy_atoms(ase_atoms: AseAtoms, pbc: bool = True) -> PhonopyAtoms:
    """Convert ASE Atoms to Phonopy Atoms.

    Args:
        ase_atoms: ASE atoms object.
        pbc: Periodic boundary conditions flag.

    Returns:
        PhonopyAtoms: Converted Phonopy atoms object.
    """
    return PhonopyAtoms(
        symbols=ase_atoms.symbols,
        positions=ase_atoms.get_positions(),
        pbc=pbc,
        cell=ase_atoms.get_cell(),
    )


@dataclass(slots=True)
class PhononConfig:
    """Configuration for phonon calculations.

    Attributes:
        structure: Initial pymatgen Structure.
        supercell_dims: Supercell dimensions (nx, ny, nz).
        displacement: Displacement magnitude for finite difference (Å).
        num_snapshots: Number of displacement snapshots (None for default).
        write_fc: Whether to write FORCE_CONSTANTS file.
        output_poscars: Whether to write POSCAR files for each displacement.
        path: Output directory path.
    """

    structure: Structure
    supercell_dims: list[int] = (2, 2, 2)
    displacement: float = 0.01
    num_snapshots: int | None = None
    write_fc: bool = True
    output_poscars: bool = False
    path: str = "."

    def to_dict(self) -> dict[str, object]:
        """Convert configuration to a dictionary.

        Returns:
            dict: Dictionary representation with structure serialized.
        """
        result = asdict(self)
        if hasattr(self, "structure") and isinstance(self.structure, Structure):
            result["structure"] = self.structure.as_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PhononConfig":
        """Create configuration from a dictionary.

        Args:
            data: Dictionary containing configuration parameters.

        Returns:
            PhononConfig: Reconstructed configuration instance.
        """
        params = data.copy()
        if "structure" in params and isinstance(params["structure"], dict):
            params["structure"] = Structure.from_dict(params["structure"])
        return cls(**params)  # type: ignore[arg-type]


class PhononCalculator:
    """Phonon calculator using finite displacement method.

    This class manages the generation of supercells with displacements,
    force calculations via a machine learning model, and extraction of
    force constants and phonon properties.
    """

    def __init__(self, config: PhononConfig) -> None:
        """Initialize the phonon calculator.

        Args:
            config: Phonon configuration object.
        """
        self.config = config

        if config.path and not os.path.exists(config.path):
            os.makedirs(config.path, exist_ok=True)

        self.phonopy_structure = get_phonopy_structure(config.structure)
        self.supercell = self._create_supercell()
        self.phonon: Phonopy | None = None
        self.force_constants: np.ndarray | None = None

    def _create_supercell(self) -> Structure:
        """Create a supercell from the primitive structure and write it to file.

        Returns:
            Structure: The supercell structure.
        """
        new_structure = self.config.structure.copy()
        new_structure.make_supercell(self.config.supercell_dims)

        supercell_name = os.path.join(
            self.config.path,
            f"SPOSCAR_{self.config.supercell_dims[0]}{self.config.supercell_dims[1]}{self.config.supercell_dims[2]}",
        )
        new_structure.to(filename=supercell_name)
        return new_structure

    def calculate_force_constants(self, calculator: ModelBase) -> np.ndarray:
        """Calculate force constants using a machine learning calculator.

        Args:
            calculator: ML model instance that provides forces.

        Returns:
            np.ndarray: Force constants matrix (3N x 3N).

        Raises:
            RuntimeError: If displacement generation or force calculation fails.
        """
        phonon = Phonopy(
            self.phonopy_structure,
            supercell_matrix=[
                [self.config.supercell_dims[0], 0, 0],
                [0, self.config.supercell_dims[1], 0],
                [0, 0, self.config.supercell_dims[2]],
            ],
        )

        phonon.generate_displacements(
            distance=self.config.displacement,
            number_of_snapshots=self.config.num_snapshots,
        )

        supercells = phonon.supercells_with_displacements

        if supercells is None:
            raise RuntimeError("Failed to generate supercells with displacements")

        forces_list = []

        for i, scell in enumerate(supercells):
            ase_atoms = phonopy_to_ase_atoms(scell, pbc=True)

            if self.config.output_poscars:
                poscar_path = os.path.join(self.config.path, f"POSCAR-{i + 1:03d}")
                ase_atoms.write(poscar_path, format="vasp", direct="True")

            result = calculator.calculate(ase_atoms)
            forces = result.get("forces")

            if forces is not None:
                drift_force = forces.sum(axis=0)
                if np.any(np.abs(drift_force) > TOLERANCE):
                    forces -= drift_force / forces.shape[0]
                forces_list.append(forces)

        if not forces_list:
            raise RuntimeError("No forces calculated")

        phonon.forces = forces_list
        phonon.produce_force_constants()

        if self.config.write_fc:
            fc_path = os.path.join(self.config.path, "FORCE_CONSTANTS")
            write_FORCE_CONSTANTS(phonon.force_constants, filename=fc_path)

        self.phonon = phonon
        self.force_constants = phonon.force_constants

        return self.force_constants

    def get_phonon(self, calculator: ModelBase) -> Phonopy:
        """Get the Phonopy object, computing force constants if needed.

        Args:
            calculator: ML model instance for force calculations.

        Returns:
            Phonopy: Phonopy object with force constants.
        """
        if self.phonon is None:
            self.calculate_force_constants(calculator)
        return self.phonon  # type: ignore[return-value]

    def get_force_constants(self, calculator: ModelBase) -> np.ndarray:
        """Get the force constants matrix, computing if needed.

        Args:
            calculator: ML model instance for force calculations.

        Returns:
            np.ndarray: Force constants matrix.
        """
        if self.force_constants is None:
            self.calculate_force_constants(calculator)
        return self.force_constants  # type: ignore[return-value]

    def save(self, filename: str = "phonon_calculator.pkl") -> None:
        """Save the calculator object to a pickle file.

        Args:
            filename: Name of the pickle file (saved in config.path).
        """
        import pickle

        save_path = os.path.join(self.config.path, filename)
        with open(save_path, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
