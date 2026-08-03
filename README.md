# PhononsML

**Phonon calculations with machine learning potentials**

PhononsML is a Python package for computing phonon properties using machine-learned interatomic potentials (MLIPs). It provides a streamlined workflow for force constant calculations, phonon band structures, and thermodynamic property analysis.

## Features

- **Multiple ML potentials**: Support for MatterSim and MACE models
- **Phonon calculations**: Finite displacement method with Phonopy integration
- **Visualization**: Band structure plots with density of states
- **Thermodynamic analysis**: Free energy, entropy, heat capacity, and stability assessment
- **Batch processing**: Process multiple structures sequentially or in parallel
- **Caching**: Automatic caching of calculation results
- **Flexible input**: Support for ASE, Phonopy, and pymatgen structures

## Installation

### Prerequisites
- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Basic installation

```bash
# Clone the repository
git clone https://github.com/yourusername/phononsml.git
cd phononsml

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .

## Installation with ML potentials

Choose one of the following depending on your needs:

```bash
# With MACE support
uv pip install -e ".[mace]"

# With MatterSim support
uv pip install -e ".[mattersim]"

# With both (may cause version conflicts)
uv pip install -e ".[mace,mattersim]"

# Development installation
uv pip install -e ".[dev]"


Note: MACE and MatterSim have conflicting PyTorch version requirements. It's recommended to install only one ML potential package at a time.


## Quick Start

### Basic usage

```python
import json
from pymatgen.core import Structure
from phonons_ml.pipeline.implementations.single import PipelineConfigSingle, PipelineSingle

# Load your structure
structure_dict = json.loads('{"lattice": ..., "sites": ...}')
structure = Structure.from_dict(structure_dict)

# Configure the pipeline
config = {
    "material_id": "example_material",
    "calculator_config": {
        "model_name": "mace",  # or "mattersim"
        "model_config": {
            "device": "cpu",
            "default_dtype": "float64",
            "verbose": True,
        },
    },
    "phonon_config": {
        "structure": structure_dict,
        "supercell_dims": [2, 2, 2],
        "displacement": 0.01,
        "path": "data/results/example/",
    },
    "visualizer_config": {
        "units": "THz",
        "line_density": 50,
        "output_dir": "data/results/example/",
    },
    "properties_config": {
        "material_id": "example_material",
        "t_min": 0,
        "t_max": 1000,
        "t_step": 10,
        "mesh_density": 40,
    },
}

# Create and run the pipeline
pipeline_config = PipelineConfigSingle.from_dict(config)
pipeline = PipelineSingle(config=pipeline_config)
results = pipeline.run(structure=structure)

# Access results
print(f"Stability status: {results.get('stability_status')}")
print(f"Has imaginary modes: {results.get('has_imaginary_modes')}")
