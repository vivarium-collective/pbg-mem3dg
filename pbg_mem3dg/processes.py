"""Mem3DG Process wrapper for process-bigraph.

Wraps the Mem3DG membrane mechanics simulator as a time-driven Process
using the bridge pattern. The internal Mem3DG System and Euler integrator
are lazily initialized on first update() call.
"""

import os
import tempfile
import numpy as np
from process_bigraph import Process


def _write_ply(vertex, face, path):
    """Write vertex/face arrays to an ASCII PLY file."""
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {vertex.shape[0]}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write(f'element face {face.shape[0]}\n')
        f.write('property list uchar int vertex_indices\n')
        f.write('end_header\n')
        for v in vertex:
            f.write(f'{v[0]} {v[1]} {v[2]}\n')
        for fc in face:
            f.write(f'3 {fc[0]} {fc[1]} {fc[2]}\n')


class Mem3DGProcess(Process):
    """Bridge Process wrapping Mem3DG membrane mechanics simulation.

    Simulates lipid membrane dynamics on a triangulated surface mesh
    using discrete differential geometry. On each update(), advances
    the Mem3DG Euler integrator by the requested time interval using
    manual stepping (march()), then returns updated geometry and energy
    as deltas or overwrites.

    Config:
        mesh_type: Initial mesh shape ('icosphere', 'hexagon', 'cylinder')
        radius: Mesh radius
        subdivision: Mesh subdivision level
        Kbc: Bending rigidity coefficient
        H0c: Spontaneous curvature coefficient
        tension_modulus: Surface tension modulus (harmonic model)
        preferred_area: Preferred surface area (0 = auto from initial mesh)
        osmotic_strength: Osmotic pressure strength (harmonic model)
        preferred_volume_fraction: Fraction of initial volume as preferred
        Kse, Ksl, Kst: Spring regularization constants
        characteristic_timestep: Euler integrator base timestep
        tolerance: Convergence tolerance
        shape_variation: Enable shape evolution
        protein_variation: Enable protein density evolution
    """

    config_schema = {
        # Mesh initialization
        'mesh_type': {'_type': 'string', '_default': 'icosphere'},
        'radius': {'_type': 'float', '_default': 1.0},
        'subdivision': {'_type': 'integer', '_default': 3},
        'axial_subdivision': {'_type': 'integer', '_default': 0},
        # Bending
        'Kbc': {'_type': 'float', '_default': 8.22e-5},
        'H0c': {'_type': 'float', '_default': 0.0},
        # Tension (harmonic preferred-area model)
        'tension_modulus': {'_type': 'float', '_default': 0.1},
        'preferred_area': {'_type': 'float', '_default': 0.0},
        'preferred_area_scale': {'_type': 'float', '_default': 1.0},
        # Osmotic pressure
        'osmotic_model': {'_type': 'string', '_default': 'preferred_volume'},
        'osmotic_strength': {'_type': 'float', '_default': 0.02},
        'osmotic_pressure': {'_type': 'float', '_default': 0.0},
        'preferred_volume_fraction': {'_type': 'float', '_default': 0.7},
        # Spring regularization
        'Kse': {'_type': 'float', '_default': 0.0},
        'Ksl': {'_type': 'float', '_default': 0.0},
        'Kst': {'_type': 'float', '_default': 0.0},
        # Boundary conditions
        'boundary_condition': {'_type': 'string', '_default': 'none'},
        # Integrator
        'characteristic_timestep': {'_type': 'float', '_default': 2.0},
        'tolerance': {'_type': 'float', '_default': 1e-11},
        # Variation control
        'shape_variation': {'_type': 'boolean', '_default': True},
        'protein_variation': {'_type': 'boolean', '_default': False},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._system = None
        self._geometry = None  # keep reference (tracks System state in-place)
        self._integrator = None
        self._output_dir = None
        self._ply_path = None
        self._converged = False

    def inputs(self):
        return {}

    def outputs(self):
        return {
            'vertex_positions': 'overwrite[list]',
            'mean_curvatures': 'overwrite[list]',
            'total_energy': 'overwrite[float]',
            'bending_energy': 'overwrite[float]',
            'surface_energy': 'overwrite[float]',
            'pressure_energy': 'overwrite[float]',
            'surface_area': 'overwrite[float]',
            'volume': 'overwrite[float]',
            'converged': 'overwrite[boolean]',
        }

    def _read_state(self):
        """Read current geometry and energy from the Mem3DG System."""
        energy = self._system.getEnergy()
        vertex = self._geometry.getVertexMatrix()
        H = self._geometry.getVertexMeanCurvatures()
        return {
            'vertex_positions': vertex.tolist(),
            'mean_curvatures': H.tolist(),
            'total_energy': float(energy.totalEnergy),
            'bending_energy': float(energy.spontaneousCurvatureEnergy),
            'surface_energy': float(energy.surfaceEnergy),
            'pressure_energy': float(energy.pressureEnergy),
            'surface_area': float(self._geometry.getSurfaceArea()),
            'volume': float(self._geometry.getVolume()),
            'converged': self._converged,
        }

    def initial_state(self):
        self._build_system()
        return self._read_state()

    def get_faces(self):
        """Return the face connectivity array as a list of [i, j, k] triples.

        Must be called after initial_state() or update() has triggered
        system initialization.
        """
        self._build_system()
        return self._geometry.getFaceMatrix().tolist()

    def _build_system(self):
        """Lazily initialize the Mem3DG System and Euler integrator."""
        if self._system is not None:
            return

        import pymem3dg as dg
        import pymem3dg.boilerplate as dgb
        from functools import partial

        cfg = self.config

        # Generate mesh
        if cfg['mesh_type'] == 'icosphere':
            face, vertex = dg.getIcosphere(
                radius=cfg['radius'],
                subdivision=cfg['subdivision'])
        elif cfg['mesh_type'] == 'hexagon':
            face, vertex = dg.getHexagon(
                radius=cfg['radius'],
                subdivision=cfg['subdivision'])
        elif cfg['mesh_type'] == 'cylinder':
            axial = cfg['axial_subdivision'] or cfg['subdivision'] * 2
            face, vertex = dg.getCylinder(
                radius=cfg['radius'],
                radialSubdivision=cfg['subdivision'],
                axialSubdivision=axial)
        else:
            raise ValueError(
                f"Unknown mesh_type: {cfg['mesh_type']}. "
                f"Use 'icosphere', 'hexagon', or 'cylinder'.")

        # Write PLY (workaround for Geometry array constructor segfault)
        self._ply_path = tempfile.mktemp(suffix='.ply')
        _write_ply(vertex, face, self._ply_path)
        geo = dg.Geometry(self._ply_path)
        self._geometry = geo  # keep reference; tracks System state in-place

        # Parameters
        p = dg.Parameters()
        p.bending.Kbc = cfg['Kbc']
        p.bending.H0c = cfg['H0c']
        p.spring.Kse = cfg['Kse']
        p.spring.Ksl = cfg['Ksl']
        p.spring.Kst = cfg['Kst']
        p.variation.isShapeVariation = cfg['shape_variation']
        p.variation.isProteinVariation = cfg['protein_variation']

        # Tension model
        sa = geo.getSurfaceArea()
        if cfg['preferred_area'] > 0:
            preferred_area = cfg['preferred_area']
        else:
            preferred_area = sa * cfg['preferred_area_scale']
        p.tension.form = partial(
            dgb.preferredAreaSurfaceTensionModel,
            modulus=cfg['tension_modulus'],
            preferredArea=preferred_area)

        # Osmotic model
        vol = geo.getVolume()
        if cfg['osmotic_model'] == 'constant':
            p.osmotic.form = partial(
                dgb.constantOsmoticPressureModel,
                pressure=cfg['osmotic_pressure'])
        else:
            p.osmotic.form = partial(
                dgb.preferredVolumeOsmoticPressureModel,
                preferredVolume=cfg['preferred_volume_fraction'] * vol,
                reservoirVolume=0,
                strength=cfg['osmotic_strength'])

        # Boundary conditions
        if cfg['boundary_condition'] != 'none':
            p.boundary.shapeBoundaryCondition = cfg['boundary_condition']

        # Build System
        self._system = dg.System(geometry=geo, parameters=p)
        self._system.initialize()

        # Build Euler integrator for manual stepping
        self._output_dir = tempfile.mkdtemp(prefix='mem3dg_')
        self._integrator = dg.Euler(
            system=self._system,
            characteristicTimeStep=cfg['characteristic_timestep'],
            tolerance=cfg['tolerance'],
            outputDirectory=self._output_dir)

    def update(self, state, interval):
        self._build_system()

        if self._converged:
            return {}

        # Advance by interval using manual stepping
        target_time = self._system.time + interval
        dt = self.config['characteristic_timestep']
        while self._system.time < target_time - 1e-12:
            self._integrator.status()
            if self._integrator.EXIT:
                self._converged = True
                break
            self._integrator.march()

        return self._read_state()

    def __del__(self):
        """Clean up temporary files."""
        try:
            import shutil
            if self._ply_path and os.path.exists(self._ply_path):
                os.unlink(self._ply_path)
            if self._output_dir and os.path.exists(self._output_dir):
                shutil.rmtree(self._output_dir, ignore_errors=True)
        except (ImportError, TypeError):
            pass  # interpreter shutting down
