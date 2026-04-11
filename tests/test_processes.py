"""Unit tests for Mem3DGProcess."""

import pytest
from process_bigraph import allocate_core
from pbg_mem3dg.processes import Mem3DGProcess


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('Mem3DGProcess', Mem3DGProcess)
    return c


def test_instantiation(core):
    proc = Mem3DGProcess(
        config={'subdivision': 2, 'characteristic_timestep': 1.0},
        core=core)
    assert proc.config['subdivision'] == 2
    assert proc.config['mesh_type'] == 'icosphere'


def test_initial_state(core):
    proc = Mem3DGProcess(
        config={'subdivision': 2, 'characteristic_timestep': 1.0},
        core=core)
    state = proc.initial_state()
    assert 'vertex_positions' in state
    assert 'total_energy' in state
    assert 'surface_area' in state
    assert 'volume' in state
    assert len(state['vertex_positions']) > 0
    assert state['total_energy'] >= 0
    assert state['surface_area'] > 0
    assert state['volume'] > 0
    assert state['converged'] is False


def test_single_update(core):
    proc = Mem3DGProcess(
        config={'subdivision': 2, 'characteristic_timestep': 1.0},
        core=core)
    state0 = proc.initial_state()
    result = proc.update({}, interval=5.0)
    assert 'vertex_positions' in result
    assert 'total_energy' in result
    assert isinstance(result['total_energy'], float)
    assert isinstance(result['surface_area'], float)
    assert isinstance(result['volume'], float)


def test_energy_decreases_or_stable(core):
    proc = Mem3DGProcess(
        config={
            'subdivision': 2,
            'characteristic_timestep': 1.0,
            'Kbc': 8.22e-5,
            'tension_modulus': 0.1,
            'osmotic_strength': 0.02,
            'preferred_volume_fraction': 0.7,
        },
        core=core)
    state0 = proc.initial_state()
    e0 = state0['total_energy']

    result = proc.update({}, interval=20.0)
    e1 = result['total_energy']
    # Energy should decrease or stay stable during relaxation
    assert e1 <= e0 + 1e-6, f'Energy increased: {e0} -> {e1}'


def test_outputs_schema(core):
    proc = Mem3DGProcess(config={'subdivision': 2}, core=core)
    outputs = proc.outputs()
    expected_ports = [
        'vertex_positions', 'mean_curvatures', 'total_energy',
        'bending_energy', 'surface_energy', 'pressure_energy',
        'surface_area', 'volume', 'converged',
    ]
    for port in expected_ports:
        assert port in outputs, f'Missing output port: {port}'


def test_config_defaults(core):
    proc = Mem3DGProcess(config={}, core=core)
    assert proc.config['mesh_type'] == 'icosphere'
    assert proc.config['radius'] == 1.0
    assert proc.config['subdivision'] == 3
    assert proc.config['Kbc'] == 8.22e-5
    assert proc.config['shape_variation'] is True
    assert proc.config['protein_variation'] is False
