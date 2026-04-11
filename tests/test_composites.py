"""Integration tests for Mem3DG composites."""

import pytest
from process_bigraph import Composite, allocate_core, gather_emitter_results
from process_bigraph.emitter import RAMEmitter
from pbg_mem3dg.processes import Mem3DGProcess
from pbg_mem3dg.composites import make_membrane_document


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('Mem3DGProcess', Mem3DGProcess)
    c.register_link('ram-emitter', RAMEmitter)
    return c


def test_composite_assembly(core):
    doc = make_membrane_document(
        subdivision=2,
        characteristic_timestep=1.0,
        interval=5.0)
    sim = Composite({'state': doc}, core=core)
    assert sim is not None


def test_composite_short_run(core):
    doc = make_membrane_document(
        subdivision=2,
        characteristic_timestep=1.0,
        interval=5.0)
    sim = Composite({'state': doc}, core=core)
    sim.run(10.0)

    # Check stores populated
    stores = sim.state['stores']
    assert stores['total_energy'] >= 0
    assert stores['surface_area'] > 0
    assert stores['volume'] > 0
    assert len(stores['vertex_positions']) > 0


def test_emitter_collects_timeseries(core):
    doc = make_membrane_document(
        subdivision=2,
        characteristic_timestep=1.0,
        interval=5.0)
    sim = Composite({'state': doc}, core=core)
    sim.run(20.0)

    raw_results = gather_emitter_results(sim)
    # Results are keyed by emitter path tuple
    emitter_data = raw_results[('emitter',)]
    assert len(emitter_data) >= 2
    # Each entry is a dict with the emitted fields
    assert 'total_energy' in emitter_data[0]
    assert 'time' in emitter_data[0]
    # Energy values should be finite
    for entry in emitter_data[1:]:
        assert entry['total_energy'] is not None
        assert entry['total_energy'] >= 0


def test_document_factory_params(core):
    doc = make_membrane_document(
        mesh_type='icosphere',
        radius=0.5,
        subdivision=2,
        Kbc=1e-4,
        tension_modulus=0.2,
        osmotic_strength=0.05,
        preferred_volume_fraction=0.8,
        interval=10.0)
    sim = Composite({'state': doc}, core=core)
    sim.run(10.0)
    stores = sim.state['stores']
    assert stores['surface_area'] > 0
