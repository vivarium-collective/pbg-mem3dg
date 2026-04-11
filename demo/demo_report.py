"""Demo: Mem3DG membrane relaxation via process-bigraph.

Simulates a spherical membrane relaxing under bending rigidity,
surface tension, and osmotic pressure. The preferred volume is set
to 70% of the initial sphere volume, driving a shape change toward
a deflated vesicle. Produces time series plots of energy components,
surface area, and volume.
"""

import os
import sys
import matplotlib.pyplot as plt
from process_bigraph import Composite, allocate_core, gather_emitter_results
from process_bigraph.emitter import RAMEmitter
from pbg_mem3dg.processes import Mem3DGProcess
from pbg_mem3dg.composites import make_membrane_document


def run_demo():
    # Setup
    core = allocate_core()
    core.register_link('Mem3DGProcess', Mem3DGProcess)
    core.register_link('ram-emitter', RAMEmitter)

    doc = make_membrane_document(
        mesh_type='icosphere',
        radius=1.0,
        subdivision=3,
        Kbc=8.22e-5,
        tension_modulus=0.1,
        osmotic_strength=0.02,
        preferred_volume_fraction=0.7,
        characteristic_timestep=2.0,
        tolerance=1e-11,
        interval=100.0,
    )

    print('Building composite...')
    sim = Composite({'state': doc}, core=core)

    print('Running membrane relaxation (1000 time units)...')
    sim.run(1000.0)

    # Collect results
    raw_results = gather_emitter_results(sim)
    emitter_data = raw_results[('emitter',)]

    time = [d['time'] for d in emitter_data]
    total_energy = [d['total_energy'] for d in emitter_data]
    bending_energy = [d['bending_energy'] for d in emitter_data]
    surface_energy = [d['surface_energy'] for d in emitter_data]
    pressure_energy = [d['pressure_energy'] for d in emitter_data]
    surface_area = [d['surface_area'] for d in emitter_data]
    volume = [d['volume'] for d in emitter_data]

    print(f'Collected {len(time)} time points')
    print(f'Energy: {total_energy[0]:.6f} -> {total_energy[-1]:.6f}')
    print(f'Surface area: {surface_area[0]:.4f} -> {surface_area[-1]:.4f}')
    print(f'Volume: {volume[0]:.4f} -> {volume[-1]:.4f}')

    # Plot
    demo_dir = os.path.dirname(os.path.abspath(__file__))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Mem3DG Membrane Relaxation via process-bigraph', fontsize=14)

    # Total energy
    ax = axes[0, 0]
    ax.plot(time, total_energy, 'k-', linewidth=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Total Energy')
    ax.set_title('Total Energy')
    ax.grid(True, alpha=0.3)

    # Energy components
    ax = axes[0, 1]
    ax.plot(time, bending_energy, label='Bending', linewidth=1.2)
    ax.plot(time, surface_energy, label='Surface', linewidth=1.2)
    ax.plot(time, pressure_energy, label='Osmotic', linewidth=1.2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Components')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Surface area
    ax = axes[1, 0]
    ax.plot(time, surface_area, 'b-', linewidth=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Surface Area')
    ax.set_title('Surface Area')
    ax.grid(True, alpha=0.3)

    # Volume
    ax = axes[1, 1]
    ax.plot(time, volume, 'r-', linewidth=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Volume')
    ax.set_title('Volume')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(demo_dir, 'demo_output.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved {out_path}')
    plt.close()


if __name__ == '__main__':
    run_demo()
