"""Pre-built composite document factories for Mem3DG simulations."""


def make_membrane_document(
    mesh_type='icosphere',
    radius=1.0,
    subdivision=3,
    Kbc=8.22e-5,
    tension_modulus=0.1,
    osmotic_strength=0.02,
    preferred_volume_fraction=0.7,
    characteristic_timestep=2.0,
    tolerance=1e-11,
    interval=10.0,
):
    """Create a composite document for a membrane mechanics simulation.

    Returns a document dict ready for use with Composite().

    Args:
        mesh_type: Initial mesh shape ('icosphere', 'hexagon', 'cylinder')
        radius: Mesh radius
        subdivision: Mesh subdivision level
        Kbc: Bending rigidity coefficient
        tension_modulus: Surface tension modulus
        osmotic_strength: Osmotic pressure strength
        preferred_volume_fraction: Fraction of initial volume as target
        characteristic_timestep: Euler integrator base timestep
        tolerance: Convergence tolerance
        interval: Time interval between process updates

    Returns:
        dict: Composite document with membrane process, stores, and emitter
    """
    return {
        'membrane': {
            '_type': 'process',
            'address': 'local:Mem3DGProcess',
            'config': {
                'mesh_type': mesh_type,
                'radius': radius,
                'subdivision': subdivision,
                'Kbc': Kbc,
                'tension_modulus': tension_modulus,
                'osmotic_strength': osmotic_strength,
                'preferred_volume_fraction': preferred_volume_fraction,
                'characteristic_timestep': characteristic_timestep,
                'tolerance': tolerance,
            },
            'interval': interval,
            'inputs': {},
            'outputs': {
                'vertex_positions': ['stores', 'vertex_positions'],
                'mean_curvatures': ['stores', 'mean_curvatures'],
                'total_energy': ['stores', 'total_energy'],
                'bending_energy': ['stores', 'bending_energy'],
                'surface_energy': ['stores', 'surface_energy'],
                'pressure_energy': ['stores', 'pressure_energy'],
                'surface_area': ['stores', 'surface_area'],
                'volume': ['stores', 'volume'],
                'converged': ['stores', 'converged'],
            },
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {
                'emit': {
                    'total_energy': 'float',
                    'bending_energy': 'float',
                    'surface_energy': 'float',
                    'pressure_energy': 'float',
                    'surface_area': 'float',
                    'volume': 'float',
                    'time': 'float',
                },
            },
            'inputs': {
                'total_energy': ['stores', 'total_energy'],
                'bending_energy': ['stores', 'bending_energy'],
                'surface_energy': ['stores', 'surface_energy'],
                'pressure_energy': ['stores', 'pressure_energy'],
                'surface_area': ['stores', 'surface_area'],
                'volume': ['stores', 'volume'],
                'time': ['global_time'],
            },
        },
    }
