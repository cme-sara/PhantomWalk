import numpy as np  
import freud
import gsd, gsd.hoomd 
import hoomd 

def initialize_snapshot_rand_walk(num_pol, num_mon, density, bond_length=1.0, seed=1234):
    ''' 
    Create a HOOMD snapshot of a cubic box with the number density given by input parameters. Configure particles using a random walk. 

    '''
    rng = np.random.default_rng(seed)
    N = num_pol * num_mon
    L = np.cbrt(N / density)
    positions = np.empty((N, 3))
    starts = rng.uniform(0, L, size=(num_pol, 3))
    thetas = rng.uniform(0,2*np.pi,size=(num_pol,num_mon-1))
    phis = np.arccos(rng.uniform(-1,1,size=(num_pol,num_mon-1)))
    x = np.sin(phis)*np.cos(thetas)
    y = np.sin(phis)*np.sin(thetas)
    z = np.cos(phis)
    deltas = np.stack([x,y,z],axis=2) * bond_length
    displacements = np.cumsum(deltas, axis=1)
    positions_view = positions.reshape(num_pol, num_mon, 3)
    positions_view[:, 0, :] = starts
    positions_view[:, 1:, :] = starts[:, None, :] + displacements
    positions %= L
    positions -= L/2 #TODO: use box in flowerMD
    indices = np.arange(N).reshape(num_pol, num_mon)
    bonds = np.column_stack([
        indices[:, :-1].ravel(),
        indices[:, 1:].ravel()
    ])
    frame = gsd.hoomd.Frame()
    frame.particles.types = ['A']
    frame.particles.N = N
    frame.particles.position = positions
    frame.bonds.N = len(bonds)
    frame.bonds.group = bonds
    frame.bonds.types = ['b']
    frame.configuration.box = [L,L,L,0,0,0]
    return frame

def add_hoomd_writers(
    sim,
    gsd_file_name="trajectory.gsd",
    gsd_write_freq=10,
    log_file_name="log.txt",
    log_write_freq=10
):
    """Add GSD trajectory and log writers to a HOOMD simulation.

    This function creates:
    - a GSD trajectory writer for particle configurations
    - a table logger for thermodynamic and force quantities
    - thermodynamic compute operations for system properties

    Parameters
    ----------
    sim : hoomd.Simulation
        HOOMD simulation object to which writers and
        computes will be attached.
    gsd_file_name : str, default 'trajectory.gsd'
        the file that the gsd trajectory data will be saved to
    gsd_write_freq : int, default 10
        Period to write simulation data to the gsd file.
    log_file_name : str, default 'log.txt'
        the file that the .txt log file will be saved to
    log_write_freq : int, default 10
        Period to write simulation data to the log file.

    Returns
    -------
    None
        This function modifies the simulation object in place
        and does not return a value.

    """

    class FreudRDFCalc(hoomd.custom.Action):
        """Compute RDF periodically as the simulation progresses."""
    
        def __init__(self, sim, rdf):
            self._sim = sim
            self._rdf = rdf
    
        def act(self, timestep):
            snap = self._sim.state.get_snapshot()
            self._rdf.compute(system=snap, reset=True)

    rdf = freud.density.RDF(bins=100, r_max=2.0)
    rdf_calc = FreudRDFCalc(sim, rdf)

    
    gsd_logger = hoomd.logging.Logger(
        categories=["scalar", "string", "sequence"]
    )
    logger = hoomd.logging.Logger(categories=["scalar", "string"])
    gsd_logger.add(sim, quantities=["timestep", "tps"])
    logger.add(sim, quantities=["timestep", "tps"])
    thermo_props = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo_props)
    log_quantities = [
            "kinetic_temperature",
            "potential_energy",
            "kinetic_energy",
            "volume",
            "pressure",
            "pressure_tensor",
        ]
    gsd_logger.add(thermo_props, quantities=log_quantities)
    logger.add(thermo_props, quantities=log_quantities)

    for f in sim.operations.integrator.forces:
        logger.add(f, quantities=["energy"])
        gsd_logger.add(f, quantities=["energy"])
    
    gsd_trigger = hoomd.trigger.Or([
        hoomd.trigger.Before(2),
        hoomd.trigger.Periodic(int(gsd_write_freq))])
    
    gsd_writer = hoomd.write.GSD(
        filename=gsd_file_name,
        trigger=gsd_trigger,
        mode="wb",
        dynamic=["momentum", "property"],
        filter=hoomd.filter.All(),
        logger=gsd_logger,
    )
    gsd_writer.maximum_write_buffer_size = 64 * 1024 * 1024
    log_trigger = hoomd.trigger.Or([
        hoomd.trigger.Before(2),
        hoomd.trigger.Periodic(int(log_write_freq))])

    table_file = hoomd.write.Table(
        output=open(log_file_name, mode="w", newline="\n"),
        trigger=log_trigger,
        logger=logger,
        max_header_len=None,
    )
    
    rdf_action = hoomd.write.CustomWriter(action=rdf_calc, trigger=log_trigger)
    sim.operations.writers.append(rdf_action)
    
    sim.operations.writers.append(gsd_writer)
    sim.operations.writers.append(table_file)
    return rdf, thermo_props

