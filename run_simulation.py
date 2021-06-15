from neural_pointers.input import Inputs
import hydra
import numpy as np
from scipy.integrate import odeint
from omegaconf import OmegaConf

from neural_pointers.model import SynapticPointerModel, DOMAIN_MODE
from neural_pointers.input import Inputs
from neural_pointers.visualization import visualize_stage, visualize_trial

# Custom resolver to easily and safely parse configs to `DOMAIN_MODE` enums.
OmegaConf.register_resolver("DOMAIN_MODE", lambda x: DOMAIN_MODE[x])


def solve_ode(model, h_0, t, ode_args, stage, plot=True):
    """Inner method to solve an ODE `model` over a range of points in time `t`.

    Args:
        model: Callable method that computes a temporal derivative of a state `h`.
        h_0: Initial state values of the system.
        t: List of timesteps to evaluate the system for.
        ode_args: Additional arguments to be passed to `model`. Consists of excitatory input to the domain neurons, as
        well as the input from the pointer neurons to the domain neurons.
        stage: String containing the current stage that is simulated. Used to make the visualizations more informative.
        plot: Boolean argument to indicate whether a visualization should be computed.

    Returns:
        An array of states for the queried points in time `t`.
    """
    h_m = odeint(model, h_0, t, args=ode_args)
    if plot:
        visualize_stage(model, h_m, t, ode_args, stage)
    return h_m


def run_simulation(config):
    """Main method to run a simulation.

    Args:
        config: Configuration automatically parsed by `hydra`.
    """
    # make simulation deterministic
    rng = np.random.default_rng(config.seed)

    # init model
    m = SynapticPointerModel(config.n_neurons, config._model_.model_params)

    # init inputs
    inputs = Inputs(config.n_neurons, rng)

    # init simulation
    t_old = 0
    h_m_old = m._h_0
    results = []

    # simulate stages
    for stage in config._simulation_.simulation_params.stages:
        t = np.arange(
            t_old, t_old + stage.duration, config._simulation_.simulation_params.dt
        )
        m.set_mode(DOMAIN_MODE(stage.mode))
        # define input
        ode_args_binding = inputs.get_input(stage.name)
        h_m = solve_ode(m, h_m_old, t, ode_args_binding, stage.name, plot=config.plot)

        # Update offset values
        t_old = t[-1]
        h_m_old = h_m[-1]

        # Concatenate results
        results.append(h_m)

    # Visualize results
    visualize_trial(m, results)


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    """Main entry point of the program.

    Args:
        cfg: Configuration parsed by `hydra` from a hierarchical configuration stored in `../conf/` with the top-level
        configuration file `../conf/config.yaml`.
    """
    # comment the next line in to view the parsed configuration
    # print(OmegaConf.to_yaml(cfg, resolve=True))

    run_simulation(cfg)


if __name__ == "__main__":
    main()
