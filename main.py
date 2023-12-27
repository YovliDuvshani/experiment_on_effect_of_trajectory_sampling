from agent import Agent, SamplingStrategy
from env import Env
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = Env()
    complete_swoop_agent = Agent(env, sampling_strategy=SamplingStrategy.COMPLETE_SWOOP)
    trajectory_agent = Agent(env, sampling_strategy=SamplingStrategy.TRAJECTORY)
    values_of_v0_complete_swoop = complete_swoop_agent.update_model()
    values_of_v0_trajectory = trajectory_agent.update_model()

    plt.plot(values_of_v0_complete_swoop, label=SamplingStrategy.COMPLETE_SWOOP)
    plt.plot(values_of_v0_trajectory, label=SamplingStrategy.TRAJECTORY)
    plt.legend()
    plt.show()

    print()
