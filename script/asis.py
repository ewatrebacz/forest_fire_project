# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: self-adaptive_sis_model-HkBZS_AX-py3.12
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SAIS model

# %%
import networkx as nx
import numpy as np
import networkx as nx

import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import seaborn as sns
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from enum import auto, Enum
import random 

from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


# %%
def plot_state(model, step_no):
    fig, ax = plt.subplots(figsize=(10, 8))
    for _ in range(step_no):
        model._update_state()
    model._plot_current_state(ax)

def plot_model_simulation(model, steps):
    fig, ax = plt.subplots(figsize=(10, 8))

    def update(frame):
        model._update_state()
        model._plot_current_state(ax)

    return FuncAnimation(fig, update, frames=steps, repeat=False)


# %%
# x = nx.erdos_renyi_graph(10, 0.4)
nx.draw(x, with_labels=True)
# [i for i in x.neighbors(u) for u in [3, 4, 8]]
[i for u in [3, 4, 8] for i in x.neighbors(u) if i >=4]

# %%
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep

# Network topology
g = nx.erdos_renyi_graph(1000, 0.15)

# Model selection
model = ep.SISModel(g)

# Model Configuration
cfg = mc.Configuration()
cfg.add_model_parameter('beta', 0.002)
cfg.add_model_parameter('lambda', 0.05)
cfg.add_model_parameter("fraction_infected", 0.05)
model.set_initial_status(cfg)

# Simulation execution
iterations = model.iteration_bunch(200)
# nx.average_degree_connectivity(g)

# %%
import matplotlib.pyplot as plt

infected = [it['node_count'][1] for it in iterations]  # 1 = infected state
suspceptible = [it['node_count'][0] for it in iterations]  # 0 = infected state
plt.plot(infected, label='Infected')
plt.plot(suspceptible, label='Suspceptible')
plt.xlabel('Iterations')
plt.ylabel('Number of Infected Nodes')
plt.title('SIS Model: Infected Over Time')
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# ## SIS model

# %%
class StateSIS(Enum):
    suspectible = 0
    infected = 1

class ParamsSIS(BaseModel):
    beta: float = Field(..., gt=0, lt=1)
    gamma: float = Field(..., gt=0, lt=1)



# %%
class SIS:
    state_color_mapping = {
        StateSIS.suspectible: "blue",
        StateSIS.infected: "red",
    }
    
    def __init__(
        self,
        graph: nx.Graph,
        params: ParamsSIS,
        pos_seed: int = 3
    ) -> None:
        self.graph = graph
        self.pos = nx.spring_layout(self.graph, seed=pos_seed)
        self.params = params
    
    def init_state(self, I0: int, seed: int):
        random.seed(seed)
        np.random.seed(seed=seed)
        nx.set_node_attributes(self.graph, StateSIS.suspectible, "state")
        nx.set_node_attributes(
            self.graph,
            {
                node: {"state": StateSIS.infected}
                for node in random.sample(list(self.graph.nodes), k=I0)
            }
        )
    
    def _update_state(self):
        """Asynchronous updates considered."""
        node_states = nx.get_node_attributes(self.graph, "state")
        infected_nodes = [n for n, state in node_states.items() if state == StateSIS.infected]
        next_states = {}

        trans_IS = np.random.random(len(infected_nodes)) <= self.params.gamma
        recovered = {node: StateSIS.suspectible for node, is_change in zip(infected_nodes, trans_IS) if is_change}
        dangered_suscteptible = [v for u in infected_nodes for v in self.graph.neighbors(u) if node_states[v] == StateSIS.suspectible]
        trans_SI = np.random.random(len(dangered_suscteptible)) <= self.params.beta
        infected = {node: StateSIS.infected for node, is_change in zip(dangered_suscteptible, trans_SI) if is_change}

        next_states.update(recovered)
        next_states.update(infected)
        nx.set_node_attributes(self.graph, {node: {"state": state} for node, state in next_states.items()})


    def run_simulation(self, max_steps: int):
        statistics = {
            "infected_count": [],
            "suspectible_count": [],
        }
        is_finished = False
        step_no = 0
        while not is_finished and step_no < max_steps:
            self._update_state()
            states = nx.get_node_attributes(self.graph, "state").values()
            statistics["infected_count"].append(
                sum(1 for s in states if s == StateSIS.infected)
            )
            statistics["suspectible_count"].append(
                sum(1 for s in states if s == StateSIS.suspectible)
            )
            step_no += 1
            is_finished = SIS.is_finished(self.graph)
        return statistics

    @staticmethod
    def is_finished(graph: nx.Graph) -> bool:
        states = nx.get_node_attributes(graph, "state").values()
        total_infected = (
                sum(1 for s in states if s == StateSIS.infected)
        )
        return len(graph) == total_infected or total_infected == 0



    def _get_nodes_state_mapping(self) -> list[int]:
        return [self.state_color_mapping[s] for s in nx.get_node_attributes(self.graph,'state').values()]        
    
    def _plot_current_state(self, ax):
        ax.clear()
        nx.draw_networkx(
            self.graph, 
            pos=self.pos, 
            with_labels=False, 
            node_color=self._get_nodes_state_mapping(),
            node_size=7,
            edge_color='gray',
            width=0.3,
            alpha=0.8,
            ax=ax
        )
        ax.set_title(f"SIR model on the graph")
        ax.set_axis_off()


VIDEO_WRITER = animation.FFMpegWriter(fps=30, bitrate=1800, extra_args=['-vcodec', 'libx264', '-preset', 'ultrafast'])
VIDEO_WRITER_SLOW = animation.FFMpegWriter(fps=2, bitrate=1800, extra_args=['-vcodec', 'libx264', '-preset', 'slow'])
def save_animation(ani: FuncAnimation, filename: str, writer: animation.FFMpegWriter = VIDEO_WRITER_SLOW):
    ani.save(filename, writer=writer)



# %% [markdown]
# ### Test

# %%
x = nx.erdos_renyi_graph(20, 0.4)
nx.draw(x, with_labels=True)

# %%
fig, axs = plt.subplots(1, 2, figsize = (10,5))

g = nx.erdos_renyi_graph(1000, 0.12)
g1 = g.copy()
params = ParamsSIS(beta=0.001, gamma=0.05)
I0 = int(1000*0.05)

#Benchmark
model = ep.SISModel(g)
cfg = mc.Configuration()
cfg.add_model_parameter('beta', params.beta)
cfg.add_model_parameter('lambda', params.gamma)
cfg.add_model_parameter("fraction_infected", 0.05)
model.set_initial_status(cfg)
iterations = model.iteration_bunch(200)
infected = [it['node_count'][1] for it in iterations]  # 1 = infected state
suspceptible = [it['node_count'][0] for it in iterations]  # 0 = infected state
axs[0].plot(infected, label='Infected')
axs[0].plot(suspceptible, label='Suspceptible')
axs[0].set_xlabel('Iterations')
axs[0].set_ylabel('Number of Infected Nodes')
axs[0].set_title('SIS Model: Infected Over Time [NDlib]')
axs[0].grid()

plt.legend()
plt.grid(True)

#Defined
model = SIS(g1, params)
model.init_state(I0=I0, seed=1)
stats = model.run_simulation(200)
stats["infected_count"]
axs[1].plot(stats["infected_count"], label='Infected')
axs[1].plot(stats["suspectible_count"], label='Suspceptible')
axs[1].set_xlabel('Iterations')
axs[1].set_ylabel('Number of Infected Nodes')
axs[1].set_title('SIS Model: Infected Over Time [My implelemtation]')
axs[1].grid()
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# # SAIS

# %%
class StateSAIS(Enum):
    suspectible = auto()
    infected = auto()
    alert = auto()

class ParamsSAIS(BaseModel):
    beta: float = Field(..., gt=0, lt=1)
    beta_a: float = Field(..., gt=0, lt=1)
    delta: float = Field(..., gt=0, lt=1)
    kappa: float = Field(..., gt=0, lt=1)


# %%
class SAIS:
    state_color_mapping = {
        StateAlertSpreadingSAIS.suspectible: "blue",
        StateAlertSpreadingSAIS.infected: "red",
        StateAlertSpreadingSAIS.alert: "yellow",
    }
    
    def __init__(
        self,
        graph: nx.Graph,
        params: ParamsSAIS,
        pos_seed: int = 3
    ) -> None:
        self.graph = graph
        self.pos = nx.spring_layout(self.graph, seed=pos_seed)
        self.params = params
    
    def init_state(self, I0: int, seed: int):
        random.seed(seed)
        np.random.seed(seed=seed)
        nx.set_node_attributes(self.graph, StateAlertSpreadingSAIS.suspectible, "state")
        nx.set_node_attributes(
            self.graph,
            {
                node: {"state": StateAlertSpreadingSAIS.infected}
                for node in random.sample(list(self.graph.nodes), k=I0)
            }
        )
    
    def _update_state(self):
        """Asynchronous updates considered."""
        node_states = nx.get_node_attributes(self.graph, "state")
        infected_nodes = [n for n, state in node_states.items() if state == StateAlertSpreadingSAIS.infected]
        next_states = {}

        trans_IS = np.random.random(len(infected_nodes)) <= self.params.delta
        recovered = {node: StateAlertSpreadingSAIS.suspectible for node, is_change in zip(infected_nodes, trans_IS) if is_change}
        dangered_suscteptible = [v for u in infected_nodes for v in self.graph.neighbors(u) if node_states[v] == StateAlertSpreadingSAIS.suspectible]
        trans_SA = np.random.random(len(dangered_suscteptible)) <= self.params.kappa
        alerted = {node: StateAlertSpreadingSAIS.alert for node, is_change in zip(dangered_suscteptible, trans_SA) if is_change}
        trans_SI = np.random.random(len(dangered_suscteptible)) <= self.params.beta
        infected = {node: StateAlertSpreadingSAIS.infected for node, is_change in zip(dangered_suscteptible, trans_SI) if is_change}
        dangered_alert = [v for u in infected_nodes for v in self.graph.neighbors(u) if node_states[v] == StateAlertSpreadingSAIS.alert]
        trans_AI = np.random.random(len(dangered_alert)) <= self.params.beta_a
        infected_in_alert = {node: StateAlertSpreadingSAIS.infected for node, is_change in zip(dangered_alert, trans_AI) if is_change}

        next_states.update(recovered)
        #Infections should overwrite alerts
        next_states.update(alerted)
        next_states.update(infected)
        next_states.update(infected_in_alert)
        nx.set_node_attributes(self.graph, {node: {"state": state} for node, state in next_states.items()})


    def run_simulation(self, max_steps: int):
        statistics = {
            "infected_count": [],
            "suspectible_count": [],
            "alert_count": [],
        }
        is_finished = False
        step_no = 0
        while not is_finished and step_no < max_steps:
            self._update_state()
            states = nx.get_node_attributes(self.graph, "state").values()
            statistics["infected_count"].append(
                sum(1 for s in states if s == StateAlertSpreadingSAIS.infected)
            )
            statistics["suspectible_count"].append(
                sum(1 for s in states if s == StateAlertSpreadingSAIS.suspectible)
            )
            statistics["alert_count"].append(
                sum(1 for s in states if s == StateAlertSpreadingSAIS.alert)
            )
            step_no += 1
            is_finished = SAIS.is_finished(self.graph)
        return statistics

    @staticmethod
    def is_finished(graph: nx.Graph) -> bool:
        states = nx.get_node_attributes(graph, "state").values()
        total_infected = (
                sum(1 for s in states if s == StateAlertSpreadingSAIS.infected)
        )
        return len(graph) == total_infected or total_infected == 0



    def _get_nodes_state_mapping(self) -> list[int]:
        return [self.state_color_mapping[s] for s in nx.get_node_attributes(self.graph,'state').values()]        
    
    def _plot_current_state(self, ax):
        ax.clear()
        nx.draw_networkx(
            self.graph, 
            pos=self.pos, 
            with_labels=False, 
            node_color=self._get_nodes_state_mapping(),
            node_size=7,
            edge_color='gray',
            width=0.3,
            alpha=0.8,
            ax=ax
        )
        ax.set_title(f"SAIS model on the graph")
        ax.set_axis_off()



# %% [markdown]
# ### Tests

# %%
g = nx.erdos_renyi_graph(1000, 0.15)
params = ParamsSAIS(
    beta=0.001,
    beta_a=0.0005,
    delta=0.05,
    kappa=0.01,
)
I0=50

# %%
model = SAIS(g, params)
model.init_state(I0=I0, seed=1)
stats = model.run_simulation(200)

# %%
fig, axs = plt.subplots(figsize=(5,5))
#Defined
healthy =np.array(stats["suspectible_count"]) + np.array(stats["alert_count"]) 
stats["infected_count"]
axs.plot(stats["infected_count"], label='Infected')
axs.plot(healthy, label='Not Infected')
axs.set_xlabel('Iterations')
axs.set_ylabel('Number of Infected Nodes')
axs.set_title('SAIS Model: Infected Over Time [My implelemtation]')
axs.grid()
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# # Alert spreading SAIS

# %%
class StateAlertSpreadingSAIS(Enum):
    suspectible = auto()
    infected = auto()
    alert = auto()

class ParamsAlertSpreadingSAIS(BaseModel):
    beta: float = Field(..., gt=0, lt=1)
    beta_a: float = Field(..., gt=0, lt=1)
    delta: float = Field(..., gt=0, lt=1)
    kappa_s: float = Field(..., gt=0, lt=1)
    kappa_a: float = Field(..., gt=0, lt=1)


# %%
class AlertSpreadingSAIS:
    state_color_mapping = {
        StateSAIS.suspectible: "blue",
        StateSAIS.infected: "red",
        StateSAIS.alert: "yellow",
    }
    
    def __init__(
        self,
        graph: nx.Graph,
        params: ParamsAlertSpreadingSAIS,
        pos_seed: int = 3
    ) -> None:
        self.graph = graph
        self.pos = nx.spring_layout(self.graph, seed=pos_seed)
        self.params = params
    
    def init_state(self, I0: int, seed: int):
        random.seed(seed)
        np.random.seed(seed=seed)
        nx.set_node_attributes(self.graph, StateSAIS.suspectible, "state")
        nx.set_node_attributes(
            self.graph,
            {
                node: {"state": StateSAIS.infected}
                for node in random.sample(list(self.graph.nodes), k=I0)
            }
        )
    
    def _update_state(self):
        """Asynchronous updates considered."""
        node_states = nx.get_node_attributes(self.graph, "state")
        infected_nodes = [n for n, state in node_states.items() if state == StateSAIS.infected]
        alerted_nodes = [n for n, state in node_states.items() if state == StateSAIS.alert]
        next_states = {}

        trans_IS = np.random.random(len(infected_nodes)) <= self.params.delta
        recovered = {node: StateSAIS.suspectible for node, is_change in zip(infected_nodes, trans_IS) if is_change}
        dangered_suscteptible = [v for u in infected_nodes for v in self.graph.neighbors(u) if node_states[v] == StateSAIS.suspectible]
        trans_SA = np.random.random(len(dangered_suscteptible)) <= self.params.kappa_s
        alerted_by_I = {node: StateSAIS.alert for node, is_change in zip(dangered_suscteptible, trans_SA) if is_change}
        neighbours_of_alerted = [v for u in alerted_nodes for v in self.graph.neighbors(u) if node_states[v] == StateSAIS.alert]
        trans_SA2 = np.random.random(len(dangered_suscteptible)) <= self.params.kappa_a
        alerted_by_A = {node: StateSAIS.alert for node, is_change in zip(neighbours_of_alerted, trans_SA2) if is_change}
        trans_SI = np.random.random(len(dangered_suscteptible)) <= self.params.beta
        infected = {node: StateSAIS.infected for node, is_change in zip(dangered_suscteptible, trans_SI) if is_change}
        dangered_alert = [v for u in infected_nodes for v in self.graph.neighbors(u) if node_states[v] == StateSAIS.alert]
        trans_AI = np.random.random(len(dangered_alert)) <= self.params.beta_a
        infected_in_alert = {node: StateSAIS.infected for node, is_change in zip(dangered_alert, trans_AI) if is_change}

        next_states.update(recovered)
        #Infections should overwrite alerts
        next_states.update(alerted_by_I)
        next_states.update(alerted_by_A)
        next_states.update(infected)
        next_states.update(infected_in_alert)
        nx.set_node_attributes(self.graph, {node: {"state": state} for node, state in next_states.items()})


    def run_simulation(self, max_steps: int):
        statistics = {
            "infected_count": [],
            "suspectible_count": [],
            "alert_count": [],
        }
        is_finished = False
        step_no = 0
        while not is_finished and step_no < max_steps:
            self._update_state()
            states = nx.get_node_attributes(self.graph, "state").values()
            statistics["infected_count"].append(
                sum(1 for s in states if s == StateSAIS.infected)
            )
            statistics["suspectible_count"].append(
                sum(1 for s in states if s == StateSAIS.suspectible)
            )
            statistics["alert_count"].append(
                sum(1 for s in states if s == StateSAIS.alert)
            )
            step_no += 1
            is_finished = SAIS.is_finished(self.graph)
        return statistics

    @staticmethod
    def is_finished(graph: nx.Graph) -> bool:
        states = nx.get_node_attributes(graph, "state").values()
        total_infected = (
                sum(1 for s in states if s == StateSAIS.infected)
        )
        return len(graph) == total_infected or total_infected == 0



    def _get_nodes_state_mapping(self) -> list[int]:
        return [self.state_color_mapping[s] for s in nx.get_node_attributes(self.graph,'state').values()]        
    
    def _plot_current_state(self, ax):
        ax.clear()
        nx.draw_networkx(
            self.graph, 
            pos=self.pos, 
            with_labels=False, 
            node_color=self._get_nodes_state_mapping(),
            node_size=7,
            edge_color='gray',
            width=0.3,
            alpha=0.8,
            ax=ax
        )
        ax.set_title(f"SAIS model on the graph")
        ax.set_axis_off()



# %% [markdown]
# ### Tests

# %%
g = nx.erdos_renyi_graph(1000, 0.15)
params = ParamsAlertSpreadingSAIS(
    beta=0.001,
    beta_a=0.0005,
    delta=0.05,
    kappa_a=0.04,
    kappa_s=0.06,
)
I0=50

# %%
model = AlertSpreadingSAIS(g, params)
model.init_state(I0=I0, seed=1)
stats = model.run_simulation(200)

# %%
fig, axs = plt.subplots(figsize=(5,5))
#Defined
healthy =np.array(stats["suspectible_count"]) + np.array(stats["alert_count"]) 
stats["infected_count"]
axs.plot(stats["infected_count"], label='Infected')
axs.plot(healthy, label='Not Infected')
axs.set_xlabel('Iterations')
axs.set_ylabel('Number of Infected Nodes')
axs.set_title('AlertSpreadingSAIS Model: Infected Over Time [My implelemtation]')
axs.grid()
plt.legend()
plt.grid(True)
plt.show()

