# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: A-SIS
#     language: python
#     name: a-sis
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
import pandas as pd

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
    gamma: float = Field(..., gt=0, le=1)



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
    delta: float = Field(..., gt=0, le=1)
    kappa: float = Field(..., gt=0, lt=1)


# %%
class SAIS:
    state_color_mapping = {
        StateSAIS.suspectible: "blue",
        StateSAIS.infected: "red",
        StateSAIS.alert: "orange",
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
        next_states = {}

        trans_IS = np.random.random(len(infected_nodes)) <= self.params.delta
        recovered = {node: StateSAIS.suspectible for node, is_change in zip(infected_nodes, trans_IS) if is_change}
        dangered_suscteptible = [v for u in infected_nodes for v in self.graph.neighbors(u) if node_states[v] == StateSAIS.suspectible]
        trans_SA = np.random.random(len(dangered_suscteptible)) <= self.params.kappa
        alerted = {node: StateSAIS.alert for node, is_change in zip(dangered_suscteptible, trans_SA) if is_change}
        trans_SI = np.random.random(len(dangered_suscteptible)) <= self.params.beta
        infected = {node: StateSAIS.infected for node, is_change in zip(dangered_suscteptible, trans_SI) if is_change}
        dangered_alert = [v for u in infected_nodes for v in self.graph.neighbors(u) if node_states[v] == StateSAIS.alert]
        trans_AI = np.random.random(len(dangered_alert)) <= self.params.beta_a
        infected_in_alert = {node: StateSAIS.infected for node, is_change in zip(dangered_alert, trans_AI) if is_change}

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
params = ParamsSAIS(
    beta=0.001,
    beta_a=0.0005,
    delta=0.05,
    kappa=0.01,
)
I0=50

g = nx.erdos_renyi_graph(320, 0.2)
params = ParamsSAIS(
    beta=0.03,
    beta_a=0.03,
    delta=1,
    kappa=0.05,
)
I0=int(320*0.02)



# %%
model = SAIS(g, params)
model.init_state(I0=I0, seed=1)
stats = model.run_simulation(200)

# %%
fig, ax = plt.subplots(figsize = (10,5))

g1 = g.copy()

#Benchmark
model = ep.SISModel(g)
cfg = mc.Configuration()
cfg.add_model_parameter('beta', params.beta)
cfg.add_model_parameter('lambda', params.delta)
cfg.add_model_parameter("fraction_infected", 0.02)
model.set_initial_status(cfg)
iterations = model.iteration_bunch(200)
infected = [it['node_count'][1] / 320 for it in iterations]  # 1 = infected state
ax.plot(infected, label='SIS[NdLib]')
ax.set_xlabel('Iterations')
ax.set_ylabel('Number of Infected Nodes')
ax.set_title('SIS Model: Infected Over Time [NDlib]')
ax.grid()


#Defined
model = SAIS(g, params)
model.init_state(I0=I0, seed=1)
stats = model.run_simulation(200)
ax.plot(np.array(stats["infected_count"])/320, label='SAIS')

paramsSIS = ParamsSIS(beta=0.03, gamma=1)
model = SIS(g, paramsSIS)
model.init_state(I0=I0, seed=1)
stats = model.run_simulation(200)
ax.plot(np.array(stats["infected_count"])/320, label='SIS')
plt.legend()
plt.grid(True)


# %%
fig, axs = plt.subplots(figsize=(5,5))
#Defined
stats["infected_count"]
axs.plot(np.array(stats["infected_count"])/320, label='Infected')
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
        StateSAIS.alert: "orange",
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


# %% [markdown]
# # Alert Spreading SAICS

# %%
class StateAlertSpreadingSAICS(Enum):
    suspectible = auto()
    infected = auto()
    alert = auto()
    coruptor = auto()

class ParamsAlertSpreadingSAICS(BaseModel):
    beta: float = Field(..., ge=0, lt=1)
    beta_a: float = Field(..., ge=0, lt=1)
    beta_prim: float = Field(..., ge=0, lt=1)
    beta_a_prim: float = Field(..., ge=0, lt=1)
    delta: float = Field(..., ge=0, le=1)
    delta_a: float = Field(..., ge=0, le=1)
    kappa_s: float = Field(..., ge=0, lt=1)
    kappa_a: float = Field(..., ge=0, lt=1)


# %%
class AlertSpreadingSAICS:
    state_color_mapping = {
        StateAlertSpreadingSAICS.suspectible: "blue",
        StateAlertSpreadingSAICS.infected: "red",
        StateAlertSpreadingSAICS.alert: "orange",
        StateAlertSpreadingSAICS.coruptor: "gray",
    }
    
    def __init__(
        self,
        graph: nx.Graph,
        params: ParamsAlertSpreadingSAICS,
        pos_seed: int = 3
    ) -> None:
        self.graph = graph
        self.pos = nx.spring_layout(self.graph, seed=pos_seed)
        self.params = params
    
    def init_state(self, I0: int, seed: int):
        random.seed(seed)
        np.random.seed(seed=seed)
        nx.set_node_attributes(self.graph, StateAlertSpreadingSAICS.suspectible, "state")
        nx.set_node_attributes(
            self.graph,
            {
                node: {"state": StateAlertSpreadingSAICS.infected}
                for node in random.sample(list(self.graph.nodes), k=I0)
            }
        )
    
    def _update_state(self):
        """Asynchronous updates considered."""
        node_states = nx.get_node_attributes(self.graph, "state")
        infected_nodes = [n for n, state in node_states.items() if state == StateAlertSpreadingSAICS.infected]
        coruptor_nodes = [n for n, state in node_states.items() if state == StateAlertSpreadingSAICS.coruptor]
        general_infected_nodes = infected_nodes + coruptor_nodes
        alerted_nodes = [n for n, state in node_states.items() if state == StateAlertSpreadingSAICS.alert]
        next_states = {}

        throw_recovery = np.random.random(len(infected_nodes))
        trans_IA = throw_recovery <= self.params.delta_a
        recovered_alerted = {node: StateAlertSpreadingSAICS.alert for node, is_change in zip(infected_nodes, trans_IA) if is_change}
        trans_IS = (self.params.delta_a < throw_recovery) & (throw_recovery <= self.params.delta)
        recovered = {node: StateAlertSpreadingSAICS.suspectible for node, is_change in zip(infected_nodes, trans_IS) if is_change}

        dangered_suscteptible = [v for u in general_infected_nodes for v in self.graph.neighbors(u) if node_states[v] == StateAlertSpreadingSAICS.suspectible]
        throw_suspectible = np.random.random(len(dangered_suscteptible))
        trans_SC = throw_suspectible <= self.params.beta_prim
        corupted = {node: StateAlertSpreadingSAICS.coruptor for node, is_change in zip(dangered_suscteptible, trans_SC) if is_change}
        trans_SI =  (self.params.beta_prim < throw_suspectible) & (throw_suspectible <= self.params.beta)
        infected = {node: StateAlertSpreadingSAICS.infected for node, is_change in zip(dangered_suscteptible, trans_SI) if is_change}


        trans_SA = np.random.random(len(dangered_suscteptible)) <= self.params.kappa_s
        alerted_by_I = {node: StateAlertSpreadingSAICS.alert for node, is_change in zip(dangered_suscteptible, trans_SA) if is_change}
        
        neighbours_of_alerted = [v for u in alerted_nodes for v in self.graph.neighbors(u) if node_states[v] == StateAlertSpreadingSAICS.suspectible]
        trans_SA2 = np.random.random(len(neighbours_of_alerted)) <= self.params.kappa_a
        alerted_by_A = {node: StateAlertSpreadingSAICS.alert for node, is_change in zip(neighbours_of_alerted, trans_SA2) if is_change}
        
        dangered_alert = [v for u in general_infected_nodes for v in self.graph.neighbors(u) if node_states[v] == StateAlertSpreadingSAICS.alert]
        throw_alert = np.random.random(len(dangered_alert))
        trans_AC = throw_alert <= self.params.beta_a_prim
        corupted_in_alert = {node: StateAlertSpreadingSAICS.coruptor for node, is_change in zip(dangered_alert, trans_AC) if is_change}
        trans_AI = (self.params.beta_a_prim < throw_alert) & (throw_alert <= self.params.beta_a)
        infected_in_alert = {node: StateAlertSpreadingSAICS.infected for node, is_change in zip(dangered_alert, trans_AI) if is_change}

        next_states.update(recovered)
        next_states.update(recovered_alerted)
        #Infections should overwrite alerts
        next_states.update(alerted_by_I)
        next_states.update(alerted_by_A)
        next_states.update(infected)
        next_states.update(infected_in_alert)
        next_states.update(corupted)
        next_states.update(corupted_in_alert)

        nx.set_node_attributes(self.graph, {node: {"state": state} for node, state in next_states.items()})


    def run_simulation(self, max_steps: int):
        statistics = {
            "infected_count": [],
            "suspectible_count": [],
            "alert_count": [],
            "corrupted_count": [],
        }
        is_finished = False
        step_no = 0
        while not is_finished and step_no < max_steps:
            self._update_state()
            states = nx.get_node_attributes(self.graph, "state").values()
            statistics["infected_count"].append(
                sum(1 for s in states if s == StateAlertSpreadingSAICS.infected)
            )
            statistics["suspectible_count"].append(
                sum(1 for s in states if s == StateAlertSpreadingSAICS.suspectible)
            )
            statistics["alert_count"].append(
                sum(1 for s in states if s == StateAlertSpreadingSAICS.alert)
            )
            statistics["corrupted_count"].append(
                sum(1 for s in states if s == StateAlertSpreadingSAICS.coruptor)
            )
            step_no += 1
            is_finished = AlertSpreadingSAICS.is_finished(self.graph)
        return statistics

    @staticmethod
    def is_finished(graph: nx.Graph) -> bool:
        states = nx.get_node_attributes(graph, "state").values()
        total_infected = (
                sum(1 for s in states if s in (StateAlertSpreadingSAICS.infected, StateAlertSpreadingSAICS.coruptor))
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
            node_size=15,
            edge_color='gray',
            width=0.3,
            alpha=0.8,
            ax=ax
        )
        ax.set_title(f"StateAlertSpreadingSAICS model on the graph")
        ax.set_axis_off()



# %% [markdown]
# ## Tests

# %%
g = nx.erdos_renyi_graph(1000, 0.15)
params = ParamsAlertSpreadingSAICS(
    beta=0.001,
    beta_prim=0.00002,
    beta_a=0.0005,
    beta_a_prim=0.00001,
    delta=0.05,
    delta_a=0.01,
    kappa_a=0.04,
    kappa_s=0.06,
)
I0=50

# %%
model = AlertSpreadingSAICS(g, params)
model.init_state(I0=I0, seed=1)
stats = model.run_simulation(200)

    # %%
    fig, axs = plt.subplots(figsize=(5,5))
    #Defined
    healthy =np.array(stats["suspectible_count"]) + np.array(stats["alert_count"]) 
    infected = np.array(stats["infected_count"]) + np.array(stats["corrupted_count"])
    stats["infected_count"]
    axs.plot(infected, label='Infected')
    axs.plot(healthy, label='Not Infected')
    axs.set_xlabel('Iterations')
    axs.set_ylabel('Number of Infected Nodes')
    axs.set_title('AlertSpreadingSAICS Model: Infected Over Time [My implelemtation]')
    axs.grid()
    plt.legend()
    plt.grid(True)
    plt.show()


# %%
g = nx.erdos_renyi_graph(320, 0.2)
params = ParamsAlertSpreadingSAICS(
    beta=0.03,
    beta_prim=0.00,
    beta_a=0.01,
    beta_a_prim=0.0,
    delta=1,
    delta_a=0,
    kappa_a=0.0,
    kappa_s=0.05,
)
I0=int(320 * 0.2)

# %%
mcs = 50
total_stats = pd.DataFrame()
for idx in range(mcs):
    g = nx.erdos_renyi_graph(320, 0.2)
    model = AlertSpreadingSAICS(g, params)
    model.init_state(I0=I0, seed=3)
    stats = pd.DataFrame(model.run_simulation(200))
    stats["steps"] = np.arange(len(stats))
    total_stats = pd.concat([total_stats, stats], ignore_index=True)

# %%
df = total_stats.groupby("steps").mean()
fig, axs = plt.subplots(figsize=(5,5))
#Defined
healthy =(np.array(df["suspectible_count"]) + df["alert_count"])  / len(g)
infected = (df["infected_count"] + df["corrupted_count"]) / len(g)
stats["infected_count"]
axs.plot(infected, label='Infected')
axs.plot(healthy, label='Not Infected')
axs.set_xlabel('Dimensionless Time, $\\bar{t}$')
axs.set_ylabel('Population Fraction')
axs.set_title('Infection dynamics for AlertSpreadingSAICS')
axs.grid()
plt.legend()
plt.grid(True)
plt.show()


# %%
g = nx.erdos_renyi_graph(320, 0.2)
params = ParamsAlertSpreadingSAICS(
    beta=0.03,
    beta_prim=0.00,
    beta_a=0.01,
    beta_a_prim=0.0,
    delta=1,
    delta_a=0,
    kappa_a=0.9,
    kappa_s=0.05,
)
I0=int(320 * 0.2)

mcs = 50
total_stats = pd.DataFrame()
for idx in range(mcs):
    g = nx.erdos_renyi_graph(320, 0.2)
    model = AlertSpreadingSAICS(g, params)
    model.init_state(I0=I0, seed=3)
    stats = pd.DataFrame(model.run_simulation(200))
    stats["steps"] = np.arange(len(stats))
    total_stats = pd.concat([total_stats, stats], ignore_index=True)

df = total_stats.groupby("steps").mean()
fig, axs = plt.subplots(figsize=(5,5))
#Defined
healthy =(np.array(df["suspectible_count"]) + df["alert_count"])  / len(g)
infected = (df["infected_count"] + df["corrupted_count"]) / len(g)
axs.plot(infected, label='Infected')
axs.plot(healthy, label='Not Infected')
axs.set_xlabel('Dimensionless Time, $\\bar{t}$')
axs.set_ylabel('Population Fraction')
axs.set_title('Infection dynamics for AlertSpreadingSAICS')
axs.grid()
plt.legend()
plt.grid(True)
plt.show()


# %%
g = nx.erdos_renyi_graph(320, 0.2)
params = ParamsAlertSpreadingSAICS(
    beta=0.03,
    beta_prim=0.00,
    beta_a=0.01,
    beta_a_prim=0.0,
    delta=1,
    delta_a=0,
    kappa_a=0.5,
    kappa_s=0.05,
)
I0=int(320 * 0.02)

mcs = 50
total_stats = pd.DataFrame()
for idx in range(mcs):
    g = nx.erdos_renyi_graph(320, 0.2)
    model = AlertSpreadingSAICS(g, params)
    model.init_state(I0=I0, seed=3)
    stats = pd.DataFrame(model.run_simulation(200))
    stats["steps"] = np.arange(len(stats))
    total_stats = pd.concat([total_stats, stats], ignore_index=True)

df = total_stats.groupby("steps").mean()
fig, axs = plt.subplots(figsize=(5,5))
#Defined
healthy =(np.array(df["suspectible_count"]) + df["alert_count"])  / len(g)
infected = (df["infected_count"] + df["corrupted_count"]) / len(g)
axs.plot(infected, label='Infected')
axs.plot(healthy, label='Not Infected')
axs.set_xlabel('Dimensionless Time, $\\bar{t}$')
axs.set_ylabel('Population Fraction')
axs.set_title('Infection dynamics for AlertSpreadingSAICS')
axs.grid()
plt.legend()
plt.grid(True)
plt.show()


# %%
fig, axs = plt.subplots(figsize=(5,5))
#Defined
healthy =(np.array(stats["suspectible_count"]) + np.array(stats["alert_count"]) ) / len(g)
infected = (np.array(stats["infected_count"]) + np.array(stats["corrupted_count"])) / len(g)
stats["infected_count"]
axs.plot(infected, label='Infected')
axs.plot(healthy, label='Not Infected')
axs.set_xlabel('Dimensionless Time, $\\bar{t}$')
axs.set_ylabel('Population Fraction')
axs.set_title('Infection dynamics for AlertSpreadingSAICS')
axs.grid()
plt.legend()
plt.grid(True)
plt.show()


# %%
g = nx.watts_strogatz_graph(1000, k=4, p=10)
g = nx.random_regular_graph(3, 1000)
params = ParamsAlertSpreadingSAICS(
    beta=0.015,
    beta_prim=0.01,
    beta_a=0.005,
    beta_a_prim=0.0,
    delta=0.01,
    delta_a=0.01,
    kappa_a=0.003,
    kappa_s=0.001,
)
I0=3

# %%
model = AlertSpreadingSAICS(g, params)
model.init_state(I0=I0, seed=3)
stats = model.run_simulation(500)

# %%
fig, axs = plt.subplots(figsize=(5,5))
#Defined
healthy =(np.array(stats["suspectible_count"]) + np.array(stats["alert_count"]) ) / len(g)
infected = (np.array(stats["infected_count"]) + np.array(stats["corrupted_count"])) / len(g)
stats["infected_count"]
axs.plot(infected, label='Infected')
axs.plot(healthy, label='Not Infected')
axs.set_xlabel('Dimensionless Time, $\\bar{t}$')
axs.set_ylabel('Population Fraction')
axs.set_title('Infection dynamics for AlertSpreadingSAICS')
axs.grid()
plt.legend()
plt.grid(True)
plt.show()


# %%
g = nx.random_regular_graph(3, 20)
params = ParamsAlertSpreadingSAICS(
    beta=0.15,
    beta_prim=0.1,
    beta_a=0.05,
    beta_a_prim=0.0,
    delta=0.1,
    delta_a=0.1,
    kappa_a=0.03,
    kappa_s=0.1,
)
I0=3
model = AlertSpreadingSAICS(g, params)
model.init_state(I0=I0, seed=3)

# plot_state(model, 2)
# save_animation(
#     plot_model_simulation(model, steps=200),
#     "AlertSpreadingSAICS.mp4"

# )

# %%
