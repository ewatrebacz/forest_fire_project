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
# # A-SIS implementation

# %%
import networkx as nx
import numpy as np
import networkx as nx

import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import seaborn as sns
import matplotlib.pyplot as plt
from pydantic import BaseModel, confloat
from enum import auto, Enum

# %%
g = nx.erdos_renyi_graph(1000, 0.1)

# Model selection
model = ep.SISModel(g)

# Model Configuration
cfg = mc.Configuration()
cfg.add_model_parameter('beta', 0.01)
cfg.add_model_parameter('lambda', 0.005)
cfg.add_model_parameter("fraction_infected", 0.05)
model.set_initial_status(cfg)

# Simulation execution
iterations = model.iteration_bunch(200)


# %%
def init_example_graph() -> nx.Graph:
    mat_bp = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0],
    ])
    g = nx.from_numpy_array(mat_bp)
    return g

def init_matrices(g: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    bp = nx.adjacency_matrix(g)
    bd = nx.adjacency_matrix(nx.line_graph(g))
    return bp, bd


# %%
class Probability(BaseModel):
    value: confloat(ge=0, le=1) 

class StateSIS(Enum):
    suspectible = 0
    infected = 1


# %%

class A_SIS:
    def __init__(
        self,
        graph: nx.Graph,
        x0: np.ndarray,
        y0: np.ndarray,
        gamma: float,
        beta: float,
        eps: Probability,
        mc_steps: int = 1_000,
    ) -> None:
        self.graph = graph
        self.pos = nx.spring_layout(self.graph, seed=3)

        self.mc_steps = mc_steps
        self.eps = eps
        self.r = beta / gamma
        # Each describe node state
        self.x = x0
        self.y = y0

        self.inc_mat = nx.incidence_matrix(self.graph)
        self.n, self.m = self.inc_mat.shape
    
    def init_state(self, I0, seed: int):
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
        ap_mat = self.inc_mat @ np.diag(self.y) @ self.inc_mat.T - np.diag(self.inc_mat @ self.y)
        ad_mat = self.inc_mat.T @ np.diag(self.x) @ self.inc_mat - np.diag(self.inc_mat.T @ self.x)
        
        x_next = self.r * (np.eye(self.n) - np.diag(self.x)) @ ap_mat @ self.x
        y_next = self.r * (np.eye(self.m) - np.diag(self.y)) @ ad_mat @ self.y

        self.x = x_next
        self.y = y_next

    def run_simulation(self):
        statistics = {
            "x": [],
            "y": [],
        }
        statistics["x"].append(self.x)
        statistics["y"].append(self.y)

        x_prev = self.x
        y_prev = self.y
        self._update_state()

        while np.linalg.norm(self.x - x_prev) / np.linalg.norm(x_prev) + np.linalg.norm(self.y - y_prev) / np.linalg.norm(y_prev) > self.eps:
            statistics["x"].append(self.x)
            statistics["y"].append(self.y)
            x_prev = self.x
            y_prev = self.y
            self._update_state()
        return statistics


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


# %%
p = 0.25
y0 = np.ones(4) * p
x0 = np.ones(4) * p
beta = 0.004
gamma = 0.001
model = A_SIS(
    graph=init_example_graph(),
    y0=y0,
    x0=x0,
    beta=beta,
    gamma=gamma,
    eps=0.01
)
print(beta/gamma)

# %%
nx.adjacency_matrix(init_example_graph()).toarray()
init_matrices(init_example_graph())[0].toarray()

# %%
model.run_simulation()

# %%
