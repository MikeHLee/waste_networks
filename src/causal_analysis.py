"""
causal_analysis.py: Implementation of causal analysis and Bayesian networks for waste analysis.
"""

import numpy as np
import pandas as pd
import networkx as nx
import pymc as pm
from typing import Dict, List, Tuple, Optional
import arviz as az
from scipy import stats

class WasteCausalNetwork:
    def __init__(self):
        """Initialize a Bayesian network for causal analysis of waste processes."""
        self.graph = nx.DiGraph()
        self.data = pd.DataFrame()
        self.model = None
        self.trace = None
        self.variables = {}
        
    def add_node(self, node_id: str, node_type: str, 
                 distribution: str = 'normal',
                 params: Dict = None):
        """
        Add a node to the causal network.
        
        Args:
            node_id: Unique identifier for the node
            node_type: Type of node (cause, effect, confounder)
            distribution: Statistical distribution for this variable
            params: Parameters for the distribution
        """
        self.graph.add_node(node_id,
                           node_type=node_type,
                           distribution=distribution,
                           params=params or {})
        
    def add_edge(self, source: str, target: str, effect_size: float = None):
        """Add a causal relationship between nodes."""
        self.graph.add_edge(source, target, effect_size=effect_size)
        
    def add_data(self, data: pd.DataFrame):
        """Add observational data for causal inference."""
        self.data = data
        
    def build_model(self):
        """Build a PyMC model based on the network structure."""
        with pm.Model() as self.model:
            # Create variables dictionary to store PyMC variables
            self.variables = {}
            
            # First pass: Create all root nodes (nodes with no parents)
            for node in self.graph.nodes():
                if len(list(self.graph.predecessors(node))) == 0:
                    node_data = self.graph.nodes[node]
                    dist = node_data['distribution']
                    params = node_data['params']
                    
                    if dist == 'normal':
                        if node in self.data.columns:
                            # Observed variable
                            self.variables[node] = pm.Normal(
                                node,
                                mu=params.get('mu', 0.0),
                                sigma=params.get('sigma', 1.0),
                                observed=self.data[node].values
                            )
                        else:
                            # Latent variable
                            self.variables[node] = pm.Normal(
                                node,
                                mu=params.get('mu', 0.0),
                                sigma=params.get('sigma', 1.0)
                            )
            
            # Second pass: Create all other nodes in topological order
            for node in nx.topological_sort(self.graph):
                if node not in self.variables:  # Skip if already created
                    node_data = self.graph.nodes[node]
                    dist = node_data['distribution']
                    params = node_data['params']
                    
                    # Get parent nodes and their effect sizes
                    parents = list(self.graph.predecessors(node))
                    parent_effects = [self.graph[p][node]['effect_size'] 
                                    for p in parents]
                    
                    # Calculate mean based on parents
                    parent_terms = sum(self.variables[p] * effect 
                                    for p, effect in zip(parents, parent_effects))
                    
                    if dist == 'normal':
                        if node in self.data.columns:
                            # Observed variable
                            self.variables[node] = pm.Normal(
                                node,
                                mu=parent_terms,
                                sigma=params.get('sigma', 1.0),
                                observed=self.data[node].values
                            )
                        else:
                            # Latent variable
                            self.variables[node] = pm.Normal(
                                node,
                                mu=parent_terms,
                                sigma=params.get('sigma', 1.0)
                            )
        
    def fit(self, samples=2000):
        """Fit the Bayesian model using MCMC."""
        if self.model is None:
            self.build_model()
            
        with self.model:
            self.trace = pm.sample(samples, tune=1000)
            
    def get_causal_effect(self, cause: str, effect: str) -> Tuple[float, float]:
        """
        Calculate the causal effect between two variables.
        
        Returns:
            Tuple of (effect size, standard error)
        """
        if self.trace is None:
            raise ValueError("Model must be fit before calculating causal effects")
            
        # Calculate effect using the model's parameters
        effect_path = nx.shortest_path(self.graph, cause, effect)
        total_effect = 1.0
        
        # Multiply effect sizes along the path
        for i in range(len(effect_path) - 1):
            source, target = effect_path[i], effect_path[i + 1]
            total_effect *= self.graph[source][target]['effect_size']
            
        # Calculate standard error using bootstrap of the posterior samples
        n_bootstrap = 1000
        bootstrap_effects = []
        
        # Get posterior samples for effect variable
        effect_samples = self.trace.posterior[effect].values.mean(axis=0).flatten()
        
        # Get cause data (either observed or posterior mean)
        if cause in self.data.columns:
            cause_data = self.data[cause].values
        else:
            cause_data = self.trace.posterior[cause].values.mean(axis=0).flatten()
            
        n_samples = len(cause_data)
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            cause_bootstrap = cause_data[indices]
            effect_bootstrap = effect_samples[indices]
            
            # Calculate correlation for this bootstrap sample
            bootstrap_effect = np.corrcoef(cause_bootstrap, effect_bootstrap)[0, 1]
            bootstrap_effects.append(bootstrap_effect)
            
        return total_effect, np.std(bootstrap_effects)
        
    def discover_causal_structure(self, significance_threshold: float = 0.05):
        """
        Discover causal relationships from observational data.
        Uses PC algorithm-inspired approach for causal discovery.
        """
        n_vars = len(self.data.columns)
        
        # Start with complete undirected graph
        discovered_graph = nx.Graph()
        discovered_graph.add_nodes_from(self.data.columns)
        discovered_graph.add_edges_from([(i, j) 
                                       for i in self.data.columns 
                                       for j in self.data.columns if i < j])
        
        # Phase 1: Remove edges based on conditional independence
        for i in self.data.columns:
            for j in self.data.columns:
                if i < j:
                    # Test conditional independence
                    corr = stats.pearsonr(self.data[i], self.data[j])[1]
                    if corr > significance_threshold:
                        discovered_graph.remove_edge(i, j)
                        
        # Phase 2: Orient edges (simplified version)
        directed_graph = nx.DiGraph()
        directed_graph.add_nodes_from(discovered_graph.nodes())
        
        for i, j in discovered_graph.edges():
            # Use temporal information or domain knowledge if available
            # For now, use correlation strength to determine direction
            corr_i_j = abs(stats.pearsonr(self.data[i], self.data[j])[0])
            corr_j_i = abs(stats.pearsonr(self.data[j], self.data[i])[0])
            
            if corr_i_j > corr_j_i:
                directed_graph.add_edge(i, j)
            else:
                directed_graph.add_edge(j, i)
                
        return directed_graph
        
    def plot_causal_graph(self, highlight_path: List[str] = None):
        """Plot the causal network with optional path highlighting."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        
        # Draw nodes with different colors based on type
        node_colors = ['lightblue' if self.graph.nodes[node]['node_type'] == 'cause'
                      else 'lightgreen' if self.graph.nodes[node]['node_type'] == 'effect'
                      else 'lightgray' for node in self.graph.nodes()]
        
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors,
                             node_size=2000)
        nx.draw_networkx_labels(self.graph, pos)
        
        # Draw edges
        edge_labels = {(u, v): f"{d['effect_size']:.2f}"
                      for u, v, d in self.graph.edges(data=True)
                      if 'effect_size' in d}
        
        if highlight_path:
            path_edges = list(zip(highlight_path[:-1], highlight_path[1:]))
            nx.draw_networkx_edges(self.graph, pos,
                                 edgelist=path_edges,
                                 edge_color='r',
                                 width=2)
            
            other_edges = [(u, v) for u, v in self.graph.edges()
                          if (u, v) not in path_edges]
            nx.draw_networkx_edges(self.graph, pos,
                                 edgelist=other_edges,
                                 edge_color='gray')
        else:
            nx.draw_networkx_edges(self.graph, pos, edge_color='gray')
            
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        
        plt.title("Waste Causal Network")
        plt.axis('off')
        return plt
