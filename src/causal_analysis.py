"""
causal_analysis.py: Implementation of causal analysis and Bayesian networks for waste analysis.
"""

import numpy as np
import pandas as pd
import networkx as nx
import pymc as pm
from typing import Dict, List, Tuple, Optional, Union, Callable
import arviz as az
from scipy import stats
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class RegressionResult:
    """Container for regression results."""
    coefficients: Dict[str, Tuple[float, float]]  # (mean, std)
    predictions: np.ndarray
    prediction_intervals: np.ndarray  # Shape (n_samples, 2) for lower and upper bounds
    r2_score: float
    model_summary: str

class WasteCausalNetwork:
    def __init__(self):
        """Initialize a Bayesian network for causal analysis of waste processes."""
        self.graph = nx.DiGraph()
        self.data = pd.DataFrame()
        self.model = None
        self.trace = None
        self.variables = {}
        self.regression_results = {}

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
            # Create variables for each feature
            feature_vars = {}
            for col in self.data.columns:
                if col != 'waste':  # Skip target variable
                    feature_vars[col] = pm.Normal(
                        col,
                        mu=self.data[col].mean(),
                        sigma=self.data[col].std(),
                        observed=self.data[col].values
                    )
            
            # Create coefficients
            intercept = pm.Normal('intercept', mu=0, sigma=10)
            coeffs = {
                col: pm.Normal(f'beta_{col}', mu=0, sigma=2)
                for col in feature_vars.keys()
            }
            
            # Linear combination
            mu = intercept
            for col, coeff in coeffs.items():
                mu = mu + coeff * feature_vars[col]
            
            # Likelihood
            sigma = pm.HalfNormal('sigma', sigma=1)
            waste = pm.Normal('waste', mu=mu, sigma=sigma, observed=self.data['waste'])
            
    def fit(self, samples=2000):
        """Fit the Bayesian model using MCMC."""
        if self.model is None:
            self.build_model()
            
        with self.model:
            self.trace = pm.sample(samples, tune=1000)
            
            # Extract regression results
            coefficients = {}
            for var in self.model.named_vars:
                if var.startswith('beta_') or var == 'intercept':
                    trace_vals = self.trace.posterior[var].values.flatten()
                    coefficients[var.replace('beta_', '')] = (float(trace_vals.mean()), float(trace_vals.std()))
            
            # Calculate predictions
            X = self.data.drop('waste', axis=1)
            predictions = (
                coefficients['intercept'][0] +
                sum(coefficients[col][0] * X[col] for col in X.columns)
            )
            
            # Calculate prediction intervals
            residuals = self.data['waste'] - predictions
            std_residuals = residuals.std()
            prediction_intervals = np.column_stack([
                predictions - 1.96 * std_residuals,
                predictions + 1.96 * std_residuals
            ])
            
            # Calculate R² score
            r2 = 1 - (residuals ** 2).sum() / ((self.data['waste'] - self.data['waste'].mean()) ** 2).sum()
            
            # Create model summary
            summary = (
                f"Model Summary:\n"
                f"Number of observations: {len(self.data)}\n"
                f"R² score: {r2:.3f}\n"
                f"Residual std: {std_residuals:.3f}"
            )
            
            # Store regression results
            self.regression_results = {
                'waste': RegressionResult(
                    coefficients=coefficients,
                    predictions=predictions,
                    prediction_intervals=prediction_intervals,
                    r2_score=r2,
                    model_summary=summary
                )
            }
        
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

    def fit_regression(
        self,
        target: str,
        features: List[str],
        model_type: str = 'linear',
        link_function: Optional[Callable] = None,
        samples: int = 2000
    ) -> RegressionResult:
        """
        Fit a Bayesian regression model.
        
        Args:
            target: Target variable name
            features: List of feature variable names
            model_type: Type of regression ('linear' or 'logistic')
            link_function: Optional custom link function for GLM
            samples: Number of MCMC samples
            
        Returns:
            RegressionResult object containing coefficients, predictions, etc.
        """
        X = self.data[features].values
        y = self.data[target].values
        
        with pm.Model() as model:
            # Standardize features
            X_standardized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            
            # Priors for coefficients
            alpha = pm.Normal('alpha', mu=0, sigma=10)
            betas = pm.Normal('betas', mu=0, sigma=2, shape=len(features))
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # Linear predictor
            mu = alpha + pm.math.dot(X_standardized, betas)
            
            # Model type
            if model_type == 'linear':
                # Normal likelihood for linear regression
                y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
            elif model_type == 'logistic':
                # Bernoulli likelihood with logistic link for classification
                p = pm.math.sigmoid(mu) if link_function is None else link_function(mu)
                y_obs = pm.Bernoulli('y_obs', p=p, observed=y)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Sample from posterior
            trace = pm.sample(samples, tune=1000)
            
            # Get coefficient summaries
            coef_means = {}
            coef_stds = {}
            
            # Intercept
            alpha_samples = trace.posterior['alpha'].values.flatten()
            coef_means['intercept'] = np.mean(alpha_samples)
            coef_stds['intercept'] = np.std(alpha_samples)
            
            # Feature coefficients
            beta_samples = trace.posterior['betas'].values.reshape(-1, len(features))
            for i, feature in enumerate(features):
                coef_means[feature] = np.mean(beta_samples[:, i])
                coef_stds[feature] = np.std(beta_samples[:, i])
            
            # Generate predictions
            if model_type == 'linear':
                y_pred = coef_means['intercept'] + np.dot(X_standardized, 
                    [coef_means[f] for f in features])
            else:
                linear_pred = coef_means['intercept'] + np.dot(X_standardized,
                    [coef_means[f] for f in features])
                y_pred = 1 / (1 + np.exp(-linear_pred))
            
            # Calculate prediction intervals
            pred_samples = np.zeros((len(y), samples))
            for i in range(samples):
                alpha_i = alpha_samples[i]
                beta_i = beta_samples[i]
                pred_i = alpha_i + np.dot(X_standardized, beta_i)
                if model_type == 'logistic':
                    pred_i = 1 / (1 + np.exp(-pred_i))
                pred_samples[:, i] = pred_i
            
            pred_intervals = np.percentile(pred_samples, [2.5, 97.5], axis=1).T
            
            # Calculate R² score for linear regression
            if model_type == 'linear':
                r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
            else:
                r2 = None  # Not applicable for logistic regression
            
            # Create model summary
            r2_str = f"{r2:.3f}" if r2 is not None else "N/A"
            summary = (
                f"Bayesian {model_type.capitalize()} Regression Results\n"
                f"Number of observations: {len(y)}\n"
                f"Number of features: {len(features)}\n"
                f"R² Score: {r2_str}\n\n"
                "Coefficients:\n"
            )
            
            summary += f"{'Parameter':<20} {'Mean':>10} {'Std':>10}\n"
            summary += "-" * 40 + "\n"
            summary += f"{'Intercept':<20} {coef_means['intercept']:>10.3f} {coef_stds['intercept']:>10.3f}\n"
            for feature in features:
                summary += f"{feature:<20} {coef_means[feature]:>10.3f} {coef_stds[feature]:>10.3f}\n"
            
            # Store results
            coefficients = {
                name: (coef_means[name], coef_stds[name])
                for name in ['intercept'] + features
            }
            
            result = RegressionResult(
                coefficients=coefficients,
                predictions=y_pred,
                prediction_intervals=pred_intervals,
                r2_score=r2,
                model_summary=summary
            )
            
            self.regression_results[target] = result
            return result

    def plot_regression_results(
        self,
        target: str,
        feature: Optional[str] = None,
        show_intervals: bool = True
    ) -> plt.Figure:
        """
        Plot regression results.
        
        Args:
            target: Target variable name
            feature: Optional feature to plot against (for 2D visualization)
            show_intervals: Whether to show prediction intervals
        """
        if target not in self.regression_results:
            raise ValueError(f"No regression results found for {target}")
            
        result = self.regression_results[target]
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if feature is not None:
            # 2D visualization
            x = self.data[feature].values
            y = self.data[target].values
            y_pred = result.predictions
            
            # Sort by x for clean visualization
            sort_idx = np.argsort(x)
            x = x[sort_idx]
            y = y[sort_idx]
            y_pred = y_pred[sort_idx]
            
            # Plot actual vs predicted
            ax.scatter(x, y, alpha=0.5, label='Actual')
            ax.plot(x, y_pred, 'r-', label='Predicted')
            
            if show_intervals and result.prediction_intervals is not None:
                intervals = result.prediction_intervals[sort_idx]
                ax.fill_between(x, intervals[:, 0], intervals[:, 1],
                              alpha=0.2, color='r', label='95% PI')
            
            ax.set_xlabel(feature)
            ax.set_ylabel(target)
            
        else:
            # 1D visualization (actual vs predicted)
            y = self.data[target].values
            y_pred = result.predictions
            
            ax.scatter(y, y_pred, alpha=0.5)
            
            # Plot diagonal line
            min_val = min(min(y), min(y_pred))
            max_val = max(max(y), max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--',
                   label='Perfect Prediction')
            
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
        
        ax.set_title(f'Regression Results for {target}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig

    def visualize_causal_graph(self, show_effects: bool = True) -> plt.Figure:
        """Create a visualization of the causal graph with effect sizes."""
        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, ax=ax, node_color='lightblue',
                             node_size=2000, alpha=0.7)
        nx.draw_networkx_labels(self.graph, pos)
        
        # Draw edges with effect sizes
        if show_effects:
            edge_labels = {}
            for u, v, data in self.graph.edges(data=True):
                if 'effect_size' in data:
                    edge_labels[(u, v)] = f"{data['effect_size']:.2f}"
            
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        
        nx.draw_networkx_edges(self.graph, pos, ax=ax, edge_color='gray',
                             arrowsize=20)
        
        # Add regression results if available
        for target, result in self.regression_results.items():
            if target in pos:
                x, y = pos[target]
                text = f"\nR² = {result.r2_score:.2f}" if result.r2_score else ""
                ax.annotate(text, (x, y), xytext=(0, -20),
                          textcoords="offset points", ha='center')
        
        ax.set_title("Causal Network Structure")
        ax.axis('off')
        
        return fig
