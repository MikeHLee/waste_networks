"""
advanced_network.py: Advanced network modeling functionality with support for
complex node types, inventory management, and multi-dimensional edge properties.
"""

from enum import Enum
from typing import Dict, List, Tuple, Any, Union, Callable, Optional, Set
import networkx as nx
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
from .causal_analysis import WasteCausalNetwork, RegressionResult

class NodeType(Enum):
    SOLUTION_PROVIDER = "solution_provider"
    END_CONSUMER = "end_consumer"
    INITIAL_PRODUCER = "initial_producer"
    FOOD_PROCESSOR = "food_processor"
    FOOD_HANDLER = "food_handler"

class EdgeType(Enum):
    INVENTORY = "inventory"
    SERVICE = "service"
    CURRENCY = "currency"

class FlowType(Enum):
    """Types of flows that can be analyzed."""
    FOOD = "food"
    CURRENCY = "currency"
    SERVICE = "service"
    INVENTORY = "inventory"  # General inventory flow type for analysis

@dataclass
class Inventory:
    """Represents inventory with mass, value and composition."""
    mass: float
    value: float
    composition: Dict[str, float]  # Component -> percentage
    currency_type: str = "USD"

class WasteFunction(ABC):
    """Abstract base class for waste calculation functions."""
    @abstractmethod
    def calculate(self, **kwargs) -> float:
        """Calculate waste percentage based on input parameters."""
        pass

class StaticWaste(WasteFunction):
    """Static waste percentage."""
    def __init__(self, fraction: float):
        self.fraction = fraction
    
    def calculate(self, **kwargs) -> float:
        return self.fraction

class TimeBasedWaste(WasteFunction):
    """Time-dependent waste calculation."""
    def __init__(self, base_rate: float, time_coefficient: float):
        self.base_rate = base_rate
        self.time_coefficient = time_coefficient
    
    def calculate(self, time: float, **kwargs) -> float:
        return self.base_rate + (self.time_coefficient * time)

class MultiVariableWaste(WasteFunction):
    """Complex waste function based on multiple variables."""
    def __init__(self, func: Callable):
        self.func = func
    
    def calculate(self, **kwargs) -> float:
        return self.func(**kwargs)

class CausalWasteFunction:
    """Waste function based on causal regression model."""
    def __init__(self, regression_result: RegressionResult):
        self.regression = regression_result
        
    def __call__(self, **features) -> float:
        """Calculate waste based on input features.
        
        Args:
            **features: Feature values matching regression model variables
        Returns:
            Predicted waste value
        """
        # Ensure all required features are provided
        required_features = set(self.regression.coefficients.keys()) - {'intercept'}
        if not all(f in features for f in required_features):
            raise ValueError(f"Missing required features: {required_features - set(features.keys())}")
            
        # Calculate prediction using regression coefficients
        waste = self.regression.coefficients.get('intercept', (0.0, 0.0))[0]
        for feature, value in features.items():
            if feature in self.regression.coefficients:
                coef = self.regression.coefficients[feature][0]  # Use mean coefficient
                waste += coef * value
                
        return max(0.0, min(1.0, waste))  # Clamp to valid waste range

class AdvancedNode:
    """Base class for all node types in the advanced network."""
    def __init__(self, node_id: str, node_type: NodeType):
        self.node_id = node_id
        self.node_type = node_type
        self.input_inventory: Dict[str, Inventory] = {}
        self.output_inventory: Dict[str, Inventory] = {}
        self.holding_periods: Dict[str, float] = {}
        self.waste_function = None
        self.features = {}  # Current feature values
        self.activities: Dict[str, Dict[str, float]] = {}  # Activity -> {cost, revenue}

    def add_activity(self, name: str, cost: float, revenue: float):
        """Add an activity with associated costs and revenues."""
        self.activities[name] = {"cost": cost, "revenue": revenue}

    def set_waste_function(self, regression_result: RegressionResult):
        """Set waste function from regression results."""
        self.waste_function = CausalWasteFunction(regression_result)
        
    def set_features(self, **features):
        """Update current feature values."""
        self.features.update(features)
        
    def calculate_waste(self) -> float:
        """Calculate current waste based on features."""
        if self.waste_function is None:
            return 0.0
        return self.waste_function(**self.features)

class SolutionProvider(AdvancedNode):
    """Node that provides services affecting edge costs."""
    def __init__(self, node_id: str):
        super().__init__(node_id, NodeType.SOLUTION_PROVIDER)
        self.service_effects: Dict[str, float] = {}

    def add_service_effect(self, service_name: str, effect_multiplier: float):
        """Add a service that affects edge costs."""
        self.service_effects[service_name] = effect_multiplier

class EndConsumer(AdvancedNode):
    """Terminal node in the food network."""
    def __init__(self, node_id: str):
        super().__init__(node_id, NodeType.END_CONSUMER)

class InitialProducer(AdvancedNode):
    """Starting node in the food network."""
    def __init__(self, node_id: str):
        super().__init__(node_id, NodeType.INITIAL_PRODUCER)
        self.material_inputs: Dict[str, float] = {}

class FoodProcessor(AdvancedNode):
    """Node that transforms food composition."""
    def __init__(self, node_id: str):
        super().__init__(node_id, NodeType.FOOD_PROCESSOR)
        self.transformation_rules: Dict[str, Dict[str, float]] = {}

    def add_transformation_rule(self, input_type: str, output_composition: Dict[str, float]):
        """Add a rule for how inputs are transformed into outputs."""
        self.transformation_rules[input_type] = output_composition

class FoodHandler(AdvancedNode):
    """Node that handles food without transformation."""
    def __init__(self, node_id: str):
        super().__init__(node_id, NodeType.FOOD_HANDLER)

class AdvancedEdge:
    """Base class for all edge types in the advanced network."""
    def __init__(self, edge_type: EdgeType):
        self.edge_type = edge_type
        self.waste_function = None
        self.features = {}
        
    def set_waste_function(self, regression_result: RegressionResult):
        """Set waste function from regression results."""
        self.waste_function = CausalWasteFunction(regression_result)
        
    def set_features(self, **features):
        """Update current feature values."""
        self.features.update(features)
        
    def calculate_waste(self) -> float:
        """Calculate current waste based on features."""
        if self.waste_function is None:
            return 0.0
        return self.waste_function(**self.features)

class InventoryEdge(AdvancedEdge):
    """Edge representing inventory transfer."""
    def __init__(self, capacity: float):
        super().__init__(EdgeType.INVENTORY)
        self.capacity = capacity
        self.current_flow = 0.0

class ServiceEdge(AdvancedEdge):
    """Edge representing service provision."""
    def __init__(self, service_name: str, effect_multiplier: float):
        super().__init__(EdgeType.SERVICE)
        self.service_name = service_name
        self.effect_multiplier = effect_multiplier

class CurrencyEdge(AdvancedEdge):
    """Edge representing currency transfer."""
    def __init__(self, currency_type: str = "USD"):
        super().__init__(EdgeType.CURRENCY)
        self.currency_type = currency_type
        self.amount = 0.0
        self.is_owed = False

class AdvancedWasteNetwork:
    """Advanced network model supporting complex nodes, edges, and waste calculations."""
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, AdvancedNode] = {}
        
    def add_node(self, node: AdvancedNode):
        """Add a node to the network."""
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id, node_type=node.node_type)
        
    def add_edge(self, source: str, target: str, edge: AdvancedEdge) -> int:
        """Add an edge to the network. Returns edge key."""
        if edge.edge_type == EdgeType.INVENTORY and \
           self.nodes[target].node_type == NodeType.INITIAL_PRODUCER:
            raise ValueError("Cannot add food transfer edge to InitialProducer")
        
        if edge.edge_type == EdgeType.INVENTORY and \
           self.nodes[source].node_type == NodeType.END_CONSUMER:
            raise ValueError("Cannot add food transfer edge from EndConsumer")
            
        return self.graph.add_edge(source, target, edge=edge)
    
    def get_edges_by_type(self, flow_type: FlowType) -> Dict[Tuple[str, str], List[AdvancedEdge]]:
        """Get all edges of a specific flow type."""
        edge_map = defaultdict(list)
        edge_type = {
            FlowType.FOOD: EdgeType.INVENTORY,
            FlowType.CURRENCY: EdgeType.CURRENCY,
            FlowType.SERVICE: EdgeType.SERVICE,
            FlowType.INVENTORY: EdgeType.INVENTORY
        }[flow_type]
        
        for source, target, edge_data in self.graph.edges(data=True):
            edge = edge_data['edge']
            if edge.edge_type == edge_type:
                edge_map[(source, target)].append(edge)
                
        return dict(edge_map)
    
    def get_valid_endpoints(self, flow_type: FlowType) -> Tuple[List[str], List[str]]:
        """Get valid start and end nodes for a given flow type."""
        if flow_type == FlowType.FOOD:
            starts = [n for n, data in self.nodes.items() 
                     if data.node_type == NodeType.INITIAL_PRODUCER]
            ends = [n for n, data in self.nodes.items() 
                   if data.node_type == NodeType.END_CONSUMER]
        elif flow_type == FlowType.INVENTORY:
            starts = [n for n, data in self.nodes.items() 
                     if data.node_type not in [NodeType.END_CONSUMER]]
            ends = [n for n, data in self.nodes.items() 
                   if data.node_type not in [NodeType.INITIAL_PRODUCER]]
        else:
            starts = list(self.nodes.keys())
            ends = list(self.nodes.keys())
            
        return starts, ends
    
    def combine_parallel_edges(
        self,
        edges: List[AdvancedEdge],
        cost_func: Optional[Callable] = None
    ) -> float:
        """Combine costs of parallel edges."""
        if cost_func:
            return cost_func(edges)
        
        # Default to average waste
        return np.mean([edge.calculate_waste() for edge in edges])
    
    def find_minimum_path(
        self,
        flow_type: FlowType,
        sources: Optional[List[str]] = None,
        targets: Optional[List[str]] = None,
        cost_func: Optional[Callable] = None,
        capacity_constraints: Optional[Dict[str, float]] = None
    ) -> List[Tuple[List[str], float]]:
        """Find minimum cost paths between sources and targets."""
        # Create a new graph for path finding
        G = nx.DiGraph()
        
        # Add nodes and edges based on flow type
        for node_id, node in self.nodes.items():
            G.add_node(node_id)
        
        for (u, v, data) in self.graph.edges(data=True):
            edge = data['edge']
            if edge.edge_type == flow_type:
                G.add_edge(u, v, weight=edge.calculate_waste())
        
        # Get default sources and targets if not specified
        if sources is None:
            sources = [n for n, d in self.graph.nodes(data=True)
                      if isinstance(self.nodes[n], InitialProducer)]
        if targets is None:
            targets = [n for n, d in self.graph.nodes(data=True)
                      if isinstance(self.nodes[n], EndConsumer)]
        
        # Find shortest paths
        paths = []
        for source in sources:
            for target in targets:
                try:
                    path = nx.shortest_path(G, source, target, weight='weight')
                    cost = sum(G[path[i]][path[i+1]]['weight'] 
                             for i in range(len(path)-1))
                    paths.append((path, cost))
                except nx.NetworkXNoPath:
                    continue
        
        # Sort by cost
        paths.sort(key=lambda x: x[1])
        return paths

    def calculate_path_waste(
        self,
        path: List[str],
        time: float = 24.0,
        temperature: Optional[float] = None,
        humidity: Optional[float] = None,
        **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate total waste along a path."""
        total_waste = 0.0
        waste_breakdown = {}
        
        # Calculate node waste
        for node in path:
            params = {"time": time, **kwargs}
            if temperature is not None:
                params["temperature"] = temperature
            if humidity is not None:
                params["humidity"] = humidity
                
            node_waste = self.nodes[node].calculate_waste()
            total_waste += node_waste
            waste_breakdown[node] = node_waste
            
        # Calculate edge waste
        for i in range(len(path)-1):
            source, target = path[i], path[i+1]
            edge_key = (source, target)
            edge_waste = 0.0
            
            for _, _, edge_data in self.graph.edges(data=True):
                edge = edge_data['edge']
                if isinstance(edge, InventoryEdge):
                    edge_waste = edge.calculate_waste()
                    total_waste += edge_waste
                    
            if edge_waste > 0:
                waste_breakdown[f"{source}->{target}"] = edge_waste
                    
        return total_waste, waste_breakdown

    def visualize_network(
        self,
        highlight_path: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        regression_results: Optional[Dict[str, RegressionResult]] = None
    ) -> Optional[plt.Axes]:
        """Create a detailed visualization of the network with annotations."""
        if ax is None:
            _, ax = plt.subplots(figsize=(15, 10))
            
        # Create layout
        pos = nx.spring_layout(self.graph, k=2)
        
        # Draw nodes
        for node_id, node in self.nodes.items():
            color = '#BAE1FF' if node_id in (highlight_path or []) else '#E8F8F5'
            if node.node_type == NodeType.SOLUTION_PROVIDER:
                shape = 'rectangle'
                size = 2000
            else:
                shape = 'circle'
                size = 1500
                
            if shape == 'rectangle':
                rect = Rectangle(
                    (pos[node_id][0] - 0.05, pos[node_id][1] - 0.05),
                    0.1, 0.1,
                    facecolor=color,
                    edgecolor='black'
                )
                ax.add_patch(rect)
            else:
                circle = Circle(
                    pos[node_id],
                    radius=0.05,
                    facecolor=color,
                    edgecolor='black'
                )
                ax.add_patch(circle)
                
            # Add node label
            ax.annotate(
                f"{node.node_type.value}\n{node_id}",
                pos[node_id],
                xytext=(0, 20),
                textcoords="offset points",
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8)
            )
            
            # Add waste value if available
            if hasattr(node, 'calculate_waste'):
                waste = node.calculate_waste()
                if waste > 0:
                    ax.annotate(
                        f"Waste: {waste:.1%}",
                        pos[node_id],
                        xytext=(0, -20),
                        textcoords="offset points",
                        ha='center',
                        va='top',
                        bbox=dict(boxstyle='round,pad=0.5', fc='#FFE4E1', alpha=0.8)
                    )
        
        # Draw edges
        for source, target, edge_data in self.graph.edges(data=True):
            edge = edge_data['edge']
            color = '#4CAF50' if source in (highlight_path or []) and target in (highlight_path or []) else '#2C3E50'
            style = '--' if edge.edge_type == EdgeType.SERVICE else '-'
            width = 3 if edge.edge_type == EdgeType.INVENTORY else 2
            
            # Draw edge
            arrow = FancyArrowPatch(
                posA=pos[source],
                posB=pos[target],
                arrowstyle=f'-|>,head_length=15,head_width=10',
                connectionstyle='arc3,rad=0.2',
                color=color,
                linestyle=style,
                linewidth=width
            )
            ax.add_patch(arrow)
            
            # Add edge label
            mid_point = ((pos[source][0] + pos[target][0])/2, (pos[source][1] + pos[target][1])/2)
            edge_label = edge.edge_type.value
            if hasattr(edge, 'calculate_waste'):
                waste = edge.calculate_waste()
                if waste > 0:
                    edge_label += f"\nWaste: {waste:.1%}"
            
            ax.annotate(
                edge_label,
                mid_point,
                xytext=(0, 10),
                textcoords="offset points",
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8)
            )
        
        # Add regression results if available
        if regression_results:
            text_y = 0.98
            for name, result in regression_results.items():
                bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9)
                summary = f"{name} Model:\n"
                for var, (coef, std) in result.coefficients.items():
                    summary += f"{var}: {coef:.3f} ± {std:.3f}\n"
                summary += f"R² = {result.r2_score:.3f}"
                
                # Position at top of plot
                ax.text(0.02, text_y, summary,
                       transform=ax.transAxes,
                       verticalalignment='top',
                       bbox=bbox_props,
                       fontsize=8)
                text_y -= 0.15  # Offset for next model
        
        # Set plot limits and title
        ax.set_xlim((-1.2, 1.2))
        ax.set_ylim((-1.2, 1.2))
        ax.set_title("Waste Network Analysis")
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return None
            
        return ax

    def visualize_path_waste(self, path, save_path=None):
        """Create a visualization focusing on waste along a specific path."""
        waste, breakdown = self.calculate_path_waste(path)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))
        
        # Plot 1: Network with highlighted path
        self.visualize_network(highlight_path=path, ax=ax1)
        ax1.set_title("Network Path Visualization", pad=20)
        
        # Plot 2: Waste breakdown
        locations = list(breakdown.keys())
        waste_values = list(breakdown.values())
        
        bars = ax2.bar(locations, waste_values)
        ax2.set_title("Waste Breakdown by Location", pad=20)
        ax2.set_xlabel("Location")
        ax2.set_ylabel("Waste Percentage")
        
        # Rotate x-axis labels for better readability
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}', ha='center', va='bottom')
        
        # Add total waste annotation
        ax2.text(0.95, 0.95, f'Total Waste: {waste:.1%}',
                transform=ax2.transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                horizontalalignment='right',
                verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            return fig
