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

class AdvancedNode:
    """Base class for all node types in the advanced network."""
    def __init__(self, node_id: str, node_type: NodeType):
        self.node_id = node_id
        self.node_type = node_type
        self.input_inventory: Dict[str, Inventory] = {}
        self.output_inventory: Dict[str, Inventory] = {}
        self.holding_periods: Dict[str, float] = {}
        self.waste_function: WasteFunction = StaticWaste(0.0)
        self.activities: Dict[str, Dict[str, float]] = {}  # Activity -> {cost, revenue}

    def add_activity(self, name: str, cost: float, revenue: float):
        """Add an activity with associated costs and revenues."""
        self.activities[name] = {"cost": cost, "revenue": revenue}

    def set_waste_function(self, waste_func: WasteFunction):
        """Set the waste calculation function for this node."""
        self.waste_function = waste_func

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
        self.waste_function: WasteFunction = StaticWaste(0.0)

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
            FlowType.SERVICE: EdgeType.SERVICE
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
        else:
            starts = list(self.nodes.keys())
            ends = list(self.nodes.keys())
            
        return starts, ends
    
    def combine_parallel_edges(
        self,
        edges: List[AdvancedEdge],
        cost_func: Optional[Callable[[List[AdvancedEdge]], float]] = None
    ) -> float:
        """Combine parallel edges into a single cost."""
        if not cost_func:
            # Default cost function: average of waste rates
            return np.mean([edge.waste_function.calculate() for edge in edges])
        return cost_func(edges)
    
    def find_minimum_path(
        self,
        flow_type: FlowType,
        sources: Optional[List[str]] = None,
        targets: Optional[List[str]] = None,
        cost_func: Optional[Callable[[List[AdvancedEdge]], float]] = None,
        capacity_constraints: Optional[Dict[str, float]] = None
    ) -> List[Tuple[List[str], float]]:
        """Find minimum cost paths between sources and targets.
        
        Args:
            flow_type: Type of flow to analyze
            sources: List of source nodes (if None, uses valid start nodes)
            targets: List of target nodes (if None, uses valid end nodes)
            cost_func: Custom function to combine parallel edge costs
            capacity_constraints: Node capacity requirements {node_id: required_capacity}
            
        Returns:
            List of (path, cost) tuples satisfying the constraints
        """
        # Get valid endpoints if not specified
        valid_starts, valid_ends = self.get_valid_endpoints(flow_type)
        sources = sources or valid_starts
        targets = targets or valid_ends
        
        # Validate endpoints
        if not all(s in valid_starts for s in sources):
            raise ValueError(f"Invalid source nodes for {flow_type.value} flow")
        if not all(t in valid_ends for t in targets):
            raise ValueError(f"Invalid target nodes for {flow_type.value} flow")
            
        # Get relevant edges
        edges_by_type = self.get_edges_by_type(flow_type)
        
        # Create a new graph for pathfinding
        path_graph = nx.DiGraph()
        
        # Add edges with combined costs
        for (source, target), edges in edges_by_type.items():
            cost = self.combine_parallel_edges(edges, cost_func)
            path_graph.add_edge(source, target, weight=cost)
            
        # Find paths satisfying constraints
        paths = []
        for source in sources:
            for target in targets:
                try:
                    path = nx.shortest_path(
                        path_graph,
                        source=source,
                        target=target,
                        weight='weight'
                    )
                    cost = nx.shortest_path_length(
                        path_graph,
                        source=source,
                        target=target,
                        weight='weight'
                    )
                    
                    # Check capacity constraints
                    if capacity_constraints:
                        satisfies_capacity = True
                        for node in path:
                            if node in capacity_constraints:
                                node_edges = [e for e in edges_by_type.values()
                                           if isinstance(e[0], InventoryEdge)]
                                available_capacity = sum(e.capacity for e in node_edges)
                                if available_capacity < capacity_constraints[node]:
                                    satisfies_capacity = False
                                    break
                        if not satisfies_capacity:
                            continue
                            
                    paths.append((path, cost))
                except nx.NetworkXNoPath:
                    continue
                    
        return sorted(paths, key=lambda x: x[1])
    
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
                
            node_waste = self.nodes[node].waste_function.calculate(**params)
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
                    edge_waste = edge.waste_function.calculate(**params)
                    total_waste += edge_waste
                    
            if edge_waste > 0:
                waste_breakdown[f"{source}->{target}"] = edge_waste
                    
        return total_waste, waste_breakdown

    def visualize_network(self, highlight_path=None, save_path=None, ax=None):
        """Create a detailed visualization of the network with annotations."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 10))
        pos = nx.spring_layout(self.graph, k=2)
        
        # Collect node types
        producers = [n for n, d in self.graph.nodes(data=True) if isinstance(self.nodes[n], InitialProducer)]
        processors = [n for n, d in self.graph.nodes(data=True) if isinstance(self.nodes[n], FoodProcessor)]
        handlers = [n for n, d in self.graph.nodes(data=True) if isinstance(self.nodes[n], FoodHandler)]
        consumers = [n for n, d in self.graph.nodes(data=True) if isinstance(self.nodes[n], EndConsumer)]
        providers = [n for n, d in self.graph.nodes(data=True) if isinstance(self.nodes[n], SolutionProvider)]
        
        # Draw nodes with different colors and sizes based on type
        node_size = 3000
        nx.draw_networkx_nodes(self.graph, pos, nodelist=producers, node_color='lightgreen',
                             node_size=node_size, alpha=0.7, label='Producers', ax=ax)
        nx.draw_networkx_nodes(self.graph, pos, nodelist=processors, node_color='lightblue',
                             node_size=node_size, alpha=0.7, label='Processors', ax=ax)
        nx.draw_networkx_nodes(self.graph, pos, nodelist=handlers, node_color='orange',
                             node_size=node_size, alpha=0.7, label='Handlers', ax=ax)
        nx.draw_networkx_nodes(self.graph, pos, nodelist=consumers, node_color='pink',
                             node_size=node_size, alpha=0.7, label='Consumers', ax=ax)
        nx.draw_networkx_nodes(self.graph, pos, nodelist=providers, node_color='purple',
                             node_size=node_size, alpha=0.7, label='Service Providers', ax=ax)
        
        # Draw edges with different styles based on type
        edges = self.graph.edges(data=True)
        inventory_edges = [(u, v) for u, v, d in edges if isinstance(d.get('edge'), InventoryEdge)]
        service_edges = [(u, v) for u, v, d in edges if isinstance(d.get('edge'), ServiceEdge)]
        currency_edges = [(u, v) for u, v, d in edges if isinstance(d.get('edge'), CurrencyEdge)]
        
        # Draw edges with different styles and colors
        nx.draw_networkx_edges(self.graph, pos, edgelist=inventory_edges, edge_color='blue',
                             width=2, alpha=0.6, label='Inventory Flow', ax=ax)
        nx.draw_networkx_edges(self.graph, pos, edgelist=service_edges, edge_color='red',
                             width=2, style='dashed', alpha=0.6, label='Service Flow', ax=ax)
        nx.draw_networkx_edges(self.graph, pos, edgelist=currency_edges, edge_color='green',
                             width=2, style='dotted', alpha=0.6, label='Currency Flow', ax=ax)
        
        # Highlight path if provided
        if highlight_path:
            path_edges = list(zip(highlight_path[:-1], highlight_path[1:]))
            nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges,
                                 edge_color='yellow', width=4, alpha=0.5,
                                 label='Highlighted Path', ax=ax)
        
        # Add node labels with waste information
        labels = {}
        for node in self.graph.nodes():
            node_obj = self.nodes[node]
            waste_info = ""
            if hasattr(node_obj, 'waste_function'):
                if isinstance(node_obj.waste_function, StaticWaste):
                    waste_info = f"\nWaste: {node_obj.waste_function.fraction:.1%}"
                elif isinstance(node_obj.waste_function, TimeBasedWaste):
                    waste_info = f"\nBase: {node_obj.waste_function.base_rate:.1%}\nTime: +{node_obj.waste_function.time_coefficient:.1%}/h"
            
            if isinstance(node_obj, SolutionProvider):
                effects = [f"{k}: {v:.1%}" for k, v in node_obj.service_effects.items()]
                waste_info = f"\nEffects:\n" + "\n".join(effects)
            
            labels[node] = f"{node}{waste_info}"
        
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8, ax=ax)
        
        # Add edge labels with flow information
        edge_labels = {}
        for u, v, data in self.graph.edges(data=True):
            edge_obj = data.get('edge')
            if isinstance(edge_obj, InventoryEdge):
                if edge_obj.current_flow:
                    edge_labels[(u, v)] = f"Mass: {edge_obj.current_flow.mass:.1f}\nValue: {edge_obj.current_flow.value:.1f}"
            elif isinstance(edge_obj, ServiceEdge):
                edge_labels[(u, v)] = f"Service: {edge_obj.service_name}\nEffect: {edge_obj.effect_multiplier:.1%}"
            elif isinstance(edge_obj, CurrencyEdge):
                edge_labels[(u, v)] = f"{edge_obj.currency_type}\n{edge_obj.amount:.1f}"
        
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=6, ax=ax)
        
        # Add title and legend
        ax.set_title("Advanced Waste Network\nNode colors indicate type, edge styles show flow type",
                    pad=20, fontsize=14)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Add annotations for network statistics
        stats_text = (
            f"Network Statistics:\n"
            f"Nodes: {self.graph.number_of_nodes()}\n"
            f"Edges: {self.graph.number_of_edges()}\n"
            f"Producers: {len(producers)}\n"
            f"Processors: {len(processors)}\n"
            f"Handlers: {len(handlers)}\n"
            f"Consumers: {len(consumers)}\n"
            f"Service Providers: {len(providers)}"
        )
        ax.text(1.1, 0.95, stats_text, transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        # Adjust layout and display
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            return ax.get_figure()

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
