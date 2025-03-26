"""
advanced_network.py: Advanced network modeling functionality with support for
complex node types, inventory management, and multi-dimensional edge properties.
"""

from enum import Enum
from typing import Dict, List, Tuple, Any, Union, Callable
import networkx as nx
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

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
    
    def calculate_path_waste(self, path: List[str], time: float = 24.0) -> Tuple[float, Dict[str, float]]:
        """Calculate total waste along a path."""
        total_waste = 0.0
        waste_breakdown = {}
        
        # Calculate node waste
        for node in path:
            node_waste = self.nodes[node].waste_function.calculate(time=time)
            total_waste += node_waste
            waste_breakdown[f"node_{node}"] = node_waste
            
        # Calculate edge waste
        for i in range(len(path)-1):
            source, target = path[i], path[i+1]
            for _, _, edge_data in self.graph.edges(data=True):
                if isinstance(edge_data['edge'], InventoryEdge):
                    edge_waste = edge_data['edge'].waste_function.calculate(time=time)
                    total_waste += edge_waste
                    waste_breakdown[f"transport_{source}_{target}"] = edge_waste
                    
        return total_waste, waste_breakdown
