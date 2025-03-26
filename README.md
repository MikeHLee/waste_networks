# Waste Networks Analysis

A Python library for analyzing waste in supply chain networks using graph theory and causal analysis.

## Features

### Implemented Features 
1. **Waste Network Model**
   - Graph-based representation of supply chains using NetworkX
   - Node attributes for capacity and waste rates
   - Edge attributes for flow capacity and transportation waste
   - Minimum waste path calculation
   - Path-specific waste breakdown analysis
   - Network visualization with highlighted paths

2. **Advanced Network Model**
   - Multiple node types with specific constraints:
     - Solution Providers: Affect edge costs through services
     - End Consumers: Terminal nodes (no outgoing food transfers)
     - Initial Producers: Source nodes (no incoming food transfers)
     - Food Processors: Transform food composition
     - Food Handlers: Maintain food composition
   - Complex edge types:
     - Inventory transfers with mass, value, and composition
     - Service provision affecting edge properties
     - Currency flows with multiple denominations
   - Polymorphic waste functions:
     - Static waste rates
     - Time-dependent waste
     - Multi-variable waste (temperature, humidity, etc.)
   - Inventory management:
     - Input/output inventory tracking
     - Holding period management
     - Composition transformation rules

3. **Causal Analysis**
   - Bayesian network modeling using PyMC
   - Causal effect estimation with uncertainty quantification
   - Bootstrap-based standard error calculation
   - Support for both observed and latent variables
   - Visualization of causal networks with effect sizes

4. **Data Management**
   - JSON-based network configuration
   - Synthetic data generation for testing
   - Support for both static and time-series data

### Planned Features 
1. **Advanced Network Analysis**
   - Dynamic network evolution over time
   - Multi-commodity flow optimization
   - Stochastic waste modeling
   - Network resilience analysis

2. **Enhanced Causal Analysis**
   - Time-series causal discovery
   - Non-linear causal relationships
   - Counterfactual analysis
   - Intervention optimization

3. **Machine Learning Integration**
   - Predictive waste modeling
   - Anomaly detection
   - Automated parameter tuning
   - Real-time monitoring

4. **Visualization and Reporting**
   - Interactive network visualizations
   - Real-time monitoring dashboards
   - Automated report generation
   - Custom visualization templates

## Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Network
See `examples/toy_ecosystem.py` for a basic example of:
- Creating a simple waste network
- Adding nodes and edges with waste attributes
- Finding minimum waste paths
- Performing causal analysis
- Visualizing results

### Advanced Network
See `examples/advanced_ecosystem.py` for an advanced example demonstrating:
- Complex node type instantiation
- Service provider integration
- Inventory transformation
- Multi-variable waste functions
- Currency flow modeling

### Data Types

1. **Node Types**
```python
from advanced_network import (
    InitialProducer, FoodProcessor, FoodHandler,
    EndConsumer, SolutionProvider, StaticWaste,
    TimeBasedWaste, MultiVariableWaste
)

# Create different node types
producer = InitialProducer("farm")
producer.set_waste_function(TimeBasedWaste(0.02, 0.001))  # Base + time coefficient
producer.material_inputs = {"fertilizer": 100, "water": 500}

processor = FoodProcessor("processor")
processor.set_waste_function(StaticWaste(0.05))  # Fixed rate
processor.add_transformation_rule(
    "raw_vegetables",
    {"cleaned_vegetables": 0.9, "waste": 0.1}
)

handler = FoodHandler("warehouse")
# Complex waste function based on temperature and humidity
def temp_humidity_waste(temperature: float, humidity: float) -> float:
    return 0.01 + 0.001 * temperature + 0.0005 * humidity
handler.set_waste_function(MultiVariableWaste(temp_humidity_waste))

consumer = EndConsumer("store")
consumer.set_waste_function(StaticWaste(0.15))  # 15% consumer waste

provider = SolutionProvider("cold_chain")
provider.add_service_effect("temperature_control", 0.7)  # 30% waste reduction
```

2. **Edge Types**
```python
from advanced_network import InventoryEdge, ServiceEdge, CurrencyEdge, Inventory

# Create inventory transfer edge
inventory = Inventory(
    mass=1000.0,
    value=5000.0,
    composition={"vegetables": 0.95, "packaging": 0.05},
    currency_type="USD"
)
inventory_edge = InventoryEdge(capacity=1000)
inventory_edge.current_flow = inventory

# Create service edge
service_edge = ServiceEdge("temperature_control", 0.7)  # 30% improvement

# Create currency edge
currency_edge = CurrencyEdge("USD")
currency_edge.amount = 5000.0
currency_edge.is_owed = False  # Payment completed

# Add edges to network
network = AdvancedWasteNetwork()
network.add_edge("farm", "processor", inventory_edge)
network.add_edge("cold_chain", "processor", service_edge)
network.add_edge("processor", "farm", currency_edge)
```

3. **Inventory Management**
```python
# Define inventory with composition
raw_inventory = Inventory(
    mass=100.0,
    value=500.0,
    composition={"raw_vegetables": 1.0},
    currency_type="USD"
)

# Add transformation rules
processor = FoodProcessor("processor")
processor.add_transformation_rule(
    "raw_vegetables",
    {
        "cleaned_vegetables": 0.9,  # 90% yield
        "compost": 0.05,           # 5% to compost
        "waste": 0.05             # 5% waste
    }
)

# Process inventory
processor.input_inventory["batch_001"] = raw_inventory
processor.holding_periods["batch_001"] = 24.0  # hours

# Calculate waste
time = 24.0  # hours
temperature = 25.0  # Celsius
humidity = 60.0  # %RH

total_waste = processor.waste_function.calculate(
    time=time,
    temperature=temperature,
    humidity=humidity
)
```

4. **Network Analysis**
```python
# Create and analyze network
network = AdvancedWasteNetwork()

# Add nodes
network.add_node(producer)
network.add_node(processor)
network.add_node(handler)
network.add_node(consumer)
network.add_node(provider)

# Add edges
network.add_edge("farm", "processor", inventory_edge)
network.add_edge("cold_chain", "processor", service_edge)
network.add_edge("processor", "warehouse", inventory_edge2)
network.add_edge("warehouse", "store", inventory_edge3)
network.add_edge("store", "consumer", inventory_edge4)

# Calculate path waste
path = ["farm", "processor", "warehouse", "store", "consumer"]
total_waste, waste_breakdown = network.calculate_path_waste(
    path=path,
    time=24.0,  # hours
    temperature=25.0,  # Celsius
    humidity=60.0  # %RH
)

print(f"Total waste along path: {total_waste:.1%}")
for location, waste in waste_breakdown.items():
    print(f"{location}: {waste:.1%}")
```

## Advanced Features

### 1. Service Provider Effects
Service providers can modify waste rates through:
- Direct node effects (e.g., cold storage)
- Edge effects (e.g., transportation conditions)
- System-wide optimizations

### 2. Inventory Transformation
Food processors can:
- Transform input composition
- Track yield coefficients
- Manage by-products and waste

### 3. Waste Functions
Three types available:
- Static: Fixed percentage
- Time-based: Linear time dependency
- Multi-variable: Complex environmental effects

### 4. Currency Management
Track financial flows:
- Payment for goods
- Service fees
- Outstanding balances

## Documentation

- [Whitepaper](whitepaper/main.tex): Technical details and methodology
- [API Documentation](docs/): Coming soon
- [Examples](examples/): Sample implementations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
