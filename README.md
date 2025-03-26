# Waste Networks Analysis

A comprehensive library for analyzing waste in supply chain networks using advanced network analysis and causal inference.

## Features

- Advanced network model with flow-specific path finding
- Bayesian causal analysis with regression capabilities
- Visualization tools for network structure and statistical analysis
- Inventory loss prediction and analysis

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Network Analysis

```python
from src.advanced_network import AdvancedWasteNetwork, AdvancedNode
from src.causal_analysis import WasteCausalNetwork

# Create network
network = AdvancedWasteNetwork()

# Add nodes and edges
farm = AdvancedNode("farm", node_type="producer")
warehouse = AdvancedNode("warehouse", node_type="storage")
network.add_node(farm)
network.add_node(warehouse)
network.add_edge(farm, warehouse, edge_type="inventory")
```

### Causal Analysis and Regression

```python
# Initialize causal network
causal_net = WasteCausalNetwork()

# Add variables
causal_net.add_node('storage_time', 'cause', 'normal')
causal_net.add_node('temperature', 'cause', 'normal')
causal_net.add_node('loss_percentage', 'effect', 'normal')

# Add relationships
causal_net.add_edge('storage_time', 'loss_percentage', 0.5)
causal_net.add_edge('temperature', 'loss_percentage', 0.3)

# Fit regression model
result = causal_net.fit_regression(
    target='loss_percentage',
    features=['storage_time', 'temperature']
)
```

## Understanding the Output

### Regression Results (regression_results.txt)

The regression analysis output contains:
1. Model Summary:
   - Number of observations
   - Number of features
   - R² score (measure of model fit, 0-1)
2. Coefficients Table:
   - Parameter names
   - Mean (effect size)
   - Standard deviation (uncertainty)
3. Sample Predictions:
   - Example scenarios with predicted loss percentages

### Visualizations

#### 1. Loss vs Storage Time (loss_vs_storage.png)
- X-axis: Storage time in days
- Y-axis: Loss percentage
- Blue dots: Actual data points
- Red line: Model predictions
- Shaded area: 95% prediction interval
- Text box: Effect size and confidence interval
- R² score in title

#### 2. Loss vs Temperature (loss_vs_temperature.png)
- X-axis: Temperature in °C
- Y-axis: Loss percentage
- Similar elements to storage time plot
- Shows temperature's impact on loss

#### 3. Causal Network Graph (causal_graph.png)
- Node Colors:
  * Light blue: Cause variables
  * Light green: Effect variables
- Node Labels:
  * Variable name
  * β: Coefficient (effect size)
  * σ: Standard deviation
- Edges:
  * Color: Red (strong effect) to Blue (weak effect)
  * Width: Proportional to effect size
  * Labels: β values showing relationship strength
- Legend: Explains node types and edge meanings
- R² score: Overall model fit

### Interpreting Effect Sizes

- β > 0: Positive relationship (increases cause increases in the effect)
- β < 0: Negative relationship (increases cause decreases)
- |β| ≈ 0: Weak or no relationship
- Standard deviation (σ) indicates uncertainty:
  * Small σ: More confident in the effect
  * Large σ: Less confident

## Examples

See the `examples/` directory for detailed examples:
- `advanced_regression.py`: Demonstrates inventory loss analysis
- More examples coming soon...

## Documentation

For detailed mathematical formulation and methodology, see the whitepaper in the `whitepaper/` directory.

## Contributing

Contributions are welcome! Please read our contributing guidelines and code of conduct.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
