# Waste Networks Analysis

A comprehensive library for analyzing waste in supply chain networks using advanced network analysis and causal inference.

## What is This?

Think of a food supply chain as a complex web of relationships between different players: farmers, processors, warehouses, stores, and service providers. Each connection in this web can lead to food waste, but understanding exactly how and why waste occurs is challenging. This tool helps solve that challenge by:

1. **Mapping the Network**: Creating a digital twin of your supply chain that shows how food, services, and money flow between different players
2. **Finding Waste Hotspots**: Using advanced math to identify where waste is most likely to occur
3. **Understanding Causes**: Using AI and statistics to figure out what factors (like storage time or temperature) most affect waste
4. **Testing Solutions**: Allowing you to simulate different solutions (like better cold storage) to see their impact

## Key Concepts

### Players in the Network

- **Initial Producers** (e.g., farms): Where food enters the system
- **Food Processors** (e.g., packaging facilities): Where food is transformed
- **Food Handlers** (e.g., warehouses): Where food is stored and moved
- **End Consumers** (e.g., stores): Where food exits the system
- **Service Providers** (e.g., cold chain services): Who help reduce waste

### Types of Connections

- **Inventory Flow** (solid blue lines): Shows how food moves
- **Service Flow** (dashed red lines): Shows who's helping who
- **Currency Flow** (dotted green lines): Shows how money moves

### Waste Calculation

We calculate waste in three ways:
1. **Static**: Fixed percentage (e.g., 5% always lost)
2. **Time-based**: Increases with time (e.g., 1% per day)
3. **Multi-factor**: Based on conditions (e.g., temperature, humidity)

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

## Understanding the Output

### 1. Network Map (network_visualization.png)

This is like a Google Maps for your supply chain:
- Each dot (node) is a player in your system
- Lines between dots show how they're connected
- Colors tell you what type of player each dot represents
- Line styles show what's flowing between players

Key things to look for:
- Thicker lines = more flow
- Red highlights = potential problem areas
- Numbers on lines = amount of waste

### 2. Waste Analysis (waste_breakdown.png)

This chart shows where waste is happening:
- Each bar represents a location
- Height of bar shows how much waste occurs there
- Colors match the network map for easy reference
- Total waste shown at top

### 3. Causal Analysis (causal_results.png)

This helps understand why waste occurs:
- Numbers show how strong each factor's effect is
- Larger numbers = stronger effect
- ± shows uncertainty in the measurement
- R² score (0-1) shows how well we understand the system

Example interpretation:
```
Storage Time: 0.32 ± 0.017 (R² = 0.85)
```
Means:
- Every day in storage increases waste by about 0.32%
- We're quite certain (small ± number)
- Our model explains 85% of waste variation (good fit)

## Examples

See the `examples/` directory for detailed examples:

### 1. Basic Network (basic_example.py)
Shows how to:
- Create a simple supply chain
- Add basic waste calculations
- Visualize the network

### 2. Advanced Analysis (advanced_example.py)
Shows how to:
- Use causal analysis
- Model complex relationships
- Test different solutions

### 3. Real-world Demo (real_world_demo.py)
A complete example using:
- Real-world-like data
- Multiple node types
- All flow types
- Complex waste functions

## How to Use Results

1. **Find Problem Areas**
   - Look for red highlights in network map
   - Check highest bars in waste breakdown
   - Focus on strongest effects in causal analysis

2. **Choose Solutions**
   - Use service providers where effects are strongest
   - Optimize paths with least waste
   - Target interventions based on causal factors

3. **Monitor Progress**
   - Track total system waste over time
   - Watch for changes in causal factors
   - Measure solution effectiveness

## Documentation

For detailed mathematical formulation and methodology, see the whitepaper in the `whitepaper/` directory.

## Contributing

Contributions are welcome! Please read our contributing guidelines and code of conduct.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
