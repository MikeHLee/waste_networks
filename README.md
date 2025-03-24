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

2. **Causal Analysis**
   - Bayesian network modeling using PyMC
   - Causal effect estimation with uncertainty quantification
   - Bootstrap-based standard error calculation
   - Support for both observed and latent variables
   - Visualization of causal networks with effect sizes

3. **Data Management**
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
pip install -r requirements.txt
```

## Usage

See `examples/toy_ecosystem.py` for a complete example of:
- Creating a waste network
- Adding nodes and edges with waste attributes
- Finding minimum waste paths
- Performing causal analysis
- Visualizing results

## Documentation

- [Whitepaper](whitepaper/main.tex): Technical details and methodology
- [API Documentation](docs/): Coming soon
- [Examples](examples/): Sample implementations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
