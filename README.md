# Quantifying and Correcting Systematic Bias in Hardy-Littlewood Conjecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## 📄 Paper Information

**Title**: Quantifying and Correcting the Systematic Bias in the Hardy-Littlewood Conjecture at Intermediate Scales

**Author**: Ruqing Chen (GUT Geoservice Inc.)

**Contact**: ruqing@hotmail.com

**Status**: Preprint / Under Review

## 📊 Abstract

The Hardy-Littlewood conjecture provides an asymptotic formula for the number of Goldbach partitions $G(N)$. However, at intermediate scales ($N = 5 \times 10^5$ to $N = 10^8$), we observe a persistent negative systematic bias with high statistical significance ($R^2 = 0.9425$, $p < 0.001$).

**Key Findings**:
- Linear regression model: $E(N) = -0.5956 \ln(N) - 33.01$
- 94.3% of error variance explained by logarithmic drift
- Low-frequency correction achieves 25.66% accuracy improvement
- High-frequency resonance hypothesis (18.14 Hz) rejected (2.72% gain)

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.9
numpy >= 1.24
scipy >= 1.11
matplotlib >= 3.7
pandas >= 1.5
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Ruqing1963/Goldbach-Bias-Analysis.git
cd Goldbach-Bias-Analysis

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

```bash
# Run complete statistical analysis
python reproduce.py

# This will generate:
# - regression_statistics.json
# - complete_analysis_data.csv
# - regression_diagnostics.png
```

## 📁 Repository Structure

```
Goldbach-Bias-Analysis/
│
├── paper/
│   ├── main_FINAL_READY.tex              # LaTeX source
│   ├── main_FINAL_READY.pdf              # Compiled PDF
│   ├── Fig1_regression_diagnostics.png   # 4-in-1 diagnostics
│   └── Figure_2_Accuracy_Gain.png        # Accuracy comparison
│
├── data/
│   ├── Table_1_Global_Optimization.csv       # Frequency test results
│   ├── Table_2_Deep_Space_Data_EXTENDED.csv  # Complete analysis data
│   └── raw/                                   # Raw computation data
│
├── scripts/
│   ├── reproduce.py                      # Main analysis script
│   ├── compute_goldbach.py               # Partition counting
│   └── visualize.py                      # Generate figures
│
├── results/
│   ├── regression_statistics.json        # Statistical outputs
│   └── complete_analysis_data.csv        # Full dataset
│
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── LICENSE                                # MIT License
├── CITATION.cff                           # Citation information
└── .gitignore                             # Git ignore rules
```

## 📊 Data Description

### Table 1: Frequency Hypothesis Testing

| Hypothesis | Frequency (Hz) | Accuracy Gain | Verdict |
|------------|----------------|---------------|---------|
| High Frequency (Resonance) | 18.14 | 2.72% | Rejected |
| Riemann Zero Base | 14.13 | 3.30% | Weak Signal |
| **Systematic Bias Correction** | **0.1** | **25.66%** | **Confirmed** |

### Table 2: Hardy-Littlewood Error Analysis

Seven test points from $N = 500,000$ to $N = 100,000,000$:

| N | ln(N) | Observed Error (%) | Predicted Error (%) | Residual (%) |
|---|-------|-------------------|---------------------|--------------|
| 500,000 | 13.12 | -40.29 | -40.82 | 0.531 |
| 1,000,000 | 13.82 | -41.43 | -41.23 | -0.196 |
| 2,000,000 | 14.51 | -41.89 | -41.65 | -0.243 |
| 5,000,000 | 15.42 | -42.45 | -42.19 | -0.258 |
| 10,000,000 | 16.12 | -42.73 | -42.61 | -0.125 |
| 50,000,000 | 17.73 | -43.42 | -43.56 | 0.144 |
| 100,000,000 | 18.42 | -43.83 | -43.98 | 0.147 |

**Complete data available in**: `data/Table_2_Deep_Space_Data_EXTENDED.csv`

## 🔬 Methodology

### 1. Goldbach Partition Counting

For each even integer $N$, we compute $G_{observed}(N)$ as:

$$G_{observed}(N) = |\{(p, q) : p + q = N, \text{ both primes}, p \leq q\}|$$

**Algorithm**:
- Sieve of Eratosthenes for prime generation: $O(N \log \log N)$
- Partition enumeration: $O(\pi(N))$
- Total complexity: $O(N \log \log N)$

### 2. Hardy-Littlewood Prediction

$$G_{predicted}(N) \approx 2 C_2 \prod_{p | N, p > 2} \left( \frac{p-1}{p-2} \right) \int_{2}^{N} \frac{dx}{(\ln x)^2}$$

where $C_2 \approx 0.6601618158$ is the twin prime constant.

### 3. Statistical Analysis

Linear regression model:
$$E(N) = \alpha \ln(N) + \beta + \epsilon$$

**Results**:
- Slope $\alpha = -0.5956$ (%/ln(N))
- Intercept $\beta = -33.01$ (%)
- $R^2 = 0.9425$
- p-value $< 0.001$
- Shapiro-Wilk test: $p = 0.146$ (residuals normally distributed)

## 📈 Key Results

### Regression Equation

```
E(N) = -0.5956 × ln(N) - 33.01
```

### Statistical Significance

- **R² = 0.9425**: 94.3% of variance explained
- **p < 0.001**: Highly significant
- **95% CI for slope**: [-0.7646, -0.4265]

### Predictive Accuracy

- At $N = 10^8$: Predicted -43.98%, Observed -43.83% (error: 0.15%)
- Mean absolute residual: 0.25%
- Max residual: 0.53%

### Extrapolations

| N | Predicted Error |
|---|-----------------|
| $10^9$ | -45.35% |
| $10^{10}$ | -46.72% |
| $10^{11}$ | -48.09% |

## 🖼️ Figures

### Figure 1: Regression Diagnostics

![Regression Diagnostics](paper/Fig1_regression_diagnostics.png)

4-in-1 diagnostic plot showing:
- Linear fit ($R^2 = 0.9425$)
- Residual distribution
- Q-Q normality test
- Residual histogram

### Figure 2: Accuracy Improvement

![Accuracy Comparison](paper/Figure_2_Accuracy_Gain.png)

Comparison of three correction strategies demonstrating the superiority of low-frequency systematic bias correction (25.66% improvement).

## 🔄 Reproducibility

All analyses are fully reproducible:

```bash
# Step 1: Generate Goldbach partition counts
python scripts/compute_goldbach.py --n_max 100000000

# Step 2: Run statistical analysis
python scripts/reproduce.py

# Step 3: Generate figures
python scripts/visualize.py
```

**Computational Environment**:
- CPU: AMD64 Family Processor
- RAM: 16 GB
- OS: Windows 11
- Software: Python 3.9 with NumPy, SciPy, Matplotlib
- Total computation time: ~2-3 hours for all test points

## 📖 Citation

If you use this work, please cite:

```bibtex
@article{chen2025goldbach,
  title={Quantifying and Correcting the Systematic Bias in the Hardy-Littlewood Conjecture at Intermediate Scales},
  author={Chen, Ruqing},
  journal={Preprint},
  year={2025},
  institution={GUT Geoservice Inc.},
  url={https://github.com/Ruqing1963/Goldbach-Bias-Analysis}
}
```

Or use the `CITATION.cff` file in this repository.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Areas for Contribution

1. **Extended Analysis**: Compute partition counts for $N > 10^9$
2. **Theoretical Development**: Derive sub-leading terms analytically
3. **Alternative Methods**: Test other correction strategies
4. **Code Optimization**: Improve computational efficiency

## 📧 Contact

**Ruqing Chen**
- Email: ruqing@hotmail.com
- Institution: GUT Geoservice Inc.
- GitHub: [@Ruqing1963](https://github.com/Ruqing1963)

## 🙏 Acknowledgments

- Hardy-Littlewood circle method foundation
- Open-source scientific Python ecosystem
- Computational number theory community

## 📚 References

Key references (see paper for complete list):

1. Hardy & Littlewood (1923). Some problems of 'Partitio numerorum'
2. Cramér (1936). On prime number gaps
3. Riemann (1859). Distribution of primes
4. Zhang (2014). Bounded gaps between primes
5. Richstein (2001). Goldbach verification to $4 \times 10^{14}$

## 🔗 Related Resources

- [Goldbach Conjecture (Wikipedia)](https://en.wikipedia.org/wiki/Goldbach%27s_conjecture)
- [Hardy-Littlewood Method (MathWorld)](https://mathworld.wolfram.com/Hardy-LittlewoodMethod.html)
- [Prime Number Theorem](https://en.wikipedia.org/wiki/Prime_number_theorem)

---

**Last Updated**: December 31, 2025

**Repository**: [https://github.com/Ruqing1963/Goldbach-Bias-Analysis](https://github.com/Ruqing1963/Goldbach-Bias-Analysis)
