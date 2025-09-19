# LLM Time Decay: The 2.25 Recursion Plateau Discovery

## 🔬 Major Finding: Peak at Fractional Depth 2.25

**UPDATE (Sept 19, 2025):** Fine-grained mapping reveals the true peak is at depth ~2.25, not 2.0!

### Key Discoveries:
- **Initial observation**: Coarse sampling suggested peak at depth 2.0
- **Fine mapping**: Revealed actual peak at 2.23 (GPT) and 2.26 (Claude)  
- **Plateau structure**: High coherence maintained across depths 2.0-2.3
- **Model fits**: Gaussian+decay (R² > 0.80) beats exponential (R² < 0.62)

This suggests LLMs process recursion along a **continuum** rather than discrete levels.

## 🚀 Current Status

### Completed Experiments:
- ✅ Initial fractional depth testing (n=1500)
- ✅ Peak mapping [1.6-2.4] in 0.1 increments (n=540)
- ✅ Statistical confirmation of Gaussian+decay model

### In Progress:
- 🔄 Ultra-fine mapping around 2.25 (0.05 increments)
- 🔄 Testing additional models (GPT-4, Claude-opus)

## 📊 Repository Structure

```
llm-time-decay/
├── src/
│   ├── config.py           # Experiment configurations
│   ├── generator.py        # Prompt generation with fractional depths
│   ├── scorer.py           # Coherence scoring system
│   ├── runner.py           # Main experiment runner
│   └── analyzer.py         # Statistical analysis
├── data/
│   ├── raw/                # Raw API responses
│   └── processed/          # Analyzed results
├── results/
│   ├── figures/            # Visualizations
│   │   ├── decay_curves.png
│   │   ├── depth2_anomaly.png
│   │   └── peak_mapping_*.png
│   └── tables/             # Statistical outputs
├── peak_mapping_experiment.py  # Peak analysis tools
├── paper_outline.md        # Draft paper structure
└── README.md               # This file
```

## 🧪 Replication Instructions

### Setup
```bash
pip install -r requirements.txt
cp .env.example .env
# Add API keys to .env
```

### Run Peak Mapping
```bash
# Fine mapping around 2.25
python3 src/runner.py --models gpt-3.5 claude-3-haiku --depths peak_mapping --trials 30

# Analyze results
python3 peak_mapping_experiment.py --data data/processed/results_*.json --plot
```

## 📈 Key Results

| Model | Peak Location | Peak Width (σ) | Gaussian R² | Exponential R² |
|-------|--------------|----------------|-------------|----------------|
| GPT-3.5 | 2.23 ± 0.05 | 0.38 | 0.898 | 0.619 |
| Claude-3-haiku | 2.26 ± 0.05 | 0.25 | 0.796 | 0.609 |

## 🔗 Related Work

- Original temporal coherence study: [temporal-coherence-llm](https://github.com/HillaryDanan/temporal-coherence-llm)
- Discovered architectural differences between GPT and Claude
- This work extends to find universal fractional-depth optimum

## 📝 Citation

```bibtex
@article{danan2025fractional,
  title={The 2.25 Recursion Plateau: Fractional Optimal Depth in LLM Metacognition},
  author={Danan, Hillary and Claude},
  year={2025},
  journal={arXiv preprint},
  note={In preparation}
}
```

## 🤝 Collaboration

Human-AI research partnership:
- **Hillary Danan**: Experimental design, execution, analysis
- **Claude**: Hypothesis refinement, statistical framework, interpretation

---

*"The best discoveries are the ones that violate your assumptions"*

## 🚀 Current Mission

Testing fine-grained temporal depths (0.5 increments) to determine if:
1. GPT uses continuous attention decay
2. Claude uses discrete processing stages
3. Why the fuck they're so different

## 📊 Repository Structure

```
llm-time-decay/
├── src/
│   ├── config.py           # API keys, model configs
│   ├── generator.py        # Fractional depth prompts
│   ├── scorer.py           # Enhanced scoring system
│   ├── runner.py           # Main experiment runner
│   └── analyzer.py         # Statistical analysis
├── data/
│   ├── raw/                # Raw API responses
│   └── processed/          # Scored results
├── results/
│   ├── figures/            # Visualizations
│   └── tables/             # Statistical outputs
├── tests/
│   └── validation.py       # Quality control
├── requirements.txt
└── README.md
```

## ⚡ Quick Start

```bash
# Setup
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env

# Run experiment
python3 src/runner.py --depths fractional --models all --trials 50

# Analyze results
python3 src/analyzer.py --hypothesis discrete-vs-continuous
```

## 🧪 Experimental Design

**Testing Matrix:**
- Depths: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
- Models: GPT-3.5-turbo, Claude-3-haiku, Gemini-1.5-flash
- Trials: 50 per condition (n=1500 total with all 3 models)
- Metrics: Temporal ordering, causal maintenance, depth accuracy, transition smoothness

## 📈 Expected Outcomes

| Hypothesis | GPT Pattern | Claude Pattern | Gemini Pattern | Interpretation |
|------------|-------------|----------------|----------------|----------------|
| Continuous | Smooth exponential | Smooth exponential | Smooth exponential | Original = sampling artifact |
| Discrete | Step functions | Step functions | Step functions | All LLMs quantize time |
| Divergent | Smooth exponential | Step functions | ??? | ARCHITECTURAL DIFFERENCE |

## 🔗 Related Work

Building on: [temporal-coherence-llm](https://github.com/HillaryDanan/temporal-coherence-llm)

## 📝 Citation

```bibtex
@article{danan2025architectural,
  title={Fine-Grained Analysis of Architectural Divergence in LLM Temporal Processing},
  author={Danan, Hillary and Claude},
  year={2025}
}
```

## 🤝 Collaboration

Human-AI research partnership:
- **Hillary**: Experimental design, execution, analysis
- **Claude**: Hypothesis development, statistical framework, implementation

---

*"The best discoveries are the ones you didn't expect to make"*