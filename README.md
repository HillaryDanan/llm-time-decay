# LLM Time Decay: Methodological Lessons in Recursive Prompt Evaluation

## Important Finding: Extreme Prompt Sensitivity in LLM Recursion

**UPDATE (September 19, 2025):** After extensive testing with multiple sampling resolutions, we have discovered that apparent "optimal depths" are highly sensitive to prompt wording and measurement methodology.

### Summary of Findings:

#### Confirmed Results:
- **Non-monotonic patterns exist** - LLMs do not show simple exponential decay
- **2.0-2.3 range shows elevated performance** - General tendency across models
- **High response variability** - Standard deviations up to 30% of means

#### Methodological Challenges:
- **Peak location unstable**: Shifted from 2.0 to 2.25 to 2.30 with different measurements
- **Prompt wording effects**: Single word changes can move apparent peaks
- **Fine sampling reveals noise**: Higher resolution did not clarify patterns but added variability

### Key Methodological Insights:

1. **Resolution Paradox**: 
   - Coarse sampling (Δ=0.5): Clean patterns, possibly misleading
   - Fine sampling (Δ=0.1): Multiple peaks emerge
   - Ultra-fine (Δ=0.05): Extreme variability dominates

2. **Prompt Sensitivity**:
   - Word choice significantly affects results at high recursion depths
   - Terms such as "substantially" versus "starting to" can shift coherence by over 20%
   - Single prompt formulation insufficient for robust conclusions

3. **Statistical Considerations**:
   - N=30-50 per condition insufficient given variance
   - Confidence intervals overlap extensively
   - Apparent peaks may represent statistical noise

## Repository Structure

```
llm-time-decay/
├── src/                    # Experimental framework
├── data/                   # Results showing high variability
├── results/                # Visualizations of unstable patterns
├── paper_outline.md        # Assessment of findings
└── README.md              # This document
```

## Research Contributions

### Methodological Insights:
- Single prompt formulations cannot establish LLM cognitive properties
- Fine-grained measurement may amplify noise rather than reveal underlying patterns
- Apparent patterns may be measurement artifacts

### Valid Findings:
- Non-monotonic patterns in recursive processing (robust)
- General elevation in 2-3 recursion range (suggestive)
- High variability in LLM responses to complex prompts (definitive)

## Future Research Directions

### Recommended Approaches:

1. **Multi-Prompt Validation**
   - Test each depth with 10 or more different phrasings
   - Report variance across formulations
   - Only claim patterns robust to wording changes

2. **Mechanism-Based Investigation**
   - Analyze attention patterns directly
   - Probe internal representations
   - Move beyond behavioral measurements

3. **Statistical Power**
   - Minimum N>100 per condition
   - Pre-registered analysis plans
   - Multiple testing corrections

### Refined Research Questions:

Instead of "What is the optimal recursion depth?", consider:
- Why are LLMs sensitive to prompt variations at high recursion depths?
- What makes 2-3 level recursion generally more stable?
- How do architectural differences affect prompt sensitivity?

## Replication Notes

**Important**: Results are highly sensitive to:
- Exact prompt wording
- Scoring methodology  
- Sampling resolution
- Random seed and trial variance

Replication attempts should:
1. Test multiple prompt formulations
2. Use large N (>100 per condition)
3. Report full variance, not just means
4. Check robustness across models

## Lessons Learned

This project demonstrates the importance of:
- **Methodological rigor** - Initial findings often require deeper scrutiny
- **Transparent reporting** - Publishing mixed results prevents research waste
- **Prompt engineering awareness** - Apparent cognitive discoveries may be prompt artifacts

## Citation

If referencing this work, please note the methodological limitations:

```bibtex
@article{danan2025prompt,
  title={Prompt Sensitivity in LLM Recursive Processing: A Methodological Investigation},
  author={Danan, Hillary and Claude},
  year={2025},
  note={Demonstrates high prompt sensitivity in recursive coherence measurements. 
        Results should be interpreted with caution.},
  url={https://github.com/HillaryDanan/llm-time-decay}
}
```

## Data Interpretation Notice

The data in this repository shows:
- High variability between trials
- Sensitivity to prompt formulation
- Unstable peak locations

**These results should not be interpreted as establishing fundamental LLM properties without further validation using robust multi-prompt methodologies.**

---

*"In science, a negative result that reveals methodological challenges is as valuable as a positive finding."*

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