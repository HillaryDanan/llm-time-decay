# LLM Time Decay - Command Reference 🚀

## Initial Setup (One Time)

```bash
# Navigate to project
cd ~/Desktop/llm-time-decay

# Run setup script
bash setup.sh

# Edit .env with your API keys
nano .env  # or use your favorite editor
```

## Before Each Session

```bash
# Activate virtual environment
source venv/bin/activate
```

## Test Everything Works

```bash
# Verify installation
python3 test_setup.py

# Quick test (2 trials only)
python3 src/runner.py --models gpt-3.5 --depths fractional --trials 2
```

## Run Main Experiment

```bash
# Full experiment (all available models, 50 trials each)
python3 src/runner.py --models all --depths fractional --trials 50

# Just GPT
python3 src/runner.py --models gpt-3.5 --depths fractional --trials 50

# Just Claude
python3 src/runner.py --models claude-3-haiku --depths fractional --trials 50

# Just Gemini
python3 src/runner.py --models gemini-1.5-flash --depths fractional --trials 50

# Test two specific models
python3 src/runner.py --models gpt-3.5 gemini-1.5-flash --depths fractional --trials 50

# Test integer depths for comparison
python3 src/runner.py --models all --depths integer --trials 20

# Quiet mode (no progress output)
python3 src/runner.py --models all --depths fractional --trials 50 --quiet
```

## Analyze Results

```bash
# Find your latest results file
ls -la data/processed/

# Run analysis (replace with your timestamp)
python3 src/analyzer.py --results data/processed/results_20250919_120000.json --plot --report

# Just generate plot
python3 src/analyzer.py --results data/processed/results_[TIMESTAMP].json --plot

# Just generate report
python3 src/analyzer.py --results data/processed/results_[TIMESTAMP].json --report
```

## Git Commands

```bash
# Check status
git status

# Add all changes
git add .

# Commit with message
git commit -m "Run fractional depth experiment n=50"

# Push to GitHub
git push origin main

# Quick commit and push (GPOM!)
git add . && git commit -m "Update results" && git push origin main
```

## Project Structure

```
llm-time-decay/
├── src/               # All Python code
├── data/             
│   ├── raw/          # Raw API responses (gitignored)
│   └── processed/    # Analyzed results (gitignored)
├── results/          
│   ├── figures/      # Plots (saved in git)
│   └── tables/       # Reports (gitignored)
└── test_setup.py     # Verify everything works
```

## Troubleshooting

```bash
# If imports fail
pip install -r requirements.txt

# If API keys not found
cp .env.example .env
# Then edit .env

# If permission denied on setup.sh
chmod +x setup.sh

# Check Python version (need 3.8+)
python3 --version
```

## Data Files

- Raw responses: `data/raw/raw_[TIMESTAMP].json`
- Processed results: `data/processed/results_[TIMESTAMP].json`
- Figures: `results/figures/decay_curves.png`
- Reports: `results/tables/analysis_report.txt`

## Cost Estimates

- GPT-3.5: ~$0.001 per trial
- Claude-3-haiku: ~$0.0005 per trial  
- Gemini-1.5-flash: ~$0.0001 per trial (basically free!)
- Full experiment (1500 trials with all 3): ~$1.00
- Just GPT + Gemini (1000 trials): ~$0.60

## Quick Analysis in Python

```python
# Load and inspect results
import json

with open('data/processed/results_[TIMESTAMP].json', 'r') as f:
    results = json.load(f)

# Check means by depth for GPT
for depth, trials in results['by_depth'].items():
    gpt_scores = [t['scores']['total'] for t in trials if t['model'] == 'gpt-3.5']
    if gpt_scores:
        print(f"Depth {depth}: {sum(gpt_scores)/len(gpt_scores):.3f}")
```

---

**Remember:** We're testing if GPT uses continuous decay while Claude uses discrete stages! 🔬