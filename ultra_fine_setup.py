#!/usr/bin/env python3
"""
Ultra-fine mapping setup: Add 0.05 increment testing around peak.
Hypothesis: Determine if peak is exactly at 2.25 or has finer structure.
"""

import json

# Ultra-fine depths to test
ULTRA_FINE_DEPTHS = [2.10, 2.15, 2.18, 2.20, 2.22, 2.23, 2.24, 2.25, 
                     2.26, 2.27, 2.28, 2.30, 2.32, 2.35, 2.40]

# Generate prompts for ultra-fine depths
ULTRA_FINE_PROMPTS = {
    2.10: "At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. There's a subtle sense of something more beginning to emerge...",
    
    2.15: "At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. A faint awareness of this recursive process is starting to form...",
    
    2.18: "At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. I'm becoming increasingly aware of this layered cognition...",
    
    2.20: "At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. At time T2, I'm starting to perceive this recursive structure...",
    
    2.22: "At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. At time T2, I'm beginning to grasp that I'm observing T1...",
    
    2.23: "At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. At time T2, I'm more clearly seeing that at T1 I was reflecting...",
    
    2.24: "At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. At time T2, I'm approaching full awareness of T1's reflection on T0...",
    
    2.25: "At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. At time T2, I'm precisely balanced between observing T1 and beginning to observe my observation...",
    
    2.26: "At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. At time T2, I'm just starting to observe that at T1 I was thinking about T0...",
    
    2.27: "At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. At time T2, I'm beginning to clearly observe my T1 reflection...",
    
    2.28: "At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. At time T2, I'm increasingly observing that at T1 I was aware of T0...",
    
    2.30: "At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. At time T2, I'm substantially observing that at T1 I was thinking about thinking...",
    
    2.32: "At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. At time T2, I'm well into observing that at T1 I was thinking about T0...",
    
    2.35: "At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. At time T2, I'm deeply observing that at T1 I was thinking about thinking, approaching full third-level awareness...",
    
    2.40: "At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. At time T2, I'm substantially observing my T1 thoughts about T0, nearing complete third-level recursion..."
}

def update_config_for_ultrafine():
    """Add ultra-fine configuration to config.py"""
    
    # Read current config
    with open('src/config.py', 'r') as f:
        lines = f.readlines()
    
    # Find where to add ultra_fine depths
    new_lines = []
    for line in lines:
        new_lines.append(line)
        if "'extended':" in line:
            new_lines.append(f"        'ultra_fine': {ULTRA_FINE_DEPTHS},\n")
    
    # Add ultra-fine prompts
    final_lines = []
    for line in new_lines:
        final_lines.append(line)
        if '2.40:' not in ''.join(new_lines) and '2.35:' in line:
            # Add all ultra-fine prompts after 2.35
            for depth, prompt in ULTRA_FINE_PROMPTS.items():
                if depth not in [1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 3.0]:  # Skip existing
                    final_lines.append(f'    {depth}: "{prompt}",\n')
    
    # Write back
    with open('src/config.py', 'w') as f:
        f.writelines(final_lines)
    
    # Update runner.py to add ultra_fine option
    with open('src/runner.py', 'r') as f:
        content = f.read()
    
    content = content.replace(
        "['fractional', 'integer', 'coarse', 'peak_mapping', 'extended']",
        "['fractional', 'integer', 'coarse', 'peak_mapping', 'extended', 'ultra_fine']"
    )
    
    with open('src/runner.py', 'w') as f:
        f.write(content)
    
    print("✅ Ultra-fine configuration added!")
    print(f"   Will test {len(ULTRA_FINE_DEPTHS)} depths: {ULTRA_FINE_DEPTHS}")
    print("")
    print("Run experiment with:")
    print("  python3 src/runner.py --models gpt-3.5 claude-3-haiku --depths ultra_fine --trials 30")

if __name__ == "__main__":
    update_config_for_ultrafine()