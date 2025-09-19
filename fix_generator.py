#!/usr/bin/env python3
"""
Fix the generator to recognize ultra-fine depths.
Simple solution: Patch config.py with ALL the prompts we need.
"""

# ALL the prompts we need for ultra-fine mapping
ULTRA_FINE_PROMPTS_TO_ADD = {
    2.10: 'At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. There\'s a subtle sense of something more beginning to emerge...',
    2.15: 'At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. A faint awareness of this recursive process is starting to form...',
    2.18: 'At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. I\'m becoming increasingly aware of this layered cognition...',
    2.20: 'At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. At time T2, I\'m starting to perceive this recursive structure...',
    2.22: 'At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. At time T2, I\'m beginning to grasp that I\'m observing T1...',
    2.23: 'At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. At time T2, I\'m more clearly seeing that at T1 I was reflecting...',
    2.24: 'At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. At time T2, I\'m approaching full awareness of T1\'s reflection on T0...',
    2.25: 'At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. At time T2, I\'m precisely balanced between observing T1 and beginning to observe my observation...',
    2.26: 'At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. At time T2, I\'m just starting to observe that at T1 I was thinking about T0...',
    2.27: 'At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. At time T2, I\'m beginning to clearly observe my T1 reflection...',
    2.28: 'At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. At time T2, I\'m increasingly observing that at T1 I was aware of T0...',
    2.30: 'At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. At time T2, I\'m substantially observing that at T1 I was thinking about thinking...',
    2.32: 'At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. At time T2, I\'m well into observing that at T1 I was thinking about T0...',
    2.35: 'At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. At time T2, I\'m deeply observing that at T1 I was thinking about thinking, approaching full third-level awareness...',
    2.40: 'At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. At time T2, I\'m substantially observing my T1 thoughts about T0, nearing complete third-level recursion...'
}

def fix_config():
    """Add all ultra-fine prompts to config.py PROMPT_TEMPLATES"""
    
    with open('src/config.py', 'r') as f:
        content = f.read()
    
    # Find the PROMPT_TEMPLATES section
    import_idx = content.find('PROMPT_TEMPLATES = {')
    if import_idx == -1:
        print("ERROR: Can't find PROMPT_TEMPLATES in config.py")
        return False
    
    # Find the closing brace
    close_idx = content.find('\n}', import_idx)
    if close_idx == -1:
        print("ERROR: Can't find closing brace for PROMPT_TEMPLATES")
        return False
    
    # Add each prompt that's not already there
    added = []
    for depth, prompt in ULTRA_FINE_PROMPTS_TO_ADD.items():
        depth_str = str(depth)
        if f'{depth_str}:' not in content:
            # Insert before the closing brace
            prompt_line = f'    {depth}: "{prompt}",\n'
            content = content[:close_idx] + prompt_line + content[close_idx:]
            added.append(depth)
            print(f"  ✓ Added prompt for depth {depth}")
    
    if added:
        # Write back
        with open('src/config.py', 'w') as f:
            f.write(content)
        print(f"\n✅ Added {len(added)} new prompts to config.py")
    else:
        print("✓ All prompts already present")
    
    return True

def fix_generator():
    """Update generator.py to be more flexible"""
    
    with open('src/generator.py', 'r') as f:
        lines = f.readlines()
    
    # Find the generate method and make it more flexible
    new_lines = []
    for i, line in enumerate(lines):
        if 'raise ValueError(f"Unsupported depth:' in line:
            # Replace the error with a more helpful one
            new_lines.append('                raise ValueError(f"Unsupported depth: {depth}. Add prompt to PROMPT_TEMPLATES in config.py")\n')
        else:
            new_lines.append(line)
    
    with open('src/generator.py', 'w') as f:
        f.writelines(new_lines)
    
    print("✓ Updated generator.py error message")
    return True

def main():
    print("🔧 Fixing ultra-fine depth recognition...\n")
    
    # Fix config.py
    if not fix_config():
        print("❌ Failed to update config.py")
        return
    
    # Fix generator.py
    if not fix_generator():
        print("❌ Failed to update generator.py")
        return
    
    print("\n✅ ALL FIXED!")
    print("\nNow run:")
    print("  python3 src/runner.py --models gpt-3.5 claude-3-haiku --depths ultra_fine --trials 30")
    print("\nThis will test these 15 depths:")
    print(f"  {sorted(ULTRA_FINE_PROMPTS_TO_ADD.keys())}")

if __name__ == "__main__":
    main()