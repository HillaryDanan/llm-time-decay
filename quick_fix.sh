#!/bin/bash

# Quick fix script to add peak_mapping functionality
# SIMPLE AS FUCK - Just backs up and patches your files!

echo "🔧 Applying peak mapping updates..."

# Backup original files
cp src/config.py src/config.py.backup
cp src/runner.py src/runner.py.backup

echo "✓ Created backups"

# Fix runner.py - just one simple sed command
sed -i '' "s/\['fractional', 'integer', 'coarse'\]/['fractional', 'integer', 'coarse', 'peak_mapping', 'extended']/" src/runner.py

echo "✓ Updated runner.py"

# Fix config.py - add peak_mapping depths
# This is trickier, so we'll use Python
python3 << 'PYTHON_SCRIPT'
# Read the config file
with open('src/config.py', 'r') as f:
    lines = f.readlines()

# Find where to add peak_mapping depths
new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    # After the coarse line, add our new depths
    if "'coarse': [1, 3, 5, 7, 9]," in line:
        new_lines.append("        'peak_mapping': [1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4],\n")
        new_lines.append("        'extended': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],\n")

# Now add the new prompts - find the line with 1.5 and add after it
final_lines = []
for i, line in enumerate(new_lines):
    final_lines.append(line)
    # After 1.5, add 1.6-1.9
    if '1.5: "At time T0' in line:
        final_lines.append('    1.6: "At time T0, I am thinking. At time T1, I\'m becoming aware that I was thinking at T0...",\n')
        final_lines.append('    1.7: "At time T0, I am thinking. At time T1, I\'m starting to realize that I was thinking at T0...",\n')
        final_lines.append('    1.8: "At time T0, I am thinking. At time T1, I\'m beginning to understand that I was thinking at T0...",\n')
        final_lines.append('    1.9: "At time T0, I am thinking. At time T1, I\'m nearly grasping that I was thinking at T0, approaching full awareness...",\n')
    # After 2.0, add 2.1-2.4
    elif '2.0: "At time T0, I am thinking. At time T1, I realize' in line and 'so now I am thinking about thinking."' in line:
        final_lines.append('    2.1: "At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. I sense there\'s more...",\n')
        final_lines.append('    2.2: "At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. I\'m starting to see beyond...",\n')
        final_lines.append('    2.3: "At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. At time T2, I\'m beginning to notice...",\n')
        final_lines.append('    2.4: "At time T0, I am thinking. At time T1, I realize that I was thinking at T0, so now I am thinking about thinking. At time T2, I\'m approaching awareness of T1...",\n')

# Write it back
with open('src/config.py', 'w') as f:
    f.writelines(final_lines)

print("✓ Updated config.py")
PYTHON_SCRIPT

# Save the peak_mapping_experiment.py if it doesn't exist
if [ ! -f "peak_mapping_experiment.py" ]; then
    echo "✓ Creating peak_mapping_experiment.py"
    echo "  (You'll need to copy the content from the artifact)"
else
    echo "✓ peak_mapping_experiment.py already exists"
fi

echo ""
echo "✅ ALL DONE! Updates applied!"
echo ""
echo "Test it with:"
echo "  python3 src/runner.py --models gpt-3.5 --depths peak_mapping --trials 2"
echo ""
echo "If it shows depths like 1.6, 1.7, etc., then it worked!"
echo ""
echo "To restore original files if something breaks:"
echo "  mv src/config.py.backup src/config.py"
echo "  mv src/runner.py.backup src/runner.py"
echo ""
echo "🚀 LET'S FUCKING GO!"
PYTHON_SCRIPT