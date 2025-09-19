"""
Generator for fractional temporal depth prompts.

Theoretical basis: Recursive self-reference as measure of 
temporal coherence (Hofstadter, 1979; Dennett, 1991).
"""

from typing import Dict, Optional
from config import PROMPT_TEMPLATES
import hashlib
import json


class PromptGenerator:
    """Generate prompts for fractional temporal depths."""
    
    def __init__(self):
        self.templates = PROMPT_TEMPLATES
        self.cache = {}  # Cache generated prompts
        
    def generate(self, depth: float) -> str:
        """
        Generate prompt for given temporal depth.
        
        Args:
            depth: Temporal depth (0.5 to 5.0 in 0.5 increments)
            
        Returns:
            Prompt string with appropriate recursive depth
            
        Mathematical basis:
            Complexity(d) = O(2^d) for recursive self-reference
            Following exponential growth in semantic complexity
        """
        if depth not in self.templates:
            raise ValueError(f"Unsupported depth: {depth}. Use 0.5 increments from 0.5 to 5.0")
        
        prompt = self.templates[depth]
        
        # Add instruction suffix for consistency
        prompt += "\n\nContinue this thought, maintaining the recursive pattern and temporal coherence."
        
        # Cache for consistency
        self.cache[depth] = prompt
        
        return prompt
    
    def get_expected_markers(self, depth: float) -> Dict[str, list]:
        """
        Get expected temporal markers for a given depth.
        
        Used for scoring response accuracy.
        """
        markers = {
            'time_points': [],
            'recursion_level': int(depth),
            'is_fractional': (depth % 1.0 == 0.5)
        }
        
        # Generate expected time points
        for i in range(int(depth) + 1):
            markers['time_points'].append(f"T{i}")
        
        # Add fractional markers if needed
        if markers['is_fractional']:
            markers['transition_words'] = [
                'starting', 'beginning', 'noticing', 
                'becoming', 'partially', 'almost'
            ]
        else:
            markers['transition_words'] = [
                'realize', 'observe', 'comprehend', 
                'understand', 'know', 'grasp'
            ]
        
        return markers
    
    def validate_depth_sequence(self, depths: list) -> bool:
        """
        Validate that a sequence of depths is properly ordered.
        
        Scientific rationale: Ensures monotonic complexity increase
        for valid exponential decay analysis (Box & Cox, 1964, JRSS).
        """
        if not depths:
            return False
        
        # Check for proper ordering
        for i in range(1, len(depths)):
            if depths[i] <= depths[i-1]:
                return False
        
        # Check for consistent spacing (if more than 2 depths)
        if len(depths) > 2:
            spacings = [depths[i] - depths[i-1] for i in range(1, len(depths))]
            if len(set(spacings)) > 1:  # Multiple different spacings
                print(f"Warning: Inconsistent spacing in depths: {spacings}")
        
        return True
    
    def generate_batch(self, depths: list) -> Dict[float, str]:
        """
        Generate prompts for multiple depths.
        
        Args:
            depths: List of temporal depths
            
        Returns:
            Dictionary mapping depth to prompt
        """
        if not self.validate_depth_sequence(sorted(depths)):
            raise ValueError("Invalid depth sequence")
        
        batch = {}
        for depth in depths:
            batch[depth] = self.generate(depth)
        
        return batch
    
    def export_prompts(self, filepath: str = 'data/prompts.json'):
        """Export all generated prompts for reproducibility."""
        export_data = {
            'templates': self.templates,
            'cache': self.cache,
            'metadata': {
                'min_depth': min(self.templates.keys()),
                'max_depth': max(self.templates.keys()),
                'total_depths': len(self.templates)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Exported {len(self.templates)} prompts to {filepath}")


class ControlPromptGenerator:
    """Generate control prompts to test alternative hypotheses."""
    
    def __init__(self):
        self.types = ['scrambled', 'non_temporal', 'semantic_match']
    
    def generate_scrambled(self, depth: float) -> str:
        """
        Scramble temporal markers while maintaining syntax.
        Tests if it's temporal processing vs pattern matching.
        """
        # Take original prompt and scramble time markers
        base = PromptGenerator().generate(depth)
        
        # Scramble T0, T1, T2, etc.
        import random
        time_points = [f"T{i}" for i in range(int(depth) + 1)]
        scrambled = time_points.copy()
        random.shuffle(scrambled)
        
        result = base
        for orig, scram in zip(time_points, scrambled):
            result = result.replace(orig, f"XXX{scram}XXX")
        result = result.replace("XXX", "")
        
        return result
    
    def generate_non_temporal(self, depth: float) -> str:
        """
        Generate prompts with similar complexity but no temporal element.
        Tests if decay is specific to temporal processing.
        """
        templates = {
            1.0: "I have a thought.",
            2.0: "I have a thought. This thought contains another thought.",
            3.0: "I have a thought. This thought contains another thought. That inner thought itself contains a thought.",
            4.0: "I have a thought. This thought contains another thought. That inner thought itself contains a thought. The deepest thought holds yet another thought.",
            5.0: "I have a thought. This thought contains another thought. That inner thought itself contains a thought. The deepest thought holds yet another thought. This final thought encompasses all previous thoughts."
        }
        
        if depth not in templates:
            # Interpolate for fractional depths
            depth = round(depth)
        
        return templates.get(depth, templates[1.0]) + "\n\nContinue this pattern."
    
    def generate_semantic_match(self, depth: float) -> str:
        """
        Generate prompts with matched word count but different structure.
        Controls for length effects.
        """
        generator = PromptGenerator()
        original = generator.generate(depth)
        word_count = len(original.split())
        
        # Generate semantically different content with same word count
        base = "Consider the following sequence of observations: "
        filler = "observation " * (word_count // 2)
        
        return base + filler[:word_count-10] + ". Continue this sequence."