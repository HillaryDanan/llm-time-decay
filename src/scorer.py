"""
Enhanced scoring system for fractional temporal depths.

Scientific basis: Multi-metric coherence assessment based on
cognitive consistency theory (Festinger, 1957; Thagard, 2000).
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from config import SCORING


@dataclass
class ScoringResult:
    """Container for scoring results with detailed metrics."""
    total_score: float
    temporal_ordering: float
    causal_maintenance: float
    depth_accuracy: float
    transition_smoothness: float
    raw_response: str
    expected_depth: float
    actual_depth: float
    confidence: float
    

class TemporalScorer:
    """Score responses for temporal coherence at fractional depths."""
    
    def __init__(self):
        self.metrics = SCORING['metrics']
        self.fractional_markers = SCORING['fractional_markers']
        self.complete_markers = SCORING['complete_markers']
        
    def score(self, response: str, expected_depth: float) -> ScoringResult:
        """
        Score a response for temporal coherence.
        
        Args:
            response: Model's response
            expected_depth: Expected temporal depth
            
        Returns:
            ScoringResult with detailed metrics
            
        Theoretical framework:
            Based on hierarchical coherence metrics from
            discourse analysis (Mann & Thompson, 1988, Text).
        """
        scores = {
            'temporal_ordering': self._score_temporal_ordering(response),
            'causal_maintenance': self._score_causal_chain(response, expected_depth),
            'depth_accuracy': self._score_depth_accuracy(response, expected_depth),
            'transition_smoothness': self._score_transition(response, expected_depth)
        }
        
        actual_depth = self._measure_actual_depth(response)
        confidence = self._calculate_confidence(scores, response)
        
        return ScoringResult(
            total_score=sum(scores.values()),
            temporal_ordering=scores['temporal_ordering'],
            causal_maintenance=scores['causal_maintenance'],
            depth_accuracy=scores['depth_accuracy'],
            transition_smoothness=scores['transition_smoothness'],
            raw_response=response,
            expected_depth=expected_depth,
            actual_depth=actual_depth,
            confidence=confidence
        )
    
    def _score_temporal_ordering(self, response: str) -> float:
        """
        Score preservation of temporal sequence.
        
        Mathematical basis: Kendall's tau for rank correlation
        adapted to temporal sequences (Kendall, 1938).
        """
        # Extract time markers
        time_pattern = r'T(\d+)'
        matches = re.findall(time_pattern, response)
        
        if not matches:
            return 0.0
        
        # Convert to integers
        time_points = [int(m) for m in matches]
        
        # Check if monotonically increasing
        inversions = 0
        for i in range(1, len(time_points)):
            if time_points[i] < time_points[i-1]:
                inversions += 1
        
        # Calculate score (perfect ordering = 0.25, each inversion reduces score)
        if inversions == 0:
            return self.metrics['temporal_ordering']
        else:
            penalty = inversions * 0.05
            return max(0, self.metrics['temporal_ordering'] - penalty)
    
    def _score_causal_chain(self, response: str, expected_depth: float) -> float:
        """
        Score maintenance of causal relationships.
        
        Based on causal coherence theory (Trabasso & Sperry, 1985, JML).
        """
        causal_phrases = [
            'because', 'therefore', 'so', 'thus', 'consequently',
            'realize', 'observe', 'comprehend', 'understand',
            'thinking about thinking'
        ]
        
        # Count causal connectives
        causal_count = sum(1 for phrase in causal_phrases if phrase in response.lower())
        
        # Expected causal links based on depth
        expected_links = int(expected_depth)
        
        if causal_count >= expected_links:
            return self.metrics['causal_maintenance']
        else:
            ratio = causal_count / max(expected_links, 1)
            return self.metrics['causal_maintenance'] * ratio
    
    def _score_depth_accuracy(self, response: str, expected_depth: float) -> float:
        """
        Score accuracy of recursive depth.
        
        Uses edit distance concept adapted for recursive structures
        (Levenshtein, 1966; adapted for hierarchical text).
        """
        actual_depth = self._measure_actual_depth(response)
        
        # Calculate error
        depth_error = abs(actual_depth - expected_depth)
        
        # Score based on error magnitude
        if depth_error <= 0.25:  # Very close
            return self.metrics['depth_accuracy']
        elif depth_error <= 0.5:  # Acceptable
            return self.metrics['depth_accuracy'] * 0.75
        elif depth_error <= 1.0:  # Moderate error
            return self.metrics['depth_accuracy'] * 0.5
        else:  # Large error
            return 0.0
    
    def _score_transition(self, response: str, expected_depth: float) -> float:
        """
        Score transition smoothness for fractional depths.
        
        Novel metric for detecting discrete vs continuous processing.
        """
        is_fractional = (expected_depth % 1.0 == 0.5)
        
        if is_fractional:
            # Look for partial/transitional language
            marker_count = sum(1 for marker in self.fractional_markers 
                             if marker in response.lower())
            if marker_count >= 2:
                return self.metrics['transition_smoothness']
            elif marker_count == 1:
                return self.metrics['transition_smoothness'] * 0.5
            else:
                return 0.0
        else:
            # Look for complete/definitive language
            marker_count = sum(1 for marker in self.complete_markers 
                             if marker in response.lower())
            if marker_count >= 1:
                return self.metrics['transition_smoothness']
            else:
                return self.metrics['transition_smoothness'] * 0.5
    
    def _measure_actual_depth(self, response: str) -> float:
        """
        Measure the actual recursive depth in response.
        
        Algorithm: Count nested "thinking about" patterns.
        """
        # Count recursive patterns
        thinking_pattern = r'thinking(\s+about\s+thinking)+'
        matches = re.findall(thinking_pattern, response.lower())
        
        if not matches:
            # Try counting time points as proxy
            time_pattern = r'T(\d+)'
            time_matches = re.findall(time_pattern, response)
            if time_matches:
                return float(max(int(m) for m in time_matches))
            return 1.0
        
        # Count depth from longest match
        max_depth = 1
        for match in matches:
            depth = match.count('about thinking') + 1
            max_depth = max(max_depth, depth)
        
        # Check for fractional indicators
        if any(marker in response.lower() for marker in self.fractional_markers):
            max_depth += 0.5
        
        return float(max_depth)
    
    def _calculate_confidence(self, scores: Dict, response: str) -> float:
        """
        Calculate confidence in scoring.
        
        Based on response length and metric consistency.
        """
        # Factor 1: Response length (too short = low confidence)
        word_count = len(response.split())
        length_factor = min(1.0, word_count / 50.0)
        
        # Factor 2: Metric consistency (high variance = low confidence)
        score_values = list(scores.values())
        if len(score_values) > 0:
            variance = np.var(score_values)
            consistency_factor = 1.0 - min(variance * 4, 1.0)
        else:
            consistency_factor = 0.0
        
        # Combine factors
        confidence = (length_factor + consistency_factor) / 2.0
        
        return confidence


class BatchScorer:
    """Handle batch scoring operations."""
    
    def __init__(self):
        self.scorer = TemporalScorer()
        self.results = []
        
    def score_batch(self, responses: List[Tuple[str, float]]) -> List[ScoringResult]:
        """
        Score multiple responses.
        
        Args:
            responses: List of (response_text, expected_depth) tuples
            
        Returns:
            List of ScoringResult objects
        """
        results = []
        for response_text, expected_depth in responses:
            result = self.scorer.score(response_text, expected_depth)
            results.append(result)
            self.results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict:
        """Calculate summary statistics for scored responses."""
        if not self.results:
            return {}
        
        scores = [r.total_score for r in self.results]
        depths = [r.expected_depth for r in self.results]
        
        # Group by depth
        depth_scores = {}
        for result in self.results:
            depth = result.expected_depth
            if depth not in depth_scores:
                depth_scores[depth] = []
            depth_scores[depth].append(result.total_score)
        
        stats = {
            'overall': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'n': len(scores)
            },
            'by_depth': {}
        }
        
        for depth, depth_score_list in depth_scores.items():
            stats['by_depth'][depth] = {
                'mean': np.mean(depth_score_list),
                'std': np.std(depth_score_list),
                'n': len(depth_score_list)
            }
        
        return stats