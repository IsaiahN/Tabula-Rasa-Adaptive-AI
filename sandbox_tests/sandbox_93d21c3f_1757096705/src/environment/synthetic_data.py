"""
Synthetic data generator for Learning Progress Drive validation.

This module creates controlled sensory sequences with known learning patterns
for offline LP validation before integration testing.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import math


class SyntheticDataGenerator:
    """
    Generates synthetic sensory data with predictable learning patterns.
    """
    
    def __init__(
        self,
        sequence_length: int = 1000,
        visual_size: Tuple[int, int, int] = (3, 32, 32),
        proprioception_size: int = 8,
        seed: Optional[int] = None
    ):
        self.sequence_length = sequence_length
        self.visual_size = visual_size
        self.proprioception_size = proprioception_size
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
    def generate_learning_breakthrough_sequence(self) -> Tuple[List[torch.Tensor], List[float]]:
        """
        Generate sequence with clear learning breakthrough pattern.
        
        Returns:
            sensory_sequence: List of sensory inputs
            ground_truth_lp: List of expected learning progress values
        """
        sensory_sequence = []
        ground_truth_lp = []
        
        # Phase 1: Random noise (no learning possible)
        for t in range(200):
            visual = torch.randn(*self.visual_size)
            proprioception = torch.randn(self.proprioception_size)
            
            sensory_sequence.append({
                'visual': visual,
                'proprioception': proprioception,
                'timestamp': t
            })
            ground_truth_lp.append(0.0)  # No learning progress expected
            
        # Phase 2: Introduce simple pattern (learning breakthrough)
        pattern_freq = 0.1
        for t in range(200, 500):
            # Simple sine wave pattern in visual data
            pattern_value = math.sin(t * pattern_freq)
            visual = torch.randn(*self.visual_size) * 0.5 + pattern_value
            
            # Correlated proprioception
            proprioception = torch.randn(self.proprioception_size) * 0.3
            proprioception[0] = pattern_value  # First channel follows pattern
            
            sensory_sequence.append({
                'visual': visual,
                'proprioception': proprioception,
                'timestamp': t
            })
            
            # High learning progress during pattern introduction
            lp_value = max(0.0, 1.0 - abs(t - 350) / 150)  # Peak at t=350
            ground_truth_lp.append(lp_value)
            
        # Phase 3: Pattern becomes predictable (learning plateaus)
        for t in range(500, 800):
            pattern_value = math.sin(t * pattern_freq)
            visual = torch.randn(*self.visual_size) * 0.1 + pattern_value  # Less noise
            
            proprioception = torch.randn(self.proprioception_size) * 0.1
            proprioception[0] = pattern_value
            
            sensory_sequence.append({
                'visual': visual,
                'proprioception': proprioception,
                'timestamp': t
            })
            ground_truth_lp.append(0.0)  # No more learning progress
            
        # Phase 4: New complex pattern (second learning phase)
        for t in range(800, 1000):
            # More complex pattern
            pattern1 = math.sin(t * pattern_freq)
            pattern2 = math.cos(t * pattern_freq * 2)
            combined_pattern = pattern1 * pattern2
            
            visual = torch.randn(*self.visual_size) * 0.3 + combined_pattern
            
            proprioception = torch.randn(self.proprioception_size) * 0.2
            proprioception[0] = pattern1
            proprioception[1] = pattern2
            
            sensory_sequence.append({
                'visual': visual,
                'proprioception': proprioception,
                'timestamp': t
            })
            
            # Moderate learning progress for complex pattern
            lp_value = max(0.0, 0.5 - abs(t - 900) / 200)
            ground_truth_lp.append(lp_value)
            
        return sensory_sequence, ground_truth_lp
        
    def generate_boredom_sequence(self) -> Tuple[List[torch.Tensor], List[float]]:
        """
        Generate sequence that should trigger boredom (constant input).
        
        Returns:
            sensory_sequence: List of sensory inputs
            ground_truth_lp: List of expected learning progress values
        """
        sensory_sequence = []
        ground_truth_lp = []
        
        # Constant input (should lead to boredom)
        constant_visual = torch.ones(*self.visual_size) * 0.5
        constant_proprioception = torch.ones(self.proprioception_size) * 0.3
        
        for t in range(self.sequence_length):
            # Add tiny amount of noise to avoid numerical issues
            visual = constant_visual + torch.randn(*self.visual_size) * 0.01
            proprioception = constant_proprioception + torch.randn(self.proprioception_size) * 0.01
            
            sensory_sequence.append({
                'visual': visual,
                'proprioception': proprioception,
                'timestamp': t
            })
            
            # Should quickly learn and then plateau (boredom)
            if t < 50:
                lp_value = max(0.0, 1.0 - t / 50)  # Quick learning
            else:
                lp_value = 0.0  # Boredom
                
            ground_truth_lp.append(lp_value)
            
        return sensory_sequence, ground_truth_lp
        
    def generate_noisy_sequence(self, noise_level: float = 1.0) -> Tuple[List[torch.Tensor], List[float]]:
        """
        Generate sequence with high noise for stress testing.
        
        Args:
            noise_level: Multiplier for noise magnitude
            
        Returns:
            sensory_sequence: List of sensory inputs
            ground_truth_lp: List of expected learning progress values
        """
        sensory_sequence = []
        ground_truth_lp = []
        
        for t in range(self.sequence_length):
            # High noise with occasional weak patterns
            visual = torch.randn(*self.visual_size) * noise_level
            proprioception = torch.randn(self.proprioception_size) * noise_level
            
            # Weak pattern every 100 steps
            if t % 100 < 10:
                pattern_strength = 0.1 / noise_level  # Weaker with more noise
                visual += pattern_strength
                proprioception += pattern_strength
                
            sensory_sequence.append({
                'visual': visual,
                'proprioception': proprioception,
                'timestamp': t
            })
            
            # Very low learning progress due to noise
            if t % 100 < 10:
                lp_value = 0.1 / noise_level  # Harder to learn with more noise
            else:
                lp_value = 0.0
                
            ground_truth_lp.append(lp_value)
            
        return sensory_sequence, ground_truth_lp
        
    def generate_multi_modal_sequence(self) -> Tuple[List[torch.Tensor], List[float]]:
        """
        Generate sequence with different patterns in different modalities.
        
        Returns:
            sensory_sequence: List of sensory inputs
            ground_truth_lp: List of expected learning progress values
        """
        sensory_sequence = []
        ground_truth_lp = []
        
        for t in range(self.sequence_length):
            # Visual pattern: slow sine wave
            visual_pattern = math.sin(t * 0.02)
            visual = torch.randn(*self.visual_size) * 0.3 + visual_pattern
            
            # Proprioception pattern: faster cosine wave
            proprio_pattern = math.cos(t * 0.1)
            proprioception = torch.randn(self.proprioception_size) * 0.2
            proprioception[0] = proprio_pattern
            
            sensory_sequence.append({
                'visual': visual,
                'proprioception': proprioception,
                'timestamp': t
            })
            
            # Learning progress should reflect both modalities
            visual_lp = 0.3 if 100 < t < 300 else 0.0  # Learn visual pattern
            proprio_lp = 0.5 if 400 < t < 600 else 0.0  # Learn proprio pattern
            combined_lp = max(visual_lp, proprio_lp)
            
            ground_truth_lp.append(combined_lp)
            
        return sensory_sequence, ground_truth_lp
        
    def generate_outlier_sequence(self) -> Tuple[List[torch.Tensor], List[float]]:
        """
        Generate sequence with extreme outliers for robustness testing.
        
        Returns:
            sensory_sequence: List of sensory inputs
            ground_truth_lp: List of expected learning progress values
        """
        sensory_sequence = []
        ground_truth_lp = []
        
        for t in range(self.sequence_length):
            # Normal pattern
            pattern = math.sin(t * 0.05)
            visual = torch.randn(*self.visual_size) * 0.2 + pattern
            proprioception = torch.randn(self.proprioception_size) * 0.2 + pattern
            
            # Inject extreme outliers
            if t % 50 == 0:  # Every 50 steps
                outlier_magnitude = 10.0
                visual += torch.randn(*self.visual_size) * outlier_magnitude
                proprioception += torch.randn(self.proprioception_size) * outlier_magnitude
                
            sensory_sequence.append({
                'visual': visual,
                'proprioception': proprioception,
                'timestamp': t
            })
            
            # Learning progress should be robust to outliers
            if 200 < t < 400:
                lp_value = 0.4
            else:
                lp_value = 0.0
                
            ground_truth_lp.append(lp_value)
            
        return sensory_sequence, ground_truth_lp


class LPValidationSuite:
    """
    Comprehensive validation suite for Learning Progress Drive.
    """
    
    def __init__(self):
        self.generator = SyntheticDataGenerator()
        self.test_results = {}
        
    def run_all_tests(self, lp_drive) -> Dict[str, Dict[str, float]]:
        """
        Run complete validation suite on LP drive.
        
        Args:
            lp_drive: LearningProgressDrive instance to test
            
        Returns:
            test_results: Dictionary with test results and metrics
        """
        tests = [
            ('breakthrough_detection', self._test_breakthrough_detection),
            ('boredom_detection', self._test_boredom_detection),
            ('noise_robustness', self._test_noise_robustness),
            ('multi_modal_handling', self._test_multi_modal_handling),
            ('outlier_robustness', self._test_outlier_robustness)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            print(f"Running {test_name}...")
            try:
                test_result = test_func(lp_drive)
                results[test_name] = test_result
                print(f"  ✓ {test_name}: {test_result['score']:.3f}")
            except Exception as e:
                print(f"  ✗ {test_name}: FAILED - {e}")
                results[test_name] = {'score': 0.0, 'error': str(e)}
                
        return results
        
    def _test_breakthrough_detection(self, lp_drive) -> Dict[str, float]:
        """Test ability to detect learning breakthroughs."""
        sequence, ground_truth = self.generator.generate_learning_breakthrough_sequence()
        
        # Reset LP drive
        lp_drive.__init__()
        
        predicted_lp = []
        for data in sequence:
            # Simulate prediction error (high initially, then decreases during learning)
            t = data['timestamp']
            if 200 <= t < 500:  # Learning phase
                base_error = 1.0
                learning_reduction = max(0.0, 1.0 - (t - 200) / 100)
                pred_error = base_error * learning_reduction + 0.1
            else:
                pred_error = 0.1  # Low error when no learning
                
            lp_signal = lp_drive.compute_learning_progress(pred_error)
            predicted_lp.append(lp_signal)
            
        # Compute correlation with ground truth
        correlation = np.corrcoef(predicted_lp, ground_truth)[0, 1]
        
        # Check for breakthrough detection
        breakthrough_detected = max(predicted_lp[200:500]) > 0.3
        
        return {
            'score': correlation if not np.isnan(correlation) else 0.0,
            'breakthrough_detected': breakthrough_detected,
            'max_lp_signal': max(predicted_lp)
        }
        
    def _test_boredom_detection(self, lp_drive) -> Dict[str, float]:
        """Test boredom detection on constant input."""
        sequence, ground_truth = self.generator.generate_boredom_sequence()
        
        # Reset LP drive
        lp_drive.__init__()
        
        for data in sequence:
            # Constant low prediction error
            lp_drive.compute_learning_progress(0.05)
            
        boredom_detected = lp_drive.is_bored()
        
        return {
            'score': 1.0 if boredom_detected else 0.0,
            'boredom_detected': boredom_detected,
            'boredom_counter': lp_drive.low_lp_counter
        }
        
    def _test_noise_robustness(self, lp_drive) -> Dict[str, float]:
        """Test robustness to high noise levels."""
        sequence, ground_truth = self.generator.generate_noisy_sequence(noise_level=5.0)
        
        # Reset LP drive
        lp_drive.__init__()
        
        predicted_lp = []
        for data in sequence:
            # High noise in prediction errors
            base_error = np.random.normal(1.0, 2.0)  # High variance
            lp_signal = lp_drive.compute_learning_progress(abs(base_error))
            predicted_lp.append(lp_signal)
            
        # Check signal stability (should not have extreme values)
        signal_std = np.std(predicted_lp)
        max_signal = max(abs(x) for x in predicted_lp)
        
        # Good robustness = low variance and bounded signals
        stability_score = 1.0 / (1.0 + signal_std)
        bounds_score = 1.0 if max_signal < 2.0 else 0.0
        
        return {
            'score': (stability_score + bounds_score) / 2.0,
            'signal_std': signal_std,
            'max_signal': max_signal
        }
        
    def _test_multi_modal_handling(self, lp_drive) -> Dict[str, float]:
        """Test handling of multi-modal sensory input."""
        sequence, ground_truth = self.generator.generate_multi_modal_sequence()
        
        # Reset LP drive
        lp_drive.__init__()
        
        predicted_lp = []
        for data in sequence:
            # Simulate different error patterns for different modalities
            t = data['timestamp']
            
            # Visual learning phase
            if 100 < t < 300:
                visual_error = max(0.1, 1.0 - (t - 100) / 200)
            else:
                visual_error = 0.1
                
            # Proprioception learning phase
            if 400 < t < 600:
                proprio_error = max(0.1, 1.0 - (t - 400) / 200)
            else:
                proprio_error = 0.1
                
            # Combined error (could be average or max)
            combined_error = (visual_error + proprio_error) / 2.0
            
            lp_signal = lp_drive.compute_learning_progress(combined_error)
            predicted_lp.append(lp_signal)
            
        # Check if both learning phases are detected
        visual_phase_lp = max(predicted_lp[100:300])
        proprio_phase_lp = max(predicted_lp[400:600])
        
        both_detected = visual_phase_lp > 0.1 and proprio_phase_lp > 0.1
        
        return {
            'score': 1.0 if both_detected else 0.0,
            'visual_phase_max': visual_phase_lp,
            'proprio_phase_max': proprio_phase_lp
        }
        
    def _test_outlier_robustness(self, lp_drive) -> Dict[str, float]:
        """Test robustness to extreme outliers."""
        sequence, ground_truth = self.generator.generate_outlier_sequence()
        
        # Reset LP drive
        lp_drive.__init__()
        
        predicted_lp = []
        outlier_steps = []
        
        for data in sequence:
            t = data['timestamp']
            
            # Normal error with occasional extreme outliers
            if t % 50 == 0:  # Outlier step
                pred_error = 100.0  # Extreme outlier
                outlier_steps.append(len(predicted_lp))
            else:
                # Normal learning pattern
                if 200 < t < 400:
                    pred_error = max(0.1, 1.0 - (t - 200) / 200)
                else:
                    pred_error = 0.1
                    
            lp_signal = lp_drive.compute_learning_progress(pred_error)
            predicted_lp.append(lp_signal)
            
        # Check if outliers cause instability
        outlier_impacts = [abs(predicted_lp[i]) for i in outlier_steps if i < len(predicted_lp)]
        max_outlier_impact = max(outlier_impacts) if outlier_impacts else 0.0
        
        # Good robustness = outliers don't cause extreme LP signals
        robustness_score = 1.0 if max_outlier_impact < 2.0 else 0.0
        
        return {
            'score': robustness_score,
            'max_outlier_impact': max_outlier_impact,
            'num_outliers': len(outlier_steps)
        }