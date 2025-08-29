#!/usr/bin/env python3
"""
Phase 0: Learning Progress Drive Validation Experiment

This script validates the LP drive on synthetic data before integration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
import argparse
from pathlib import Path
import logging
from datetime import datetime

from src.core.learning_progress import LearningProgressDrive
from src.environment.synthetic_data import LPValidationSuite, SyntheticDataGenerator


def setup_logging(log_dir: Path):
    """Set up logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"lp_validation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_lp_validation(config: dict, logger: logging.Logger) -> dict:
    """
    Run comprehensive LP validation suite.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        results: Validation results dictionary
    """
    logger.info("Starting Learning Progress Drive validation...")
    
    # Create LP drive with config parameters
    lp_config = config['learning_progress']
    lp_drive = LearningProgressDrive(
        smoothing_window=lp_config['smoothing_window'],
        derivative_clamp=tuple(lp_config['derivative_clamp']),
        boredom_threshold=lp_config['boredom_threshold'],
        boredom_steps=lp_config['boredom_steps'],
        lp_weight=lp_config['lp_weight'],
        empowerment_weight=lp_config['empowerment_weight'],
        use_adaptive_weights=lp_config['use_adaptive_weights']
    )
    
    # Create validation suite
    validation_suite = LPValidationSuite()
    
    # Run validation tests
    logger.info("Running validation test suite...")
    results = validation_suite.run_all_tests(lp_drive)
    
    # Log results
    logger.info("Validation Results:")
    logger.info("=" * 50)
    
    overall_score = 0.0
    num_tests = 0
    
    for test_name, test_result in results.items():
        if 'error' in test_result:
            logger.error(f"{test_name}: FAILED - {test_result['error']}")
        else:
            score = test_result['score']
            logger.info(f"{test_name}: {score:.3f}")
            overall_score += score
            num_tests += 1
            
            # Log additional metrics
            for key, value in test_result.items():
                if key != 'score':
                    logger.info(f"  {key}: {value}")
                    
    # Compute overall score
    if num_tests > 0:
        overall_score /= num_tests
        logger.info("=" * 50)
        logger.info(f"Overall Score: {overall_score:.3f}")
        
        # Determine pass/fail
        passing_threshold = 0.6
        if overall_score >= passing_threshold:
            logger.info("✓ VALIDATION PASSED")
        else:
            logger.warning("✗ VALIDATION FAILED")
            
    return {
        'overall_score': overall_score,
        'individual_results': results,
        'passed': overall_score >= 0.6 if num_tests > 0 else False
    }


def run_signal_quality_analysis(config: dict, logger: logging.Logger):
    """
    Run detailed signal quality analysis.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("Running signal quality analysis...")
    
    # Create LP drive
    lp_config = config['learning_progress']
    lp_drive = LearningProgressDrive(**lp_config)
    
    # Generate test sequence
    generator = SyntheticDataGenerator(
        sequence_length=config['phase0']['synthetic_data_length']
    )
    
    sequence, ground_truth = generator.generate_learning_breakthrough_sequence()
    
    # Process sequence and collect metrics
    lp_signals = []
    validation_metrics_history = []
    
    for i, data in enumerate(sequence):
        # Simulate prediction error based on learning pattern
        t = data['timestamp']
        if 200 <= t < 500:  # Learning phase
            base_error = 1.0
            learning_reduction = max(0.0, 1.0 - (t - 200) / 100)
            pred_error = base_error * learning_reduction + 0.1
        else:
            pred_error = 0.1
            
        # Compute LP signal
        lp_signal = lp_drive.compute_learning_progress(pred_error)
        lp_signals.append(lp_signal)
        
        # Collect validation metrics every 50 steps
        if i % 50 == 0:
            metrics = lp_drive.get_validation_metrics()
            validation_metrics_history.append(metrics)
            
    # Analyze signal quality
    final_metrics = lp_drive.get_validation_metrics()
    signal_quality_ok = lp_drive.validate_signal_quality()
    
    logger.info("Signal Quality Analysis:")
    logger.info(f"  Signal-to-Noise Ratio: {final_metrics['signal_to_noise_ratio']:.3f}")
    logger.info(f"  Stability Score: {final_metrics['stability_score']:.3f}")
    logger.info(f"  Signal Quality OK: {signal_quality_ok}")
    
    # Correlation with ground truth
    import numpy as np
    correlation = np.corrcoef(lp_signals, ground_truth)[0, 1]
    logger.info(f"  Correlation with Ground Truth: {correlation:.3f}")
    
    return {
        'lp_signals': lp_signals,
        'ground_truth': ground_truth,
        'validation_metrics': final_metrics,
        'signal_quality_ok': signal_quality_ok,
        'correlation': correlation
    }


def main():
    """Main experiment function."""
    parser = argparse.ArgumentParser(description='Phase 0 LP Validation Experiment')
    parser.add_argument('--config', type=str, default='configs/phase0_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='experiments/results/phase0',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir / 'logs')
    config = load_config(args.config)
    
    # Set random seed
    torch.manual_seed(config['experiment']['seed'])
    
    logger.info("Phase 0: Learning Progress Drive Validation")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {output_dir}")
    
    try:
        # Run validation suite
        validation_results = run_lp_validation(config, logger)
        
        # Run signal quality analysis
        quality_results = run_signal_quality_analysis(config, logger)
        
        # Save results
        results = {
            'validation': validation_results,
            'signal_quality': quality_results,
            'config': config
        }
        
        results_file = output_dir / 'lp_validation_results.yaml'
        with open(results_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
            
        logger.info(f"Results saved to {results_file}")
        
        # Exit with appropriate code
        if validation_results['passed']:
            logger.info("Experiment completed successfully!")
            sys.exit(0)
        else:
            logger.error("Experiment failed validation!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()