#!/usr/bin/env python3
"""
Phase 0: DNC Memory System Validation Experiment

This script validates the DNC memory system on pattern storage tasks.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
from pathlib import Path
import logging
from datetime import datetime
import numpy as np

from src.memory.dnc import DNCMemory


def setup_logging(log_dir: Path):
    """Set up logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"memory_test_{timestamp}.log"
    
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


class CopyTask:
    """Copy task for testing DNC memory capabilities."""
    
    def __init__(self, seq_length: int = 10, vector_size: int = 8):
        self.seq_length = seq_length
        self.vector_size = vector_size
        
    def generate_batch(self, batch_size: int):
        """Generate a batch of copy task sequences."""
        # Random input sequences
        input_seq = torch.randn(batch_size, self.seq_length, self.vector_size)
        
        # Add delimiter and output prompt
        delimiter = torch.zeros(batch_size, 1, self.vector_size)
        delimiter[:, :, 0] = 1.0  # First dimension indicates delimiter
        
        # Full input: sequence + delimiter + zeros for output
        full_input = torch.cat([
            input_seq,
            delimiter,
            torch.zeros(batch_size, self.seq_length, self.vector_size)
        ], dim=1)
        
        # Target: zeros during input, then copy of original sequence
        target = torch.cat([
            torch.zeros(batch_size, self.seq_length + 1, self.vector_size),
            input_seq
        ], dim=1)
        
        return full_input, target


class AssociativeRecallTask:
    """Associative recall task for testing memory associations."""
    
    def __init__(self, num_pairs: int = 5, vector_size: int = 8):
        self.num_pairs = num_pairs
        self.vector_size = vector_size
        
    def generate_batch(self, batch_size: int):
        """Generate batch of associative recall sequences."""
        # Generate key-value pairs
        keys = torch.randn(batch_size, self.num_pairs, self.vector_size)
        values = torch.randn(batch_size, self.num_pairs, self.vector_size)
        
        # Create input sequence: key1, value1, key2, value2, ..., query_key
        input_seq = []
        for i in range(self.num_pairs):
            input_seq.append(keys[:, i:i+1, :])
            input_seq.append(values[:, i:i+1, :])
            
        # Add query (one of the keys)
        query_idx = torch.randint(0, self.num_pairs, (batch_size,))
        query_keys = keys[torch.arange(batch_size), query_idx].unsqueeze(1)
        input_seq.append(query_keys)
        
        # Add output prompt
        input_seq.append(torch.zeros(batch_size, 1, self.vector_size))
        
        full_input = torch.cat(input_seq, dim=1)
        
        # Target: corresponding value for the query key
        target_values = values[torch.arange(batch_size), query_idx].unsqueeze(1)
        target = torch.cat([
            torch.zeros(batch_size, full_input.size(1) - 1, self.vector_size),
            target_values
        ], dim=1)
        
        return full_input, target


class DNCSolver(nn.Module):
    """DNC-based solver for memory tasks."""
    
    def __init__(self, input_size: int, output_size: int, memory_config: dict):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        # DNC memory
        self.dnc = DNCMemory(**memory_config)
        
        # Output layer
        self.output_layer = nn.Linear(
            memory_config['num_read_heads'] * memory_config['word_size'] + 
            memory_config['controller_size'],
            output_size
        )
        
    def forward(self, input_sequence):
        """Forward pass through DNC solver."""
        batch_size, seq_len, _ = input_sequence.shape
        
        # Initialize
        prev_reads = torch.zeros(
            batch_size, 
            self.dnc.num_read_heads * self.dnc.word_size
        )
        controller_state = None
        
        outputs = []
        
        for t in range(seq_len):
            # DNC forward pass
            read_vectors, controller_output, controller_state, debug_info = self.dnc(
                input_sequence[:, t, :], prev_reads, controller_state
            )
            
            # Combine reads and controller output
            combined = torch.cat([read_vectors, controller_output], dim=-1)
            
            # Generate output
            output = self.output_layer(combined)
            outputs.append(output)
            
            # Update for next step
            prev_reads = read_vectors
            
        return torch.stack(outputs, dim=1)


def train_memory_task(task, model, config, logger):
    """Train DNC on a memory task."""
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()
    
    num_epochs = 1000
    batch_size = config['training']['batch_size']
    
    losses = []
    accuracies = []
    
    logger.info(f"Training on {task.__class__.__name__}...")
    
    for epoch in range(num_epochs):
        # Generate batch
        inputs, targets = task.generate_batch(batch_size)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Compute loss (only on output positions)
        mask = (targets != 0).any(dim=-1)  # Non-zero targets
        if mask.any():
            loss = criterion(outputs[mask], targets[mask])
        else:
            loss = criterion(outputs, targets)
            
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        optimizer.step()
        
        # Track metrics
        losses.append(loss.item())
        
        # Compute accuracy (for output positions only)
        if mask.any():
            with torch.no_grad():
                pred_outputs = outputs[mask]
                true_outputs = targets[mask]
                accuracy = 1.0 - torch.mean(torch.norm(pred_outputs - true_outputs, dim=-1))
                accuracies.append(accuracy.item())
        else:
            accuracies.append(0.0)
            
        # Log progress
        if epoch % 100 == 0:
            logger.info(f"Epoch {epoch}: Loss = {loss.item():.4f}, Accuracy = {accuracies[-1]:.4f}")
            
    return {
        'final_loss': losses[-1],
        'final_accuracy': accuracies[-1],
        'loss_history': losses,
        'accuracy_history': accuracies
    }


def test_memory_utilization(model, task, config, logger):
    """Test memory utilization during task execution."""
    model.eval()
    
    with torch.no_grad():
        # Generate test batch
        inputs, targets = task.generate_batch(1)  # Single sequence
        
        # Track memory metrics during execution
        memory_metrics = []
        
        # Monkey patch to collect metrics
        original_forward = model.dnc.forward
        
        def forward_with_metrics(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            metrics = model.dnc.get_memory_metrics()
            memory_metrics.append(metrics)
            return result
            
        model.dnc.forward = forward_with_metrics
        
        # Run forward pass
        outputs = model(inputs)
        
        # Restore original forward
        model.dnc.forward = original_forward
        
    # Analyze memory usage
    if memory_metrics:
        avg_utilization = np.mean([m['memory_utilization'] for m in memory_metrics])
        max_utilization = np.max([m['memory_utilization'] for m in memory_metrics])
        
        logger.info(f"Memory Utilization - Average: {avg_utilization:.3f}, Max: {max_utilization:.3f}")
        
        return {
            'average_utilization': avg_utilization,
            'max_utilization': max_utilization,
            'utilization_history': [m['memory_utilization'] for m in memory_metrics]
        }
    else:
        return {'average_utilization': 0.0, 'max_utilization': 0.0}


def main():
    """Main experiment function."""
    parser = argparse.ArgumentParser(description='Phase 0 Memory Test Experiment')
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
    
    logger.info("Phase 0: DNC Memory System Validation")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {output_dir}")
    
    try:
        memory_config = config['memory']
        vector_size = 8
        
        # Test 1: Copy Task
        logger.info("=" * 50)
        logger.info("Testing Copy Task")
        
        copy_task = CopyTask(seq_length=10, vector_size=vector_size)
        copy_model = DNCSolver(vector_size, vector_size, memory_config)
        
        copy_results = train_memory_task(copy_task, copy_model, config, logger)
        copy_memory_usage = test_memory_utilization(copy_model, copy_task, config, logger)
        
        # Test 2: Associative Recall Task
        logger.info("=" * 50)
        logger.info("Testing Associative Recall Task")
        
        recall_task = AssociativeRecallTask(num_pairs=5, vector_size=vector_size)
        recall_model = DNCSolver(vector_size, vector_size, memory_config)
        
        recall_results = train_memory_task(recall_task, recall_model, config, logger)
        recall_memory_usage = test_memory_utilization(recall_model, recall_task, config, logger)
        
        # Evaluate results
        logger.info("=" * 50)
        logger.info("Memory Test Results:")
        
        copy_passed = copy_results['final_accuracy'] > 0.8
        recall_passed = recall_results['final_accuracy'] > 0.7
        memory_used = copy_memory_usage['average_utilization'] > 0.1
        
        logger.info(f"Copy Task - Accuracy: {copy_results['final_accuracy']:.3f} ({'PASS' if copy_passed else 'FAIL'})")
        logger.info(f"Recall Task - Accuracy: {recall_results['final_accuracy']:.3f} ({'PASS' if recall_passed else 'FAIL'})")
        logger.info(f"Memory Usage: {copy_memory_usage['average_utilization']:.3f} ({'PASS' if memory_used else 'FAIL'})")
        
        overall_passed = copy_passed and recall_passed and memory_used
        
        # Save results
        results = {
            'copy_task': {
                'training_results': copy_results,
                'memory_usage': copy_memory_usage,
                'passed': copy_passed
            },
            'recall_task': {
                'training_results': recall_results,
                'memory_usage': recall_memory_usage,
                'passed': recall_passed
            },
            'overall_passed': overall_passed,
            'config': config
        }
        
        results_file = output_dir / 'memory_test_results.yaml'
        with open(results_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
            
        logger.info(f"Results saved to {results_file}")
        
        if overall_passed:
            logger.info("✓ Memory system validation PASSED!")
            sys.exit(0)
        else:
            logger.error("✗ Memory system validation FAILED!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()