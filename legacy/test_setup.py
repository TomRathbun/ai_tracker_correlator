"""
Test script to verify all new modules can be imported correctly.
"""
import sys
from pathlib import Path

def test_imports():
    """Test that all new modules can be imported."""
    print("Testing module imports...")
    
    try:
        from src.config import ExperimentConfig, ModelConfig, TrainingConfig
        print("✓ src.config imported successfully")
    except Exception as e:
        print(f"✗ src.config import failed: {e}")
        return False
    
    try:
        from src.metrics import TrackingMetrics, format_metrics
        print("✓ src.metrics imported successfully")
    except Exception as e:
        print(f"✗ src.metrics import failed: {e}")
        return False
    
    try:
        from src.trainer import Trainer
        print("✓ src.trainer imported successfully")
    except Exception as e:
        print(f"✗ src.trainer import failed: {e}")
        return False
    
    try:
        from src.visualize import (
            visualize_attention_weights,
            visualize_track_predictions,
            plot_confusion_matrix
        )
        print("✓ src.visualize imported successfully")
    except Exception as e:
        print(f"✗ src.visualize import failed: {e}")
        return False
    
    try:
        from src.optimize import objective, run_optimization
        print("✓ src.optimize imported successfully")
    except Exception as e:
        print(f"✗ src.optimize import failed: {e}")
        return False
    
    return True


def test_config_creation():
    """Test configuration creation and serialization."""
    print("\nTesting configuration management...")
    
    try:
        from src.config import ExperimentConfig
        
        # Create default config
        config = ExperimentConfig.default()
        print("✓ Default config created")
        
        # Save config
        test_path = Path("test_config.json")
        config.save(str(test_path))
        print(f"✓ Config saved to {test_path}")
        
        # Load config
        loaded_config = ExperimentConfig.load(str(test_path))
        print("✓ Config loaded successfully")
        
        # Verify values match
        assert config.model.hidden_dim == loaded_config.model.hidden_dim
        print("✓ Config values match after save/load")
        
        # Cleanup
        test_path.unlink()
        print("✓ Test file cleaned up")
        
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False


def test_metrics():
    """Test metrics computation."""
    print("\nTesting metrics computation...")
    
    try:
        import torch
        from src.metrics import TrackingMetrics
        
        metrics = TrackingMetrics(match_threshold=15000.0)
        
        # Simulate some predictions and ground truth
        pred_states = torch.randn(5, 6)  # 5 predictions
        gt_states = torch.randn(3, 6)    # 3 ground truth
        
        metrics.update(pred_states, gt_states)
        result = metrics.compute()
        
        print("✓ Metrics computed successfully")
        print(f"  MOTA: {result['MOTA']:.3f}")
        print(f"  Precision: {result['precision']:.3f}")
        print(f"  Recall: {result['recall']:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("TRACKER CORRELATOR - MODULE VERIFICATION")
    print("="*60)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test config
    results.append(("Configuration", test_config_creation()))
    
    # Test metrics
    results.append(("Metrics", test_metrics()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All tests passed! The tracker correlator is ready to use.")
        print("\nNext steps:")
        print("  1. Run: python train_model.py")
        print("  2. Monitor: tensorboard --logdir runs/")
        print("  3. Evaluate: python evaluate_model.py --checkpoint checkpoints/best_model.pt")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
