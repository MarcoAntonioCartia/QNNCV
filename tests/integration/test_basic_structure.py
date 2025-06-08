import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_import_structure():
    """
    Test that the new import structure works correctly.
    
    This test validates that all modules can be imported successfully
    and that the package structure is properly organized.
    """
    print("Testing import structure...")
    
    # Test training module import
    try:
        from training.qgan_trainer import QGAN
        print("✓ Training module imported successfully")
    except ImportError as e:
        print(f"✗ Training module import failed: {e}")
        return False
    
    # Test classical models import
    try:
        from models.generators.classical_generator import ClassicalGenerator
        from models.discriminators.classical_discriminator import ClassicalDiscriminator
        print("✓ Classical models imported successfully")
    except ImportError as e:
        print(f"✗ Classical models import failed: {e}")
        return False
    
    # Test utilities import
    try:
        from utils.data_utils import load_dataset, load_synthetic_data
        from utils.visualization import plot_results
        from utils.metrics import compute_wasserstein_distance
        print("✓ Utilities imported successfully")
    except ImportError as e:
        print(f"✗ Utilities import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """
    Test basic functionality without quantum dependencies.
    
    This test validates that the core QGAN framework works with
    classical components and synthetic data.
    """
    print("\nTesting basic functionality...")
    
    try:
        import tensorflow as tf
        from training.qgan_trainer import QGAN
        from models.generators.classical_generator import ClassicalGenerator
        from models.discriminators.classical_discriminator import ClassicalDiscriminator
        from utils.data_utils import load_synthetic_data
        
        # Generate synthetic data
        data = load_synthetic_data(dataset_type="spiral", num_samples=100)
        print(f"✓ Generated synthetic data with shape: {data.shape}")
        
        # Initialize classical components
        generator = ClassicalGenerator(latent_dim=5, output_dim=2)
        discriminator = ClassicalDiscriminator(input_dim=2)
        print("✓ Classical components initialized")
        
        # Create QGAN
        qgan = QGAN(generator, discriminator, latent_dim=5)
        print("✓ QGAN framework initialized")
        
        # Test single training step
        z = tf.random.normal([10, 5])
        fake_samples = generator.generate(z)
        probs = discriminator.discriminate(fake_samples)
        
        print(f"✓ Generator output shape: {fake_samples.shape}")
        print(f"✓ Discriminator output shape: {probs.shape}")
        print(f"✓ Probability range: [{tf.reduce_min(probs):.3f}, {tf.reduce_max(probs):.3f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_configuration_loading():
    """
    Test configuration file loading from new location.
    
    This test validates that configuration files can be loaded
    from the reorganized config directory.
    """
    print("\nTesting configuration loading...")
    
    try:
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'config.yaml')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print("✓ Configuration file loaded successfully")
            print(f"✓ Config keys: {list(config.keys())}")
            return True
        else:
            print(f"✗ Configuration file not found at: {config_path}")
            return False
            
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("QNNCV Project Structure Validation")
    print("=" * 50)
    
    tests = [
        test_import_structure,
        test_basic_functionality,
        test_configuration_loading
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "PASS" if result else "FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    total_passed = sum(results)
    total_tests = len(results)
    
    print(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 All tests passed! Project reorganization successful.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
