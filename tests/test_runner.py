"""
Test runner for the Explainability AI Tool (Project Noor)
This script runs all tests and ensures they pass before deployment
"""
import unittest
import sys
import os
from pathlib import Path

def run_all_tests():
    """
    Run all tests in the test suite
    """
    print("=" * 60)
    print("EXPLAINABILITY AI TOOL - TEST SUITE EXECUTION")
    print("=" * 60)
    
    # Discover and run all tests
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    
    # Add project root to Python path
    sys.path.insert(0, str(project_root))
    
    # Discover tests
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(test_dir), pattern='test_*.py')
    
    # Count total tests
    test_count = 0
    for test_group in suite:
        test_count += test_group.countTestCases()
    
    print(f"Discovered {test_count} test cases")
    print("-" * 60)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        buffer=True  # Capture stdout/stderr during tests
    )
    
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("TEST EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors))/result.testsRun*100:.1f}%")
    
    # Print failures if any
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"\n{test}:")
            print(traceback)
    
    # Print errors if any
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"\n{test}:")
            print(traceback)
    
    # Determine overall success
    success = result.wasSuccessful()
    
    if success:
        print("\n🎉 ALL TESTS PASSED! Ready for GitHub deployment.")
    else:
        print("\n❌ SOME TESTS FAILED. Please fix before deployment.")
    
    return success

def run_specific_test_module(module_name):
    """
    Run tests from a specific module
    """
    print(f"Running tests for module: {module_name}")
    print("-" * 40)
    
    # Import the specific test module
    test_module = __import__(f'tests.{module_name}', fromlist=['Test'])
    
    # Find all test classes in the module
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for attr_name in dir(test_module):
        attr = getattr(test_module, attr_name)
        if isinstance(attr, type) and issubclass(attr, unittest.TestCase):
            suite.addTests(loader.loadTestsFromTestCase(attr))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        # Run specific test module
        module_name = sys.argv[1]
        if module_name.startswith('test_') and module_name.endswith('.py'):
            module_name = module_name[:-3]  # Remove .py extension
        success = run_specific_test_module(module_name)
    else:
        # Run all tests
        success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)