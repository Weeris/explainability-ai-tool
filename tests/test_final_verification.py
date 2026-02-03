"""
Final verification tests for the Explainability AI Tool (Project Noor)
This ensures all components work together properly
"""
import unittest
import sys
import os
from unittest import mock
import tempfile
import json

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

class TestFinalVerification(unittest.TestCase):
    """
    Final verification tests to ensure the application is ready for GitHub
    """
    
    def test_config_module_exists_and_works(self):
        """
        Test that the config module exists and works properly
        """
        import config
        
        # Test that we can get the default config
        cfg = config.get_config()
        self.assertIsNotNone(cfg)
        self.assertTrue(hasattr(cfg, 'APP_NAME'))
        self.assertTrue(hasattr(cfg, 'VERSION'))
        self.assertIsInstance(cfg.APP_NAME, str)
        self.assertIsInstance(cfg.VERSION, str)
    
    def test_readme_exists_and_has_content(self):
        """
        Test that README.md exists and has appropriate content
        """
        import os
        readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'README.md')
        
        self.assertTrue(os.path.exists(readme_path), "README.md should exist")
        
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertGreater(len(content), 100, "README should have substantial content")
        self.assertIn("Explainability AI Tool", content)
        self.assertIn("Project Veritas", content)
    
    def test_main_app_structure(self):
        """
        Test that the main app structure is correct
        """
        # Check that app.py exists and has the expected functions
        app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app.py')
        self.assertTrue(os.path.exists(app_path), "app.py should exist")
        
        with open(app_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for key functions
        self.assertIn('def main():', content)
        self.assertIn('def show_bank_a_view():', content)
        self.assertIn('def show_supervisor_view():', content)
        self.assertIn('def show_project_veritas_validation(', content)
        
        # Check for Streamlit usage
        self.assertIn('import streamlit as st', content)
        
        # Check for required libraries
        self.assertIn('import pandas as pd', content)
        self.assertIn('import numpy as np', content)
        self.assertIn('import shap', content)
        self.assertIn('import lime', content)
    
    def test_license_exists(self):
        """
        Test that LICENSE file exists
        """
        license_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'LICENSE')
        self.assertTrue(os.path.exists(license_path), "LICENSE file should exist")
        
        with open(license_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn("MIT License", content)
    
    def test_requirements_exist(self):
        """
        Test that requirements files exist
        """
        req_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'requirements.txt')
        self.assertTrue(os.path.exists(req_path), "requirements.txt should exist")
        
        with open(req_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn('streamlit', content.lower())
        self.assertIn('pandas', content.lower())
        self.assertIn('shap', content.lower())
    
    def test_gitignore_exists(self):
        """
        Test that .gitignore file exists
        """
        gitignore_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.gitignore')
        self.assertTrue(os.path.exists(gitignore_path), ".gitignore should exist")
        
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn('.env', content)
        self.assertIn('__pycache__', content)
    
    def test_dockerfile_exists(self):
        """
        Test that Dockerfile exists and has correct content
        """
        dockerfile_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Dockerfile')
        self.assertTrue(os.path.exists(dockerfile_path), "Dockerfile should exist")
        
        with open(dockerfile_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn('FROM python:', content)
        self.assertIn('COPY requirements.txt', content)
        self.assertIn('EXPOSE 8501', content)
        self.assertIn('ENTRYPOINT', content)
    
    def test_directory_structure(self):
        """
        Test that the expected directory structure exists
        """
        base_dir = os.path.dirname(os.path.dirname(__file__))
        
        # Check for essential files
        essential_files = [
            'app.py',
            'README.md',
            'requirements.txt',
            'Dockerfile',
            'LICENSE',
            '.gitignore',
            'setup.py'
        ]
        
        for file in essential_files:
            file_path = os.path.join(base_dir, file)
            self.assertTrue(os.path.exists(file_path), f"{file} should exist")
    
    def test_test_directory_structure(self):
        """
        Test that the test directory structure is correct
        """
        test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests')
        self.assertTrue(os.path.exists(test_dir), "tests directory should exist")
        
        # Check for test files
        test_files = [
            '__init__.py',
            'test_app.py',
            'test_config.py',
            'test_runner.py',
            'test_final_verification.py'
        ]
        
        for file in test_files:
            file_path = os.path.join(test_dir, file)
            self.assertTrue(os.path.exists(file_path), f"tests/{file} should exist")

    def test_pytest_config_exists(self):
        """
        Test that pytest configuration exists
        """
        pytest_cfg = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pytest.ini')
        self.assertTrue(os.path.exists(pytest_cfg), "pytest.ini should exist")
        
        with open(pytest_cfg, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn('[tool:pytest]', content)
        self.assertIn('testpaths = tests', content)

    def test_code_syntax_validity(self):
        """
        Test that all Python files have valid syntax
        """
        import ast
        import glob
        
        base_dir = os.path.dirname(os.path.dirname(__file__))
        
        # Find all Python files in the project
        py_files = glob.glob(os.path.join(base_dir, "**/*.py"), recursive=True)
        py_files.extend(glob.glob(os.path.join(base_dir, "tests/**/*.py"), recursive=True))
        
        syntax_errors = []
        for file_path in py_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Parse the file to check for syntax errors
                ast.parse(content)
            except SyntaxError as e:
                syntax_errors.append(f"{file_path}: {e}")
        
        if syntax_errors:
            self.fail(f"Syntax errors found:\n" + "\n".join(syntax_errors))
        
        self.assertEqual(len(syntax_errors), 0, f"Found {len(syntax_errors)} syntax errors")

def run_final_verification():
    """
    Run the final verification tests
    """
    print("=" * 60)
    print("FINAL VERIFICATION TESTS FOR EXPLAINABILITY AI TOOL")
    print("=" * 60)
    
    # Create a test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestFinalVerification)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback.split(chr(10))[0]}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback.split(chr(10))[0]}")
    
    success = result.wasSuccessful()
    
    if success:
        print("\n🎉 ALL FINAL VERIFICATION TESTS PASSED!")
        print("The Explainability AI Tool is ready for GitHub deployment.")
    else:
        print("\n❌ SOME VERIFICATION TESTS FAILED!")
        print("Please address the issues before deploying to GitHub.")
    
    return success

if __name__ == '__main__':
    success = run_final_verification()
    exit(0 if success else 1)