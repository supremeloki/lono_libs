import unittest
import os
import io
import sys
import shutil

# Ensure that the lono_libs package is discoverable for imports in test files
# This is crucial when running tests from a root directory outside the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'lono_libs')))

def run_all_tests_and_capture_output():
    """
    Discovers and runs all unittest test cases in the 'test' directory.
    Captures the output to be saved to a file and returns it.
    """
    print("Discovering and running all unit tests for lono_libs...")

    # Change current working directory to 'test' to ensure test discovery works relative to that directory
    # Also cleans up any previous test result directories created by the UnifiedRunner tests
    original_cwd = os.getcwd()
    test_dir = os.path.join(original_cwd, "tests")
    if os.path.exists("test_lono_results_temp"):
        shutil.rmtree("test_lono_results_temp") # Clean up artifacts from previous UnifiedRunner mock runs

    os.chdir(test_dir)

    # Discover all tests in the current directory (which is now 'test') and its subdirectories
    suite = unittest.TestLoader().discover('.')

    # Capture the test output
    buffer = io.StringIO()
    runner = unittest.TextTestRunner(stream=buffer, verbosity=2) # verbosity 2 shows individual test names
    result = runner.run(suite)

    output = buffer.getvalue()

    # Change back to original working directory
    os.chdir(original_cwd)

    return output, result.wasSuccessful()

if __name__ == "__main__":
    test_output, tests_passed = run_all_tests_and_capture_output()

    report_file_name = "test_results_summary.md"
    with open(report_file_name, "w", encoding="utf-8") as f:
        f.write("# LonoLibs Comprehensive Test Results\n\n")
        f.write("This report summarizes the execution of all unit and functional tests for the `lono_libs` package.\n\n")
        f.write(f"**Overall Status:** {'ALL TESTS PASSED' if tests_passed else 'SOME TESTS FAILED'}\n\n")
        f.write("---\n\n")
        f.write("## Detailed Test Output\n\n")
        f.write("```\n")
        f.write(test_output)
        f.write("```\n")
        f.write("\n---\n\n")
        f.write(f"**Summary:** {'All tests ran successfully.' if tests_passed else 'Some tests failed. Please review the detailed output.'}\n")

    print(f"\nComprehensive test execution completed. Results saved to '{report_file_name}'.")
    print(f"Please review '{report_file_name}' for detailed test outcomes.")

    # Also print to console for immediate feedback
    print("\n--- DETAILED TEST CONSOLE OUTPUT ---")
    print(test_output)
    print("------------------------------------")
    print(f"\nOverall Test Status: {'SUCCESS' if tests_passed else 'FAILURE'}")