test_code_unittest = '''
import unittest
runner = unittest.TextTestRunner()
print('Runner has been built...\\n')
run_result = runner.run(test_suite)
print('Runner result acquired...\\n')
total_tests = run_result.testsRun
failed_tests = len(run_result.failures)
error_tests = len(run_result.errors)
passed_tests = total_tests - failed_tests - error_tests
passed_tests = max(0, passed_tests)
pass_rate = passed_tests / total_tests if total_tests > 0 else 0
'''