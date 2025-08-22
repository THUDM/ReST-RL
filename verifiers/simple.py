from verifiers.base import Verifier
from verifiers.test_code import *
import concurrent.futures as cfuts
from tqdm import tqdm
from verifiers.execution import *


class BigCodeBenchVerifier(Verifier):
    def __init__(self, domain: str = 'BigCodeBench', reference: str = None, low_b: float = 0):
        super().__init__(domain, reference, low_b)
        self.var_dict = {}
        with time_limit(60):
            exec(self.reference, self.var_dict)
        self.test_case = self.var_dict["TestCases"]
        assert issubclass(self.test_case, unittest.TestCase), "The testcase must be a subclass of unittest.TestCase\n"
        self.test_code = test_code_unittest

    def verify(self, completions: list[str], initial_state: str = "") -> list[float]:
        results = []
        with cfuts.ProcessPoolExecutor(max_workers=8) as executor:
            futures = []
            for idx, completion in enumerate(completions):
                test_program = (
                        completion + '\n'
                        + self.test_code
                )
                futures.append(
                    executor.submit(get_bcb_score, test_program, self.reference, timeout=60, completion_id=idx))

            for future in tqdm(cfuts.as_completed(futures), total=len(futures)):
                result = future.result()
                result['score'] = result['score'] + self.low_b
                results.append(result)

        sorted_results = sorted(results, key=lambda x: x['completion_id'])
        return [each['score'] for each in sorted_results]


class DS1000Verifier(Verifier):
    def __init__(self, domain: str = 'DS1000', reference: str = None, low_b: float = 0):
        super().__init__(domain, reference, low_b)

    def verify(self, completions: list[str], initial_state: str = "") -> list[float]:
        results = []
        with cfuts.ProcessPoolExecutor(max_workers=8) as executor:
            futures = []
            for idx, completion in enumerate(completions):
                test_program = (
                        self.reference + '\n'
                        + f'code = {repr(completion.split(initial_state)[-1])}\n'
                        + 'test_execution(code)\n'
                        + ('test_string(code)\n' if 'test_string(' in self.reference else '\n')
                )
                futures.append(
                    executor.submit(check_ds_correctness, test_program, timeout=60, completion_id=idx))

            for future in tqdm(cfuts.as_completed(futures), total=len(futures)):
                result = future.result()
                result['score'] = 1.0 + self.low_b if result['passed'] else self.low_b
                results.append(result)

        sorted_results = sorted(results, key=lambda x: x['completion_id'])
        return [each['score'] for each in sorted_results]


class APPSVerifier(Verifier):
    def __init__(self, domain: str = 'APPS', reference: dict = None, low_b: float = 0):
        super().__init__(domain, reference, low_b)

    def verify(self, completions: list[str], initial_state: str = "") -> list[float]:
        results = []
        with cfuts.ProcessPoolExecutor(max_workers=8) as executor:
            futures = []
            for idx, completion in enumerate(completions):
                futures.append(
                    executor.submit(check_apps_correctness, self.reference, completion, timeout=60, completion_id=idx))

            for future in tqdm(cfuts.as_completed(futures), total=len(futures)):
                result = future.result()
                result['score'] = result['score'] + self.low_b
                results.append(result)

        sorted_results = sorted(results, key=lambda x: x['completion_id'])
        return [each['score'] for each in sorted_results]
