import os
from abc import ABC
import ast
from utils.json_operator import *


class Reader(ABC):
    def __init__(self, directory):
        self.directory = directory
        self.datas = []
        self.n_data = 0

    def read(self):
        raise NotImplementedError("The method 'read' must be implemented for a reader\n")

    def get_data(self, idx):
        raise NotImplementedError("The method 'get_data' must be implemented for a reader\n")


class BigCodeBenchReader(Reader):
    def __init__(self, directory="data/BigCodeBench", file="data.json"):
        super().__init__(directory)
        self.file = file
        self.read()

    def read(self):
        print("Reading data from {}...\n".format(os.path.join(self.directory, self.file)))
        self.datas = read_json(os.path.join(self.directory, self.file))
        self.n_data = len(self.datas)
        print(f"Number of data read: {self.n_data}\n")

    def get_data(self, idx):
        assert idx < self.n_data, "Index out of range\n"
        item = self.datas[idx]
        item_use = {
            "id": item["task_id"],
            "prompt": item["instruct_prompt"],
            "code": item["code_prompt"],
            "test": item["test"],
        }
        return item_use

    def get_n_data(self):
        return self.n_data


class DS1000Reader(Reader):
    def __init__(self, directory="data/DS1000", file="data.jsonl"):
        super().__init__(directory)
        self.file = file
        self.read()

    def read(self):
        print("Reading data from {}...\n".format(os.path.join(self.directory, self.file)))
        self.datas = read_json(os.path.join(self.directory, self.file))
        self.n_data = len(self.datas)
        print(f"Number of data read: {self.n_data}\n")

    def get_data(self, idx):
        assert idx < self.n_data, "Index out of range\n"
        item = self.datas[idx]
        item_use = {
            "id": item['metadata']["problem_id"],
            "prompt": item["prompt"],
            "code": item["code"],
            "test": item["code_context"],
        }
        return item_use

    def get_n_data(self):
        return self.n_data


class APPSReader(Reader):
    def __init__(self, directory="data/APPS", file="data_with_test.jsonl"):
        super().__init__(directory)
        self.file = file
        self.read()

    def read(self):
        print("Reading data from {}...\n".format(os.path.join(self.directory, self.file)))
        self.datas = read_json(os.path.join(self.directory, self.file))
        self.n_data = len(self.datas)
        print(f"Number of data read: {self.n_data}\n")

    def get_data(self, idx):
        assert idx < self.n_data, "Index out of range\n"
        item = self.datas[idx]
        success = True
        try:
            test = ast.literal_eval(item["test"])
        except Exception as e:
            print(f"Failed to parse test with error: {e}, trying again with json.loads\n")
            try:
                test = json.loads(item["test"])
            except Exception as e:
                print(f"Failed to parse test with error: {e}, setting test to empty\n")
                test = {}
                success = False
        item_use = {
            "id": item["id"],
            "prompt": item["prompt"],
            "code": item["code"],
            "test": test,
        }
        return item_use, success

    def get_n_data(self):
        return self.n_data
