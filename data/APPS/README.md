---
annotations_creators: []
language_creators:
- crowdsourced
- expert-generated
language: ["code"]
license:
- mit
multilinguality:
- monolingual
pretty_name: APPS
size_categories:
- unknown
source_datasets: []
task_categories:
- text-generation
task_ids:
- language-modeling
---

# APPS Dataset

## Dataset Description
[APPS](https://arxiv.org/abs/2105.09938) is a benchmark for code generation with 10000 problems. It can be used to evaluate the ability of language models to generate code from natural language specifications.
You can also find **APPS metric** in the hub here [codeparrot/apps_metric](https://huggingface.co/spaces/codeparrot/apps_metric).

## Languages

The dataset contains questions in English and code solutions in Python.

## Dataset Structure

```python
from datasets import load_dataset
load_dataset("codeparrot/apps")

DatasetDict({
    train: Dataset({
        features: ['problem_id', 'question', 'solutions', 'input_output', 'difficulty', 'url', 'starter_code'],
        num_rows: 5000
    })
    test: Dataset({
        features: ['problem_id', 'question', 'solutions', 'input_output', 'difficulty', 'url', 'starter_code'],
        num_rows: 5000
    })
})
```

### How to use it

You can load and iterate through the dataset with the following two lines of code for the train split:

```python
from datasets import load_dataset
import json

ds = load_dataset("codeparrot/apps", split="train")
sample = next(iter(ds))
# non-empty solutions and input_output features can be parsed from text format this way:
sample["solutions"] = json.loads(sample["solutions"])
sample["input_output"] = json.loads(sample["input_output"])
print(sample)

#OUTPUT:
{
 'problem_id': 0,
 'question': 'Polycarp has $n$ different binary words. A word called binary if it contains only characters \'0\' and \'1\'. For example...',
 'solutions': ["for _ in range(int(input())):\n    n = int(input())\n    mass = []\n    zo = 0\n    oz = 0\n    zz = 0\n    oo = 0\n...",...],
 'input_output': {'inputs': ['4\n4\n0001\n1000\n0011\n0111\n3\n010\n101\n0\n2\n00000\n00001\n4\n01\n001\n0001\n00001\n'], 
                  'outputs': ['1\n3 \n-1\n0\n\n2\n1 2 \n']},
 'difficulty': 'interview',
 'url': 'https://codeforces.com/problemset/problem/1259/D',
 'starter_code': ''}
}
```
Each sample consists of a programming problem formulation in English, some ground truth Python solutions, test cases that are defined by their inputs and outputs and function name if provided, as well as some metadata regarding the difficulty level of the problem and its source. 

If a sample has non empty `input_output` feature, you can read it as a dictionary with keys `inputs` and `outputs` and `fn_name` if it exists, and similarily you can parse the solutions into a list of solutions as shown in the code above. 

You can also filter the dataset for the difficulty level: Introductory, Interview and Competition. Just pass the list of difficulties as a list. E.g. if you want the most challenging problems, you need to select the competition level:

```python
ds = load_dataset("codeparrot/apps", split="train", difficulties=["competition"])
print(next(iter(ds))["question"])

#OUTPUT:
"""\
Codefortia is a small island country located somewhere in the West Pacific. It consists of $n$ settlements connected by
...

For each settlement $p = 1, 2, \dots, n$, can you tell what is the minimum time required to travel between the king's residence and the parliament house (located in settlement $p$) after some roads are abandoned?

-----Input-----

The first line of the input contains four integers $n$, $m$, $a$ and $b$ 
...

-----Output-----

Output a single line containing $n$ integers
...

-----Examples-----
Input
5 5 20 25
1 2 25
...

Output
0 25 60 40 20
...
```

### Data Fields

|Field|Type|Description|
|---|---|---|
|problem_id|int|problem id|
|question|string|problem description|
|solutions|string|some python solutions|
|input_output|string|Json string with "inputs" and "outputs" of the test cases, might also include "fn_name" the name of the function|
|difficulty|string|difficulty level of the problem|
|url|string|url of the source of the problem|
|starter_code|string|starter code to include in prompts|

we mention that only few samples have `fn_name` and `starter_code` specified

### Data Splits

The dataset contains a train and test splits with 5000 samples each.

### Dataset Statistics
* 10000 coding problems
* 131777 test cases
* all problems have a least one test case except 195 samples in the train split
* for tests split, the average number of test cases is 21.2
* average length of a problem is 293.2 words
* all files have ground-truth solutions except 1235 samples in the test split

## Dataset Creation

To create the APPS dataset, the authors manually curated problems from open-access sites where programmers share problems with each other, including Codewars, AtCoder, Kattis, and Codeforces. For more details please refer to the original [paper](https://arxiv.org/pdf/2105.09938.pdf).

## Considerations for Using the Data

In [AlphaCode](https://arxiv.org/pdf/2203.07814v1.pdf) the authors found that this dataset can generate many false positives during evaluation, where incorrect submissions are marked as correct due to lack of test coverage.

## Citation Information

```
@article{hendrycksapps2021,
  title={Measuring Coding Challenge Competence With APPS},
  author={Dan Hendrycks and Steven Basart and Saurav Kadavath and Mantas Mazeika and Akul Arora and Ethan Guo and Collin Burns and Samir Puranik and Horace He and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
``` 