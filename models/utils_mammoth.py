import os
import io
import json
import os.path


def read_json(source):
    json_list = []
    if not os.path.exists(source):
        return json_list
    with open(source, 'r', encoding='utf-8') as f:
        for line in f:
            json_list.append(json.loads(line))
    return json_list


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


PROMPT_TEMPLATE = [
    {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{prompt}\n\n### Input:\n{input}\n\n### Response:\n"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{prompt}\n\n### Response:\n"
        ),
    },
    {
        "prompt_input": (
            "You are supposed to follow an instruction, and then the input to generate proper response.\n\n"
            "### Instruction:\n{prompt}\n\n### Input:\n{input}\n\n### Response:\n"
        ),
        "prompt_no_input": (
            "You are supposed to follow an instruction to generate proper response."
            "### Instruction:\n{prompt}\n\n### Response:\n"
        ),
    },
    {
        "prompt_input": (
            "Please follow the instruction and input to give a response.\n\n"
            "### Instruction:\n{prompt}\n\n### Input:\n{input}\n\n### Response:\n"
        ),
        "prompt_no_input": (
            "Please follow the instruction to give a response."
            "### Instruction:\n{prompt}\n\n### Response:\n"
        ),
    },
    {
        "prompt_input": (
            "You are an expert, please listen to human instruction and input to generate the response.\n\n"
            "### Instruction:\n{prompt}\n\n### Input:\n{input}\n\n### Response:\n"
        ),
        "prompt_no_input": (
            "You are an expert, please listen to human instruction to generate the response.\n\n"
            "### Instruction:\n{prompt}\n\n### Response:\n"
        ),
    },
    {
        "prompt_input": (
            "Let's follow the instruction to respond to an input.\n\n"
            "### Instruction:\n{prompt}\n\n### Input:\n{input}\n\n### Response:\n"
        ),
        "prompt_no_input": (
            "Let's follow the instruction to generate a response.\n\n"
            "### Instruction:\n{prompt}\n\n### Response:\n"
        ),
    },
    {
        "prompt_input": (
            "The instruction is a description of the task. You need to follow that and respond to the paired input.\n\n"
            "### Instruction:\n{prompt}\n\n### Input:\n{input}\n\n### Response:\n"
        ),
        "prompt_no_input": (
            "The instruction is a description of the task. You need to follow that and respond.\n\n"
            "### Instruction:\n{prompt}\n\n### Response:\n"
        ),
    },
    {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "Instruction:\n{prompt}\n\nInput:\n{input}\n\nResponse:\n"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "Instruction:\n{prompt}\n\nResponse:\n"
        ),
    },
    {
        "prompt_input": (
            "You are supposed to follow an instruction, and then the input to generate proper response.\n\n"
            "#Instruction:\n{prompt}\n\nInput:\n{input}\n\nResponse:\n"
        ),
        "prompt_no_input": (
            "You are supposed to follow an instruction to generate proper response."
            "Instruction:\n{prompt}\n\nResponse:\n"
        ),
    },
    {
        "prompt_input": (
            "Please follow the instruction and input to give a response.\n\n"
            "Instruction:\n{prompt}\n\nInput:\n{input}\n\nResponse:\n"
        ),
        "prompt_no_input": (
            "Please follow the instruction to give a response."
            "Instruction:\n{prompt}\n\nResponse:\n"
        ),
    },
    {
        "prompt_input": (
            "You are an expert, please listen to human instruction and input to generate the response.\n\n"
            "Instruction:\n{prompt}\n\nInput:\n{input}\n\nResponse:\n"
        ),
        "prompt_no_input": (
            "You are an expert, please listen to human instruction to generate the response.\n\n"
            "Instruction:\n{prompt}\n\nResponse:\n"
        ),
    },
    {
        "prompt_input": (
            "Let's follow the instruction to respond to an input.\n\n"
            "Instruction:\n{prompt}\n\nInput:\n{input}\n\nResponse:\n"
        ),
        "prompt_no_input": (
            "Let's follow the instruction to generate a response.\n\n"
            "Instruction:\n{prompt}\n\nResponse:\n"
        ),
    },
    {
        "prompt_input": (
            "The instruction is a description of the task. You need to follow that and respond to the paired input.\n\n"
            "Instruction:\n{prompt}\n\nInput:\n{input}\n\nResponse:\n"
        ),
        "prompt_no_input": (
            "The instruction is a description of the task. You need to follow that and respond.\n\n"
            "Instruction:\n{prompt}\n\nResponse:\n"
        ),
    },
]

PROMPT_TEMPLATE_CODE = [
    {
        "prompt_input": (
            "Below is an instruction that describes a coding task, paired with an input that provides further context. "
            "Write a response that appropriately completes the required code.\n\n"
            "### Instruction:\n{prompt}\n\n### Input:\n{input}\n\n### Response:\n"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a coding task. "
            "Write a response that appropriately completes the required code.\n\n"
            "### Instruction:\n{prompt}\n\n### Response:\n"
        ),
    },
    {
        "prompt_input": (
            "You are supposed to follow an coding instruction, and then the input to generate proper response.\n\n"
            "### Instruction:\n{prompt}\n\n### Input:\n{input}\n\n### Response:\n"
        ),
        "prompt_no_input": (
            "You are supposed to follow an coding instruction to generate proper response."
            "### Instruction:\n{prompt}\n\n### Response:\n"
        ),
    },
    {
        "prompt_input": (
            "Please follow the coding instruction and input to give a piece of code that completes the task appropriately.\n\n"
            "### Instruction:\n{prompt}\n\n### Input:\n{input}\n\n### Response:\n"
        ),
        "prompt_no_input": (
            "Please follow the instruction to give a piece of code that completes the task appropriately."
            "### Instruction:\n{prompt}\n\n### Response:\n"
        ),
    },
    {
        "prompt_input": (
            "You are an coding expert, please listen to human instruction and input to generate a piece of code that completes the task appropriately.\n\n"
            "### Instruction:\n{prompt}\n\n### Input:\n{input}\n\n### Response:\n"
        ),
        "prompt_no_input": (
            "You are an expert, please listen to human instruction to generate a piece of code that completes the task appropriately.\n\n"
            "### Instruction:\n{prompt}\n\n### Response:\n"
        ),
    },
    {
        "prompt_input": (
            "The instruction is a description of a coding task. You need to follow that and respond to the paired input.\n\n"
            "### Instruction:\n{prompt}\n\n### Input:\n{input}\n\n### Response:\n"
        ),
        "prompt_no_input": (
            "The instruction is a description of a coding task. You need to follow that and respond.\n\n"
            "### Instruction:\n{prompt}\n\n### Response:\n"
        ),
    },
    {
        "prompt_input": (
            "Below is an instruction that describes a coding task, paired with an input that provides further context. "
            "Write a response that appropriately completes the required code, with no further explanation.\n\n"
            "### Instruction:\n{prompt}\n\n### Input:\n{input}\n\n### Response:\n"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a coding task. "
            "Write a response that appropriately completes the required code, with no further explanation.\n\n"
            "### Instruction:\n{prompt}\n\n### Response:\n"
        ),
    },
    {
        "prompt_input": (
            "You are supposed to follow an coding instruction, and then the input to generate proper response, with no further explanation.\n\n"
            "### Instruction:\n{prompt}\n\n### Input:\n{input}\n\n### Response:\n"
        ),
        "prompt_no_input": (
            "You are supposed to follow an coding instruction to generate proper response, with no further explanation."
            "### Instruction:\n{prompt}\n\n### Response:\n"
        ),
    },
    {
        "prompt_input": (
            "Please follow the coding instruction and input to give a piece of code that completes the task appropriately, with no further explanation.\n\n"
            "### Instruction:\n{prompt}\n\n### Input:\n{input}\n\n### Response:\n"
        ),
        "prompt_no_input": (
            "Please follow the instruction to give a piece of code that completes the task appropriately, with no further explanation."
            "### Instruction:\n{prompt}\n\n### Response:\n"
        ),
    },
    {
        "prompt_input": (
            "You are an coding expert, please listen to human instruction and input to generate a piece of code that completes the task appropriately, with no further explanation.\n\n"
            "### Instruction:\n{prompt}\n\n### Input:\n{input}\n\n### Response:\n"
        ),
        "prompt_no_input": (
            "You are an expert, please listen to human instruction to generate a piece of code that completes the task appropriately, with no further explanation.\n\n"
            "### Instruction:\n{prompt}\n\n### Response:\n"
        ),
    },
    {
        "prompt_input": (
            "The instruction is a description of a coding task. You need to follow that and respond to the paired input, with no further explanation of your code.\n\n"
            "### Instruction:\n{prompt}\n\n### Input:\n{input}\n\n### Response:\n"
        ),
        "prompt_no_input": (
            "The instruction is a description of a coding task. You need to follow that and respond, with no further explanation of your code.\n\n"
            "### Instruction:\n{prompt}\n\n### Response:\n"
        ),
    },
]

PROMPT_TEMPLATE_SINGLE = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{prompt}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{prompt}\n\n### Response:\n"
    ),
}
