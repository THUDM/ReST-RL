from prompts.base import Formatter
from prompts.utils import make_raw_chat_prompt, make_raw_chat_prompt_with_context


class SimpleFormatter(Formatter):
    def __init__(self, domain: str = None, tokenizer=None, **kwargs):
        super().__init__(domain, tokenizer)
        self.kwargs = kwargs

    def format(self, q: str, state: str):
        query = "Please solve this problem step by step.\nQ: " + q
        reply = "A:\n" + state
        prompt = make_raw_chat_prompt(query, reply, self.tokenizer, **self.kwargs)
        return prompt

    def format_for_think(self, q: str, state: str):
        query = "Please solve this problem step by step.\nQ: " + q
        reply = "A:\n" + state
        ask = "This solution is not yet complete. How should this solution be continued?"
        think = "Let's think step by step about how to continue:\n"
        prompt = make_raw_chat_prompt_with_context([query], [reply], ask, think, self.tokenizer, **self.kwargs)
        return prompt

    def format_for_act(self, q: str, state: str, thought: str):
        query = "Please solve this problem step by step.\nQ: " + q
        reply = "A:\n" + state
        ask = "This solution is not yet complete. How should this solution be continued?"
        think = "Let's think step by step about how to continue:\n" + thought
        new_query = "Based on these ideas, please provide a complete solution to the problem."
        context_prompts = [query, ask]
        context_responses = [reply, think]
        prompt = make_raw_chat_prompt_with_context(context_prompts, context_responses, new_query, reply, self.tokenizer, **self.kwargs)
        return prompt

    def format_thought(self, thought: str) -> str:
        return thought

    def format_sample(self, state: str, sample: str):
        return sample


class BigCodeBenchFormatter(Formatter):
    def __init__(self, domain: str = 'BigCodeBench', tokenizer=None, **kwargs):
        super().__init__(domain, tokenizer)
        self.kwargs = kwargs

    def format(self, q: str, state: str):
        query = "Please provide a self-contained Python script that solves the following problem in a markdown code block.\nProblem: " + q
        reply = "Below is a self-contained Python script that solves the problem and realizes requested functionalities:\n```python\n" + state
        prompt = make_raw_chat_prompt(query, reply, self.tokenizer, **self.kwargs)
        return prompt

    def format_for_think(self, q: str, state: str):
        query = "Please provide a self-contained Python script that solves the following problem in a markdown code block.\nProblem: " + q
        reply = "Below is a self-contained Python script that solves the problem and realizes requested functionalities:\n```python\n" + state
        ask = "The script is not yet complete. How should we proceed to complete this script?"
        think = "Let's think step by step about how to continue:\n"
        prompt = make_raw_chat_prompt_with_context([query], [reply], ask, think, self.tokenizer, **self.kwargs)
        return prompt

    def format_for_act(self, q: str, state: str, thought: str):
        query = "Please provide a self-contained Python script that solves the following problem in a markdown code block.\nProblem: " + q
        reply = "Below is a self-contained Python script that solves the problem and realizes requested functionalities:\n```python\n" + state
        ask = "The script is not yet complete. How should we proceed to complete this script?"
        think = "Let's think step by step about how to continue:\n" + thought
        new_query = "Based on these ideas, please provide a complete self-contained Python script to the problem."
        new_reply = "Based on the ideas, below is a complete self-contained Python script that solves the problem and realizes requested functionalities:\n```python\n" + state
        context_prompts = [query, ask]
        context_responses = [reply, think]
        prompt = make_raw_chat_prompt_with_context(context_prompts, context_responses, new_query, new_reply, self.tokenizer, **self.kwargs)
        return prompt

    def format_thought(self, thought: str) -> str:
        return thought

    def format_sample(self, state: str, sample: str):
        sample = sample.split("```")[0]
        return sample


class DS1000Formatter(Formatter):
    def __init__(self, domain: str = 'DS1000', tokenizer=None, **kwargs):
        super().__init__(domain, tokenizer)
        self.kwargs = kwargs

    def format(self, q: str, state: str):
        query = "Please directly complete the required program to solve the following problem.\n" + q
        reply = state
        prompt = make_raw_chat_prompt(query, reply, self.tokenizer, **self.kwargs)
        return prompt

    def format_for_think(self, q: str, state: str):
        query = "Please directly complete the required program to solve the following problem.\n" + q
        reply = state
        ask = "The program is not yet complete. How should we proceed to complete this program?"
        think = "Let's think step by step about how to continue:\n"
        prompt = make_raw_chat_prompt_with_context([query], [reply], ask, think, self.tokenizer, **self.kwargs)
        return prompt

    def format_for_act(self, q: str, state: str, thought: str):
        query = "Please directly complete the required program to solve the following problem.\n" + q
        reply = state
        ask = "The program is not yet complete. How should we proceed to complete this program?"
        think = "Let's think step by step about how to continue:\n" + thought
        new_query = "Based on these ideas, please provide a complete program to the problem.\n"
        new_reply = state
        context_prompts = [query, ask]
        context_responses = [reply, think]
        prompt = make_raw_chat_prompt_with_context(context_prompts, context_responses, new_query, new_reply,
                                                   self.tokenizer, **self.kwargs)
        return prompt

    def format_thought(self, thought: str) -> str:
        return thought

    def format_sample(self, state: str, sample: str):
        sample = sample.split('</code>')[0]
        sample = sample.split('### SOLUTION END')[0]
        sample = sample.split('# SOLUTION END')[0]
        sample = sample.split('### END SOLUTION')[0]
        sample = sample.split('# END SOLUTION')[0]
        sample = sample.split('END SOLUTION')[0]
        sample = sample.split('SOLUTION END')[0]
        return sample


class APPSFormatter(Formatter):
    """
    This formatter must be initialized at each single problem.
    The prompt format depends on whether the problem required a call-based answer or standard-io style answer.
    """
    def __init__(self, domain: str = 'APPS', tokenizer=None, use_call_style: bool = False, **kwargs):
        super().__init__(domain, tokenizer)
        self.use_call_style = use_call_style
        self.kwargs = kwargs

    def format(self, q: str, state: str):
        if not self.use_call_style:
            style = "Standard Input format"
        else:
            style = "Call-Based format"
        query = f"Please provide a self-contained Python script using {style} that solves the following problem in a markdown code block.\nProblem: " + q
        reply = "Below is a self-contained Python script that solves the problem and realizes requested functionalities:\n```python\n" + state
        prompt = make_raw_chat_prompt(query, reply, self.tokenizer, **self.kwargs)
        return prompt

    def format_for_think(self, q: str, state: str):
        if not self.use_call_style:
            style = "Standard Input format"
        else:
            style = "Call-Based format"
        query = f"Please provide a self-contained Python script using {style} that solves the following problem in a markdown code block.\nProblem: " + q
        reply = "Below is a self-contained Python script that solves the problem and realizes requested functionalities:\n```python\n" + state
        ask = "The script is not yet complete. How should we proceed to complete this script?"
        think = "Let's think step by step about how to continue:\n"
        prompt = make_raw_chat_prompt_with_context([query], [reply], ask, think, self.tokenizer, **self.kwargs)
        return prompt

    def format_for_act(self, q: str, state: str, thought: str):
        if not self.use_call_style:
            style = "Standard Input format"
        else:
            style = "Call-Based format"
        query = f"Please provide a self-contained Python script using {style} that solves the following problem in a markdown code block.\nProblem: " + q
        reply = "Below is a self-contained Python script that solves the problem and realizes requested functionalities:\n```python\n" + state
        ask = "The script is not yet complete. How should we proceed to complete this script?"
        think = "Let's think step by step about how to continue:\n" + thought
        new_query = "Based on these ideas, please provide a complete self-contained Python script to the problem."
        new_reply = "Based on the ideas, below is a complete self-contained Python script that solves the problem and realizes requested functionalities:\n```python\n" + state
        context_prompts = [query, ask]
        context_responses = [reply, think]
        prompt = make_raw_chat_prompt_with_context(context_prompts, context_responses, new_query, new_reply,
                                                   self.tokenizer, **self.kwargs)
        return prompt

    def format_thought(self, thought: str) -> str:
        return thought

    def format_sample(self, state: str, sample: str):
        sample = sample.split("```")[0]
        return sample
