# for applying chat templates
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"


def make_raw_chat_prompt(
        task_prompt: str,
        response_prefix: str,
        tokenizer,
        **kwargs,
) -> str:
    # check if the tokenizer has a tokenizer.chat_template method
    if tokenizer is None or tokenizer.chat_template is None:
        return task_prompt + '\n' + response_prefix

    task_prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": task_prompt},
            {"role": "assistant", "content": response_prefix + _MAGIC_SPLITTER_},
        ],
        tokenize=False,
        **kwargs,
    ).split(_MAGIC_SPLITTER_)[0]

    return task_prompt


def make_raw_chat_prompt_with_context(
        context_prompts: list[str],
        context_responses: list[str],
        task_prompt: str,
        response_prefix: str,
        tokenizer,
        **kwargs,
) -> str:
    # check the length of context_prompt and context_response
    assert len(context_prompts) == len(
        context_responses), "Argument 'context_prompts' and 'context_responses' must have the same length.\n"
    prompt = ""

    # check if the tokenizer has a tokenizer.chat_template method
    if tokenizer is None or tokenizer.chat_template is None:
        for context_prompt, context_response in zip(context_prompts, context_responses):
            prompt += context_prompt + '\n' + context_response + '\n'
        prompt += task_prompt + '\n' + response_prefix
        return prompt

    messages = []
    for context_prompt, context_response in zip(context_prompts, context_responses):
        messages.append({"role": "user", "content": context_prompt})
        messages.append({"role": "assistant", "content": context_response})
    messages.append({"role": "user", "content": task_prompt})
    messages.append({"role": "assistant", "content": response_prefix + _MAGIC_SPLITTER_})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        **kwargs,
    ).split(_MAGIC_SPLITTER_)[0]

    return prompt


def remove_overlap_sample(state: str, sample: str):
    if state in sample:
        if state:
            sample = sample.split(state)[-1]
    else:
        state_list = state.split("\n")
        sample_list = sample.split("\n")
        state_lines = len(state_list)
        sample_lines = len(sample_list)
        state_idx = 0
        sample_idx = 0

        while sample_idx < sample_lines and state_idx < state_lines:
            if sample_list[sample_idx].strip() == "":
                sample_idx += 1
                continue
            while state_idx < state_lines and state_list[state_idx].strip() == "":
                state_idx += 1
            if state_idx >= state_lines:
                break
            else:
                if sample_list[sample_idx] == state_list[state_idx]:
                    sample_idx += 1
                    state_idx += 1
                    continue
                else:
                    break

        if sample_idx >= sample_lines:
            return ""
        else:
            sample = "\n".join(sample_list[sample_idx:])

    return sample
