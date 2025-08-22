import openai
from openai import OpenAI
import http.client
import json
from utils.yaml_operator import *

# set the api config
config_file = "models/config/config.yaml"
api_config = load_yaml(config_file)
api_key = api_config['api_key']
api_base = api_config['url']
default_model = "gpt-4o-mini"
completion_tokens = prompt_tokens = 0

if api_key != "":
    openai.api_key = api_key
    print(f'|api_key|: {api_key}\n')
else:
    print("Warning: OPENAI_API_KEY is not set!\n")

if api_base != "":
    print("Notice: OPENAI_API_BASE is set to {}!\n".format(api_base))
    openai.api_base = api_base

client = OpenAI(api_key=api_key, base_url=api_base)


def get_ca_reply(query, base=default_model) -> str:
    """
    for ChatAnywhere api use
    """
    global completion_tokens, prompt_tokens
    messages = [{"role": "user", "content": query}]
    out = ""
    conn = http.client.HTTPSConnection("api.chatanywhere.tech")
    payload = json.dumps({
        "model": base,
        "messages": messages
    })
    headers = {
        'Authorization': f'Bearer {api_key}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }
    cnt = 3
    while cnt:
        try:
            conn.request("POST", "/v1/chat/completions", payload, headers)
            res = conn.getresponse()
            data = res.read()
            output_json = json.loads(data.decode("utf-8"))
            out = output_json["choices"][0]['message']['content']
            completion_tokens += output_json["usage"]["completion_tokens"]
            prompt_tokens += output_json["usage"]["prompt_tokens"]
            break
        except Exception as e:
            print(f"Error occurred when getting gpt reply!\nError type:{e}\n")
    show_token_usage(base)
    return out


def get_reply(messages, base=default_model, temperature=0.7, max_tokens=1024, n=1, stop=None):
    """
    for general Openai api use
    """
    global completion_tokens, prompt_tokens
    outputs = []
    res = client.chat.completions.create(model=base, messages=messages, temperature=temperature,
                                         max_tokens=max_tokens, n=n, stop=stop)
    # print(f'Got response:{res}\n\n')
    outputs.extend([choice.message.content for choice in res.choices])
    # log completion tokens
    current_completion_tokens = res.usage.completion_tokens
    current_prompt_tokens = res.usage.prompt_tokens
    completion_tokens += current_completion_tokens
    prompt_tokens += current_prompt_tokens
    return outputs, current_prompt_tokens, current_completion_tokens


def gpt_usage(backend=default_model):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    if backend == "gpt-4o-mini":
        cost = completion_tokens / 1000 * 0.00105 + prompt_tokens / 1000 * 0.00105
    else:
        cost = -1
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}


def show_token_usage(base=default_model):
    print('-' * 50)
    print(f"Completion tokens: {completion_tokens}")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Cost: ${gpt_usage(base)['cost']}")
    print('-' * 50)
