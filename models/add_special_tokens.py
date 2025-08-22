import argparse
import transformers
from typing import Dict
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def add_special_tokens(args):
    model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True).to('cuda', dtype=model_config.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    need_save = False

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    if special_tokens_dict:
        need_save = True
        print("Before adding, tokenizer length: ", len(tokenizer), '\n')
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )
        print("After adding, tokenizer length: ", len(tokenizer), '\n')
        print(f"Special token ids:\npad_token_id: {tokenizer.pad_token_id}\neos_token_id: {tokenizer.eos_token_id}\nbos_token_id: {tokenizer.bos_token_id}\nunk_token_id: {tokenizer.unk_token_id}\n")

    # Initialize model.config attributes if they do not exist or are None
    if not hasattr(model.config, 'pad_token_id') or model.config.pad_token_id is None:
        need_save = True
        model.config.pad_token_id = tokenizer.pad_token_id
        print("Added pad_token_id to model config...\n")
    if not hasattr(model.config, 'eos_token_id') or model.config.eos_token_id is None:
        need_save = True
        model.config.eos_token_id = tokenizer.eos_token_id
        print("Added eos_token_id to model config...\n")
    if not hasattr(model.config, 'bos_token_id') or model.config.bos_token_id is None:
        need_save = True
        model.config.bos_token_id = tokenizer.bos_token_id
        print("Added bos_token_id to model config...\n")
    if not hasattr(model.config, 'unk_token_id') or model.config.unk_token_id is None:
        need_save = True
        model.config.unk_token_id = tokenizer.unk_token_id
        print("Added unk_token_id to model config...\n")

    if need_save:
        print("Saving tokenizer and model...\n")
        output_dir = args.output_dir
        tokenizer.save_pretrained(output_dir)
        model.save_pretrained(output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="For adding special tokens for transformer models")
    parser.add_argument('model', type=str, help='Specify the policy path')
    parser.add_argument('output_dir', type=str, help='Directory path to save checkpoints')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    add_special_tokens(args)
