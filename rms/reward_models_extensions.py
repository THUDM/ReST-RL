# other reward models, for test only

import torch
from typing import Union
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch.nn.functional as F
from rms.reward_models import BaseRM


# outcome reward models

class SkyworkRM(BaseRM):
    """
    Skywork reward model series trained on the decontaminated version of the original Skywork Reward Preference dataset
    Input: str
    Output: scalar
    """

    def __init__(self, backend: str = None, rm_type: str = 'orm', max_length: int = 1024,
                 device: Union[str, torch.device] = 'cuda'):
        super().__init__(backend, rm_type)
        self.device = device
        self.rm_class = "skywork"
        self.max_length = max_length
        self.config = AutoConfig.from_pretrained(self.backend, trust_remote_code=True)
        self.tokenizer = None
        self.rm = None
        self.install()
        assert self.rm is not None, "The rm has not been correctly installed\n"
        assert self.tokenizer is not None, "The tokenizer has not been correctly installed\n"

    def install(self):
        self.rm = AutoModelForSequenceClassification.from_pretrained(
            self.backend,
            torch_dtype=self.config.torch_dtype,
            num_labels=1,
            trust_remote_code=True,
        ).to(self.device, dtype=self.config.torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backend, trust_remote_code=True)

    def eval(self, q: str, state: str) -> float:
        conv = [{"role": "user", "content": q}, {"role": "assistant", "content": state}]
        conv_tokenized = self.tokenizer.apply_chat_template(conv,
                                                            tokenize=True,
                                                            return_tensors="pt",
                                                            padding=True,
                                                            max_length=self.max_length,
                                                            truncation=True,
                                                            return_dict=True,
                                                            )
        inputs = {
            "input_ids": conv_tokenized['input_ids'].to(self.device),
            "attention_mask": conv_tokenized['attention_mask'].to(self.device)
        }
        with torch.no_grad():
            score = self.rm(**inputs).logits[0][0].item()
            return score

    def eval_batch(self, q: str, states: list[str]) -> list[float]:
        all_scores = []
        batch_size = 5
        for i in range(0, len(states), batch_size):
            batch_states = states[i:i + batch_size]
            convs = [[{"role": "user", "content": q}, {"role": "assistant", "content": state}] for state in
                     batch_states]
            convs_tokenized = self.tokenizer.apply_chat_template(convs,
                                                                 tokenize=True,
                                                                 return_tensors="pt",
                                                                 padding=True,
                                                                 max_length=self.max_length,
                                                                 truncation=True,
                                                                 return_dict=True,
                                                                 )
            inputs = {
                "input_ids": convs_tokenized['input_ids'].to(self.device),
                "attention_mask": convs_tokenized['attention_mask'].to(self.device)
            }
            with torch.no_grad():
                scores = self.rm(**inputs).logits[:, 0].tolist()
                all_scores.extend(scores)
        return all_scores


class InternRM(BaseRM):
    """
    InternLM reward models trained on the foundation of InternLM2-Chat-1.8B-SFT
    Input: str
    Output: scalar
    """

    def __init__(self, backend: str = None, rm_type: str = 'orm', max_length: int = 1024,
                 device: Union[str, torch.device] = 'cuda'):
        super().__init__(backend, rm_type)
        self.device = device
        self.rm_class = "intern"
        self.max_length = max_length
        self.config = AutoConfig.from_pretrained(self.backend, trust_remote_code=True)
        self.tokenizer = None
        self.rm = None
        self.install()
        assert self.rm is not None, "The rm has not been correctly installed\n"
        assert self.tokenizer is not None, "The tokenizer has not been correctly installed\n"

    def install(self):
        self.rm = AutoModel.from_pretrained(
            self.backend,
            torch_dtype=self.config.torch_dtype,
            trust_remote_code=True,
        ).to(self.device, dtype=self.config.torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backend, trust_remote_code=True)

    def eval(self, q: str, state: str) -> float:
        conv = [{"role": "user", "content": q}, {"role": "assistant", "content": state}]
        score = self.rm.get_score(self.tokenizer, conv)
        return score

    def eval_batch(self, q: str, states: list[str]) -> list[float]:
        all_scores = []
        batch_size = 5
        for i in range(0, len(states), batch_size):
            batch_states = states[i:i + batch_size]
            convs = [[{"role": "user", "content": q}, {"role": "assistant", "content": state}] for state in
                     batch_states]
            scores = self.rm.get_scores(self.tokenizer, convs)
            all_scores.extend(scores)
        return all_scores


class GeneralizableRM(BaseRM):
    """
    Reward models of "Regularizing Hidden States Enables Learning Generalizable Reward Model for LLMs"
    Input: str
    Output: scalar
    """

    def __init__(self, backend: str = None, rm_type: str = 'orm', max_length: int = 1024,
                 device: Union[str, torch.device] = 'cuda'):
        super().__init__(backend, rm_type)
        self.device = device
        self.rm_class = "generalizable"
        self.max_length = max_length
        self.config = AutoConfig.from_pretrained(self.backend, trust_remote_code=True)
        self.tokenizer = None
        self.rm = None
        self.install()
        assert self.rm is not None, "The rm has not been correctly installed\n"
        assert self.tokenizer is not None, "The tokenizer has not been correctly installed\n"

    def install(self):
        self.rm = AutoModelForSequenceClassification.from_pretrained(
            self.backend,
            torch_dtype=self.config.torch_dtype,
            trust_remote_code=True,
        ).to(self.device, dtype=self.config.torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backend, trust_remote_code=True)

    def eval(self, q: str, state: str) -> float:
        conv = [{"role": "user", "content": q}, {"role": "assistant", "content": state}]
        conv_tokenized = self.tokenizer.apply_chat_template(conv,
                                                            tokenize=True,
                                                            return_tensors="pt",
                                                            padding=True,
                                                            max_length=self.max_length,
                                                            truncation=True,
                                                            return_dict=True,
                                                            )
        inputs = {
            "input_ids": conv_tokenized['input_ids'].to(self.device),
            "attention_mask": conv_tokenized['attention_mask'].to(self.device)
        }

        with torch.no_grad():
            score = self.rm(inputs["input_ids"][0].view(1, -1), attention_mask=inputs["attention_mask"][0].view(1, -1))[
                0].cpu().detach().item()
            return score

    def eval_batch(self, q: str, states: list[str]) -> list[float]:
        all_scores = []
        batch_size = 5
        for i in range(0, len(states), batch_size):
            batch_states = states[i:i + batch_size]
            convs = [[{"role": "user", "content": q}, {"role": "assistant", "content": state}] for state in
                     batch_states]
            convs_tokenized = self.tokenizer.apply_chat_template(convs,
                                                                 tokenize=True,
                                                                 return_tensors="pt",
                                                                 padding=True,
                                                                 max_length=self.max_length,
                                                                 truncation=True,
                                                                 return_dict=True,
                                                                 )
            inputs = {
                "input_ids": convs_tokenized['input_ids'].to(self.device),
                "attention_mask": convs_tokenized['attention_mask'].to(self.device)
            }
            with torch.no_grad():
                scores = self.rm(inputs["input_ids"], attention_mask=inputs["attention_mask"]).squeeze(dim=1).tolist()
                all_scores.extend(scores)
        return all_scores


# process reward models

class QwenRM(BaseRM):
    """
    Process reward models based on Qwen2.5, trained by the Qwen team using LLM-as-a-judge and human annotation methods
    Input: str
    Output: prob, ranged between 0 and 1
    """

    def __init__(self, backend: str = None, rm_type: str = 'prm', max_length: int = 1024,
                 device: Union[str, torch.device] = 'cuda'):
        super().__init__(backend, rm_type)
        self.device = device
        self.rm_class = "qwen"
        self.max_length = max_length
        self.config = AutoConfig.from_pretrained(self.backend, trust_remote_code=True)
        self.tokenizer = None
        self.rm = None
        self.install()
        assert self.rm is not None, "The rm has not been correctly installed\n"
        assert self.tokenizer is not None, "The tokenizer has not been correctly installed\n"

    def install(self):
        self.rm = AutoModel.from_pretrained(
            self.backend,
            torch_dtype=self.config.torch_dtype,
            trust_remote_code=True,
        ).to(self.device, dtype=self.config.torch_dtype).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.backend, trust_remote_code=True)
        if self.tokenizer.truncation_side != "left":
            self.tokenizer.truncation_side = "left"
            print("Warning: setting tokenizer truncation_side to 'left'\n")

    @staticmethod
    def make_step_rewards(logits, token_masks):
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels

        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]  # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]  # valid_tokens, num_labels
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res

    def eval(self, q: str, state: str) -> float:
        messages = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": "<extra_0>".join(state.rstrip().split("\n")) + "<extra_0>"},
        ]
        conversation_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        input_ids = self.tokenizer.encode(
            conversation_str,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        ).to(self.device)

        outputs = self.rm(input_ids=input_ids)
        step_sep_id = self.tokenizer.encode("<extra_0>")[0]
        token_masks = (input_ids == step_sep_id)
        step_rewards = self.make_step_rewards(outputs[0], token_masks)[0]
        score = min(step_rewards)
        return score

    def eval_batch(self, q: str, states: list[str]) -> list[float]:
        all_scores = [self.eval(q, state) for state in states]
        return all_scores


class RLHFlowRM(BaseRM):
    """
    Process reward model trained with Math-Shepherd style annotation, from the project RLHFlow/RLHF-Reward-Modeling
    Input: str
    Output: prob, ranged between 0 and 1
    """

    def __init__(self, backend: str = None, rm_type: str = 'prm', max_length: int = 1024,
                 device: Union[str, torch.device] = 'cuda'):
        super().__init__(backend, rm_type)
        self.device = device
        self.rm_class = "rlhflow"
        self.max_length = max_length
        self.config = AutoConfig.from_pretrained(self.backend, trust_remote_code=True)
        self.tokenizer = None
        self.rm = None
        self.install()
        assert self.rm is not None, "The rm has not been correctly installed\n"
        assert self.tokenizer is not None, "The tokenizer has not been correctly installed\n"

    def install(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.backend, trust_remote_code=True)
        self.rm = AutoModelForCausalLM.from_pretrained(
            self.backend,
            torch_dtype=self.config.torch_dtype,
            trust_remote_code=True,
            ).to(self.device, dtype=self.config.torch_dtype).eval()

        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.rm.config.pad_token_id = self.rm.config.eos_token_id
        if self.tokenizer.truncation_side != "left":
            self.tokenizer.truncation_side = "left"
            print("Warning: setting tokenizer truncation_side to 'left'\n")

    def eval(self, q: str, state: str) -> float:
        single_step_score = []
        conversation = []
        plus_tag_id = self.tokenizer.encode('+')[-1]
        minus_tag_id = self.tokenizer.encode('-')[-1]
        candidate_tokens = [plus_tag_id, minus_tag_id]
        ans_list = state.rstrip().split("\n")
        for k in range(len(ans_list)):
            if k == 0:
                text = q + " " + ans_list[0]
            else:
                text = ans_list[k]
            conversation.append({"content": text, "role": "user"})
            conversation.append({"content": "+", "role": "assistant"})

            input_ids = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                logits = self.rm(input_ids).logits[:, -3, candidate_tokens]  # simple version, the +/- is predicted by the '-3' position
                scores = logits.softmax(dim=-1)[:, 0]  # 0 means the prob of + (1 mean -)
                # print(scores)
                single_step_score.append(scores[0].detach().to('cpu', dtype=torch.float32).item())

        score = min(single_step_score)
        return score

    def eval_batch(self, q: str, states: list[str]) -> list[float]:
        all_scores = [self.eval(q, state) for state in states]
        return all_scores
