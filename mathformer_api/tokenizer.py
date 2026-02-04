import os
import json
from typing import List, Dict, Union, Optional
import torch


class MathTokenizer:

    def __init__(self, model_max_length: int = 64):
        self.chars = [
            "<pad>", "<s>", "</s>", "<unk>",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "+", "-", "*", "/", "=", ".", "(", ")", "^", "%", " ",
            "Q", "R",
        ]
        self.token_to_id = {c: i for i, c in enumerate(self.chars)}
        self.id_to_token = {i: c for i, c in enumerate(self.chars)}
        self.pad_token_id = self.token_to_id["<pad>"]
        self.eos_token_id = self.token_to_id["</s>"]
        self.bos_token_id = self.token_to_id["<s>"]
        self.unk_token_id = self.token_to_id["<unk>"]
        self.padding_side = "left"
        self.model_max_length = model_max_length

    @classmethod
    def from_pretrained(cls, path: str) -> "MathTokenizer":
        config_path = os.path.join(path, "tokenizer_config.json")
        vocab_path = os.path.join(path, "vocab.json")

        tokenizer = cls()
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                tokenizer.model_max_length = config.get("model_max_length", 64)
                tokenizer.padding_side = config.get("padding_side", "left")

        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab = json.load(f)
                tokenizer.token_to_id = vocab
                tokenizer.id_to_token = {int(v): k for k, v in vocab.items()}

        return tokenizer

    def __call__(
        self,
        texts: Union[str, List[str]],
        return_tensors: Optional[str] = None,
        padding: bool = True,
    ) -> Dict:
        if isinstance(texts, str):
            texts = [texts]

        input_ids_list = []
        attention_mask_list = []

        for text in texts:
            ids = [self.token_to_id.get(c, self.unk_token_id) for c in text]
            input_ids_list.append(ids)
            attention_mask_list.append([1] * len(ids))

        if return_tensors == "pt":
            max_len = max(len(x) for x in input_ids_list)
            padded_ids = []
            padded_mask = []

            for ids, mask in zip(input_ids_list, attention_mask_list):
                pad_len = max_len - len(ids)
                if self.padding_side == "left":
                    ids = [self.pad_token_id] * pad_len + ids
                    mask = [0] * pad_len + mask
                else:
                    ids = ids + [self.pad_token_id] * pad_len
                    mask = mask + [0] * pad_len
                padded_ids.append(ids)
                padded_mask.append(mask)

            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
            }

        return {"input_ids": input_ids_list, "attention_mask": attention_mask_list}

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = False) -> str:
        result = ""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        for idx in token_ids:
            char = self.id_to_token.get(idx, "<unk>")
            if skip_special_tokens and char in ["<pad>", "<s>", "</s>"]:
                continue
            result += char
        return result

    def batch_decode(self, sequences: List, skip_special_tokens: bool = False) -> List[str]:
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]

    def __len__(self) -> int:
        return len(self.chars)
