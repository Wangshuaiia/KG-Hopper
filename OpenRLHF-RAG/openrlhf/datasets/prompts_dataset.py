import os
import sys

from torch.utils.data import Dataset
from tqdm import tqdm

try:  # the shared KG-Hopper core (pip install -e . in the repo root)
    import kghopper  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from kghopper import protocol


def preprocess_data(data, input_template=None, input_key="question", apply_chat_template=None) -> str:
    """Prompt from a kghopper sample ({"idx", "question", "topic_entities", "task", ...}).

    Training, evaluation and the reward server all build prompts with protocol.build_prompt,
    so the chat template is not used here.
    """
    question = data[input_key or "question"]
    prompt = protocol.build_prompt(question, data.get("topic_entities"), data.get("task") or "qa")
    return str(data["idx"]) + "<|idx_prompt_split|>" + prompt


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)

        self.prompts = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            self.prompts.append(preprocess_data(data, input_template, input_key))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]
