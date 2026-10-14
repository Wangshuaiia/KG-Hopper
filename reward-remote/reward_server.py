"""Remote reward for OpenRLHF (--remote_rm_url http://localhost:1278/get_reward).

Scores with kghopper.reward.RewardFn, the same reward the local trainer uses:
    python reward-remote/reward_server.py --config configs/local_cpu.yaml --data_path data/synthetic/rl.jsonl --port 1278

POST /get_reward {"query": [decoded prompt+response, ...]} -> {"rewards": [float, ...]}
Gold answers are looked up by the exact prompt the sample produces with protocol.build_prompt.
"""
import argparse
import json
import logging
import os
import re
import sys
import threading

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kghopper import protocol  # noqa: E402
from kghopper.config import add_config_args, load_config  # noqa: E402
from kghopper.data import load_samples  # noqa: E402
from kghopper.judges import normalize_answer  # noqa: E402
from kghopper.reward import RewardFn  # noqa: E402
from kghopper.rollout import make_prompt  # noqa: E402

log = logging.getLogger("reward_server")
SPECIAL_TOKEN = re.compile(r"<\|[^|>]{1,40}\|>")  # <|im_end|>, <|endoftext|>, <|eot_id|>, ...


class RewardRequest(BaseModel):
    query: list[str]


class RewardService:
    def __init__(self, cfg, data_path, log_file=None):
        self.reward_fn = RewardFn(cfg.reward)
        self.by_prompt, self.by_question = {}, {}
        for s in load_samples(data_path):
            self.by_prompt[make_prompt(s)] = s
            self.by_question[normalize_answer(s["question"])] = s
        self.log_file = log_file
        self._lock = threading.Lock()
        log.info("loaded %d training samples from %s", len(self.by_prompt), data_path)

    def lookup(self, prompt):
        if prompt in self.by_prompt:
            return self.by_prompt[prompt]
        try:
            question, _, _ = protocol.parse_prompt(prompt)
        except ValueError:
            return None
        return self.by_question.get(normalize_answer(question))

    def score(self, query):
        text = SPECIAL_TOKEN.sub("", query)
        prompt, response = protocol.split_prompt_response(text)
        sample = self.lookup(prompt) if prompt else None
        if sample is None:
            log.warning("no gold answer for query: %s", text[:200])
            return 0.0, {"error": "unknown prompt"}
        result = self.reward_fn.score_text(sample, response)
        return result["reward"], {"question": sample["question"], "response": response, **result}

    def get_rewards(self, queries):
        rewards, records = [], []
        for q in queries:
            r, rec = self.score(q)
            rewards.append(r)
            records.append(rec)
        if self.log_file:
            with self._lock, open(self.log_file, "a", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rewards


def main():
    parser = add_config_args(argparse.ArgumentParser())
    parser.add_argument("--data_path", required=True, help="training prompts (kghopper jsonl or a data/test_set name)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1278)
    parser.add_argument("--log_file", default=None, help="append scored rollouts here (jsonl)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    service = RewardService(load_config(args.config, args.set), args.data_path, args.log_file)
    app = FastAPI()

    @app.post("/get_reward")
    def get_reward(req: RewardRequest):  # sync: FastAPI runs it in a worker thread, judges may block
        return {"rewards": service.get_rewards(req.query)}

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
