"""Reward (paper Sec. III-C) with the components R_search, R_format, R_reason, R_answer.

mode = "paper": R = R_search + R_format + R_reason + R_answer, as written in the paper.
       A wrong answer with two searches (0.8 + 0.5 = 1.3) outscores a correct answer found
       without searching (1.0), so the policy can be paid for searching instead of answering.
mode = "gated" (default): R = R_format + w_answer * R_answer + R_answer * (R_search + R_reason)
       The search and reasoning bonuses are only paid for correct answers. Any correct answer
       (>= w_answer) beats any wrong one (<= R_format), and among correct answers grounded
       multi-hop search scores higher.
"""
from . import protocol
from .judges import AnswerJudge, ReasoningJudge


class RewardFn:
    def __init__(self, reward_cfg):
        c = dict(reward_cfg)
        self.mode = c.get("mode", "gated")
        self.w_answer = c.get("w_answer", 1.0)
        self.format_bonus = c.get("format_bonus", 0.5)
        self.search_per_call = c.get("search_per_call", 0.5)
        self.search_cap = c.get("search_cap", 0.8)
        self.answer_judge = AnswerJudge(c.get("answer_judge", "rule"), c.get("answer_judge_model"), c.get("answer_judge_url"),
                                        max_words=c.get("max_answer_words", 15), rule_first=c.get("answer_judge_rule_first", True))
        self.reason_judge = ReasoningJudge(c.get("reason_judge", "none"), c.get("reason_judge_model"), c.get("reason_judge_url"))
        if self.mode not in ("gated", "paper"):
            raise ValueError(f"unknown reward.mode {self.mode!r}")

    def score(self, sample, text_segments, n_search):
        """sample: {"question", "golds", "task"}; text_segments: [(text, from_model)]."""
        fmt_ok, fmt_reason = protocol.check_format(text_segments)
        model_text = "".join(t for t, m in text_segments if m)
        pred = protocol.extract_answer(model_text)
        task = sample.get("task", "qa")
        r_answer = self.answer_judge(sample["question"], pred, sample.get("golds", []), task)
        # The paper grants R_format when <think>, <search> and <answer> are all used correctly;
        # gated mode does not require a search for the format bonus.
        r_format = self.format_bonus if fmt_ok and (self.mode == "gated" or n_search > 0) else 0.0
        r_search = min(self.search_per_call * n_search, self.search_cap)
        reasoning = "".join(t for t, _ in text_segments).split(protocol.THINK_CLOSE)[0]
        needs_reason = self.mode == "paper" or r_answer > 0
        r_reason = self.reason_judge(sample["question"], reasoning) if needs_reason else 0.0

        if self.mode == "paper":
            total = r_search + r_format + r_reason + r_answer
        else:
            total = r_format + self.w_answer * r_answer + r_answer * (r_search + r_reason)
        return {"reward": total, "r_answer": r_answer, "r_format": r_format, "r_search": r_search, "r_reason": r_reason,
                "format": fmt_reason, "pred": pred, "n_search": n_search}

    def score_trajectory(self, traj):
        return self.score(traj.sample, traj.text_segments, traj.n_search)

    def score_text(self, sample, response_text):
        """For callers that only have decoded text (the OpenRLHF reward server)."""
        segments = protocol.segments_from_text(response_text)
        n_search = sum(1 for _, from_model in segments if not from_model)
        return self.score(sample, segments, n_search)
