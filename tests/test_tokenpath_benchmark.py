from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from benchmarking import config
from benchmarking.common import aggregate as agg
from benchmarking.common.io_utils import (append_jsonl, load_done_ids,
                                           read_jsonl, read_jsonl_latest)
from benchmarking.common.tokenpath import Timed
from benchmarking.exp1_longbench_cite import native_run, run, tune_threshold
from benchmarking.exp1_longbench_cite.methods.base import CitedAnswer
from benchmarking.exp1_longbench_cite.methods.tokenpath_method import TokenPathMethod


EXAMPLE = {
    "idx": "demo",
    "dataset": "hotpotqa",
    "context": "Alpha. Beta.",
    "query": "Which one?",
}
ANSWER = "Alpha."


class FakeTokenPathClient:
    def __init__(self):
        self.calls = 0

    def heatmap(self, document, query, answer):
        self.calls += 1
        return Timed(
            {
                "shape": [1, 2],
                "row": [0, 0],
                "col": [0, 1],
                "data": [0.9, 0.1],
                "answer_offsets": [[0, 6]],
                "document_offsets": [[0, 6], [7, 12]],
            },
            0.25,
        )


class TokenPathMethodTests(unittest.TestCase):
    def test_cite_records_effective_aggregation_config(self):
        method = TokenPathMethod(
            FakeTokenPathClient(),
            agg_cfg={"row_norm": True},
        )

        record = method.cite(EXAMPLE, ANSWER).to_record()

        self.assertEqual(record["extra"]["mass_threshold"], agg.BASELINE["threshold"])
        self.assertEqual(record["extra"]["agg_cfg"], method.agg_cfg)
        self.assertEqual(
            record["extra"]["cache_signature"],
            method.cache_signature_for(EXAMPLE, ANSWER),
        )
        self.assertNotEqual(
            method.cache_signature_for(EXAMPLE, ANSWER),
            method.cache_signature_for(EXAMPLE, "Different answer."),
        )
        self.assertEqual(record["statements"][0]["citation"][0]["cite"], "Alpha.")

    def test_constructor_accepts_legacy_threshold_and_positional_config(self):
        legacy = TokenPathMethod(FakeTokenPathClient(), 0.42)
        positional_cfg = TokenPathMethod(FakeTokenPathClient(), {"threshold": 0.27})

        self.assertEqual(legacy.mass_threshold, 0.42)
        self.assertTrue(legacy.agg_cfg["row_norm"])
        self.assertEqual(positional_cfg.mass_threshold, 0.27)
        self.assertFalse(positional_cfg.agg_cfg["row_norm"])

    def test_run_builder_applies_tuned_threshold_without_mutating_defaults(self):
        original = dict(config.TOKENPATH_AGG)
        cfg = config.RunConfig(tokenpath_mass_threshold=0.47)

        with mock.patch.object(run.env, "tokenpath_client", return_value=object()):
            method = run.build_method("tokenpath", cfg)

        self.assertEqual(method.mass_threshold, 0.47)
        self.assertTrue(method.agg_cfg["row_norm"])
        self.assertTrue(method.agg_cfg["merge_adjacent"])
        self.assertEqual(config.TOKENPATH_AGG, original)


class ThresholdFlowTests(unittest.TestCase):
    def test_native_threshold_precedence(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            native_run, "RESULTS_DIR", tmp
        ):
            self.assertEqual(
                native_run._tokenpath_agg_cfg()["threshold"],
                config.TOKENPATH_AGG["threshold"],
            )
            with open(os.path.join(tmp, "exp1_threshold.json"), "w", encoding="utf-8") as f:
                json.dump({"best_threshold": 0.41}, f)
            self.assertEqual(native_run._tokenpath_agg_cfg()["threshold"], 0.41)
            self.assertEqual(native_run._tokenpath_agg_cfg(0.0)["threshold"], 0.0)

    def test_threshold_tuner_uses_production_aggregator(self):
        calls = []

        def fake_aggregate(hm, start, end, doc_sents, cfg, answer_text=None):
            calls.append((hm, start, end, doc_sents, cfg, answer_text))
            return [(0, 6, 0.9)]

        class FakeJudge:
            def __init__(self, client, model):
                self.cost_usd = 0.0

            def get_citation_score(self, record, max_statement_num=40):
                self.assert_record = record
                return {**record, "citation_f1": 1.0}

        heatmap_without_legacy_helper = object()
        with mock.patch.object(tune_threshold.agg, "aggregate", side_effect=fake_aggregate), \
             mock.patch.object(tune_threshold, "CitationJudge", FakeJudge):
            score, cost = tune_threshold._score_threshold(
                0.46,
                {"demo": heatmap_without_legacy_helper},
                [EXAMPLE],
                {"demo": ANSWER},
                object(),
                "judge",
                {"demo": [[0, 6], [7, 12]]},
            )

        self.assertEqual((score, cost), (1.0, 0.0))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][4]["threshold"], 0.46)
        self.assertTrue(calls[0][4]["row_norm"])
        self.assertTrue(calls[0][4]["merge_adjacent"])
        self.assertEqual(calls[0][5], ANSWER)


class CacheRecoveryTests(unittest.TestCase):
    def test_error_attempt_is_retryable_and_latest_success_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cache.jsonl")
            append_jsonl(path, {"idx": "x", "extra": {"cache_signature": {"v": 1}}})
            append_jsonl(path, {"idx": "x", "extra": {"error": "boom"}})

            self.assertEqual(load_done_ids(path), set())
            self.assertEqual(read_jsonl_latest(path)[0]["extra"]["error"], "boom")

            append_jsonl(path, {"idx": "x", "extra": {"cache_signature": {"v": 2}}})
            self.assertEqual(load_done_ids(path), {"x"})
            self.assertEqual(
                load_done_ids(path, expected_signature={"v": 1}),
                set(),
            )
            self.assertEqual(
                load_done_ids(path, expected_signature={"v": 2}),
                {"x"},
            )

    def test_both_cite_stages_retry_errors_and_invalidate_changed_config(self):
        class Method:
            def __init__(self, signature, fail=False):
                self.cache_signature = signature
                self.fail = fail
                self.calls = 0

            def cite(self, example, answer):
                self.calls += 1
                if self.fail:
                    raise RuntimeError("broken")
                return CitedAnswer(
                    idx=example["idx"], dataset=example["dataset"],
                    query=example["query"], prediction=answer,
                    statements=[], method="tokenpath",
                    extra={"cache_signature": self.cache_signature},
                )

        for module in (run, native_run):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(module, "RESULTS_DIR", tmp):
                broken = Method({"threshold": 0.3}, fail=True)
                with self.assertRaisesRegex(RuntimeError, "failed for 1/1"):
                    module.cite_stage("tokenpath", broken, [EXAMPLE], {"demo": ANSWER})

                fixed = Method({"threshold": 0.3})
                module.cite_stage("tokenpath", fixed, [EXAMPLE], {"demo": ANSWER})
                self.assertEqual(fixed.calls, 1)

                changed = Method({"threshold": 0.4})
                module.cite_stage("tokenpath", changed, [EXAMPLE], {"demo": ANSWER})
                self.assertEqual(changed.calls, 1)

                path = (
                    os.path.join(tmp, "exp1_cited_tokenpath.jsonl")
                    if module is run
                    else os.path.join(tmp, "exp1nat_cited_tokenpath.jsonl")
                )
                self.assertEqual(len(read_jsonl(path)), 3)
                self.assertEqual(
                    read_jsonl_latest(path)[0]["extra"]["cache_signature"],
                    {"threshold": 0.4},
                )

    def test_both_cite_stages_reject_partial_failures(self):
        second = {**EXAMPLE, "idx": "bad"}

        class MixedMethod:
            cache_signature = {"threshold": 0.3}

            def cite(self, example, answer):
                if example["idx"] == "bad":
                    raise RuntimeError("broken")
                return CitedAnswer(
                    idx=example["idx"], dataset=example["dataset"],
                    query=example["query"], prediction=answer,
                    statements=[], method="tokenpath",
                    extra={"cache_signature": self.cache_signature},
                )

        for module in (run, native_run):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as tmp, \
                 mock.patch.object(module, "RESULTS_DIR", tmp):
                with self.assertRaisesRegex(RuntimeError, "failed for 1/2"):
                    module.cite_stage(
                        "tokenpath",
                        MixedMethod(),
                        [EXAMPLE, second],
                        {"demo": ANSWER, "bad": ANSWER},
                    )

    def test_both_judges_replace_cached_error_once(self):
        class FakeJudge:
            calls = 0

            def __init__(self, client, model):
                self.cost_usd = 0.01

            def get_citation_score(self, record, max_statement_num=40):
                type(self).calls += 1
                return {
                    **record,
                    "citation_recall": 1.0,
                    "citation_precision": 1.0,
                    "citation_f1": 1.0,
                }

        signature = {"agg_cfg": {"threshold": 0.3}}
        cited_error = {
            **EXAMPLE,
            "prediction": ANSWER,
            "statements": [],
            "method": "tokenpath",
            "latency_s": 0.0,
            "cost_usd": 0.0,
            "extra": {"error": "old failure", "cache_signature": signature},
        }
        cited_success = {
            **cited_error,
            "latency_s": 0.25,
            "cost_usd": 0.1,
            "extra": {"cache_signature": signature},
        }
        judged_error = {
            **cited_error,
            "citation_recall": 0.0,
            "citation_precision": 0.0,
            "citation_f1": 0.0,
            "judge_model": "judge",
        }

        for module in (run, native_run):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as tmp, \
                 mock.patch.object(module, "RESULTS_DIR", tmp), \
                 mock.patch.object(module, "CitationJudge", FakeJudge), \
                 mock.patch.object(module.env, "openrouter_client", return_value=object()):
                FakeJudge.calls = 0
                if module is run:
                    cited_path = os.path.join(tmp, "exp1_cited_tokenpath.jsonl")
                    judged_path = os.path.join(tmp, "exp1_judged_tokenpath.jsonl")
                else:
                    cited_path = os.path.join(tmp, "exp1nat_cited_tokenpath.jsonl")
                    judged_path = os.path.join(tmp, "exp1nat_judged_tokenpath.jsonl")

                append_jsonl(cited_path, cited_error)
                append_jsonl(cited_path, cited_success)
                append_jsonl(judged_path, judged_error)

                if module is run:
                    cfg = config.RunConfig(judge_model="judge")
                    module.judge_stage("tokenpath", cited_path, cfg)
                    module.judge_stage("tokenpath", cited_path, cfg)
                    with mock.patch.object(module, "mean_citation_len", return_value=(0.0, 0)), \
                         mock.patch.object(module, "cite_len_backend", return_value="words"):
                        summary = module.aggregate("tokenpath", judged_path, cited_path, cfg)
                    self.assertEqual(summary["n_examples"], 1)
                    self.assertEqual(summary["avg_reported"]["citation_f1"], 1.0)
                    self.assertEqual(summary["cost_latency"]["n"], 1)
                else:
                    with mock.patch.object(module.config, "JUDGE_MODEL", "judge"):
                        module.judge_stage("tokenpath")
                        module.judge_stage("tokenpath")

                self.assertEqual(FakeJudge.calls, 1)
                self.assertEqual(len(read_jsonl(judged_path)), 2)
                self.assertNotIn("error", read_jsonl_latest(judged_path)[0]["extra"])


if __name__ == "__main__":
    unittest.main()
