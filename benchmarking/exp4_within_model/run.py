"""Experiment 4 — within-model: is a model's attention a better citation source
than the model's own words?

The question (issue #196) is deliberately narrow. Exp 1 compared TokenPath's
attention against a *different, larger* model's prompted citations, which changes
two variables at once — method and model — so it cannot answer "is attention
better than prompting". Here ONE model does everything: it writes the answer, it
cites its own answer when asked, and its attention is read for citations. The only
variable left is where the citation comes from.

No frontier rows, no cross-model rows. The claim this experiment can support is
exactly "for this model, attention cites better (or worse) than its own words",
and the table contains nothing that would invite a broader reading.

Two arms, both scored by the same judge on the same statements:

  Arm 1 (frozen)  The model answers WITHOUT citations; that answer is frozen. Then
                  (A) the model is asked to cite its own frozen answer, and
                  (B) its attention is read over the same frozen answer.
                  Cleanest isolation: identical statements, identical text.

  Arm 2 (inline)  The model answers AND cites in one pass, LongCite-style. Then
                  its attention is read over that same answer text, using the
                  statement boundaries the model itself chose. This removes the
                  objection that Arm 1's "add citations" call is artificial.

Both conditions in both arms run on the same weights on the same GPU, so the
latency and $/query columns are a straight work comparison: generating citation
text vs one prefill and an attention read.
"""

from __future__ import annotations

import argparse
import os

from .. import config
from ..common import env
from ..common.io_utils import read_json, read_jsonl_latest, write_json
from ..exp1_longbench_cite import freeze_answers, load_data, run as exp1
from ..exp1_longbench_cite.methods.prompted_method import PromptedMethod
from ..exp1_longbench_cite.methods.tokenpath_method import TokenPathMethod
from .methods.inline_method import InlineCiteMethod

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PREFIX = "exp4"
INLINE_PREFIX = "exp4inline"
FROZEN_PATH = os.path.join(RESULTS_DIR, "exp4_frozen_answers.jsonl")


def agg_cfg(threshold_override: float | None = None) -> dict:
    """Aggregation config for the attention condition.

    The threshold comes from Exp 4's own coarse val sweep — Exp 1's tuned value was
    fitted to a different model's answers, and reusing it would run the attention
    condition at an operating point chosen for another answer distribution. Kept
    deliberately coarse (see tune_threshold_coarse.py): a result that needs a
    finely-tuned knob is a weaker result than one that does not.
    """
    tuned = read_json(os.path.join(RESULTS_DIR, "exp4_threshold.json"), {})
    threshold = (
        threshold_override
        if threshold_override is not None
        else tuned.get("best_threshold", config.TOKENPATH_AGG["threshold"])
    )
    return {**config.TOKENPATH_AGG, "threshold": float(threshold)}


def _cfg(args) -> config.RunConfig:
    """Provenance for this run: one model fills every LLM role.

    tokenpath_mass_threshold must be the RESOLVED threshold, not RunConfig's
    default. Left unset it recorded config.TOKENPATH_MASS_THRESHOLD (0.30) into
    every scores file while the run actually used the val-tuned 0.10 — a
    provenance block that contradicts the run it describes is worse than none.
    """
    return config.RunConfig(
        run_date=args.run_date,
        generator_model=args.model,
        prompted_model=args.model,
        judge_model=args.judge_model,
        tokenpath_mass_threshold=agg_cfg(args.threshold)["threshold"],
        seed=args.seed,
        extra={"experiment": "exp4_within_model", "model_under_test": args.model},
    )


# --------------------------------------------------------------------------- #
# Arm 1 — frozen answer, cited two ways by the same model                     #
# --------------------------------------------------------------------------- #
def run_arm1(examples: list[dict], args, cfg: config.RunConfig) -> dict:
    only = {e["idx"] for e in examples}
    freeze_answers.freeze(examples, args.model, out_path=FROZEN_PATH)
    frozen = {
        r["idx"]: r["prediction"]
        for r in read_jsonl_latest(FROZEN_PATH)
        if r.get("prediction")
    }

    client = env.openrouter_client()
    methods = {
        # Condition A: the model cites its own answer. Run twice — plain, and with
        # the thought channel on — so the prompted side is reported at its best
        # rather than at whichever setting we happened to pick. Thinking needs a
        # bigger budget because the reasoning shares the completion window.
        "prompted": PromptedMethod(client, model=args.model, enable_thinking=False,
                                  max_tokens=args.prompted_max_tokens),
        "prompted_thinking": PromptedMethod(
            client, model=args.model, enable_thinking=True,
            max_tokens=args.thinking_max_tokens, name="prompted_thinking",
        ),
        # Condition B: the model's attention over that same answer.
        "tokenpath": TokenPathMethod(env.tokenpath_client(), agg_cfg=agg_cfg(args.threshold)),
    }

    summaries = {}
    for name, method in methods.items():
        cited = exp1.cite_stage(name, method, examples, frozen, prefix=PREFIX)
        judged = exp1.judge_stage(name, cited, cfg, prefix=PREFIX)
        summaries[name] = exp1.aggregate(name, judged, cited, cfg, prefix=PREFIX,
                                         only_idx=only)
        _attach_format_adherence(summaries[name], cited, name, PREFIX, only_idx=only)
        _sanity_check(name, summaries[name], cited, judged, only_idx=only)

    # Name the steelman explicitly so the writeup quotes the stronger prompted row.
    prompted_rows = {k: v for k, v in summaries.items() if k.startswith("prompted")}
    best = max(prompted_rows, key=lambda k: prompted_rows[k]["avg_reported"]["citation_f1"])
    summaries["_condition_a_best"] = {"row": best,
                                      "citation_f1": prompted_rows[best]["avg_reported"]["citation_f1"]}
    return summaries


def _sanity_check(name: str, summary: dict, cited_path: str, judged_path: str,
                  only_idx: set | None = None) -> None:
    """Fail loudly on the failure shapes that already fooled this harness once.

    Three separate bugs (an unparsed thought channel, an over-strict statement
    regex, and stale judgments reused because a record carried no cache signature)
    each produced a confident 0.000 for a generated-text condition while leaving
    the attention condition untouched — i.e. each one silently favoured the result
    we wanted. These invariants make that shape crash instead of publish.
    """
    problems = []

    # 1. Judging must never change the citations it was handed. If counts differ,
    #    the judge scored different records than the cite stage produced (stale
    #    cache, mismatched idx, truncated statement list).
    cited = {r["idx"]: r for r in read_jsonl_latest(cited_path)}
    judged = {r["idx"]: r for r in read_jsonl_latest(judged_path)}
    if only_idx is not None:
        cited = {i: r for i, r in cited.items() if i in only_idx}
        judged = {i: r for i, r in judged.items() if i in only_idx}
    for idx, cr in cited.items():
        jr = judged.get(idx)
        if jr is None:
            problems.append(f"idx {idx} cited but never judged")
            continue
        n_c = sum(len(s.get("citation", [])) for s in cr.get("statements", []))
        n_j = sum(len(s.get("citation", [])) for s in jr.get("statements", []))
        if n_c != n_j:
            problems.append(
                f"idx {idx}: {n_c} citations produced but {n_j} judged — the judge "
                "did not score the records the cite stage wrote"
            )

    # 2. Truncation is a budget we chose, not a property of the model. When a
    #    condition-A row runs out of completion window it emits no citations and
    #    scores as if it cited badly — which reads as "prompting is worse" when it
    #    actually means "we starved it". The guard above missed exactly this,
    #    because a partly-truncated response still parses sometimes.
    #    Some truncation is irreducible (a model that rambles past any budget), so
    #    only a rate high enough to distort the score is fatal. Between the two
    #    bounds it warns: visible, recorded in format_adherence.truncated_rate, and
    #    reportable as a caveat rather than silently blocking a usable run.
    trunc = [r for r in cited.values() if (r.get("extra") or {}).get("truncated")]
    rate = len(trunc) / len(cited) if cited else 0.0
    if rate > 0.02:
        by_ds: dict = {}
        for r in trunc:
            by_ds[r.get("dataset")] = by_ds.get(r.get("dataset"), 0) + 1
        msg = (f"{len(trunc)}/{len(cited)} responses ({rate:.1%}) hit the token cap "
               f"({by_ds}) — a truncated response scores as a citation failure when "
               "it is really our budget")
        if rate > 0.10:
            problems.append(msg + " — too high to report; raise the budget")
        else:
            print(f"  WARNING [{name}] {msg}")

    # 3. A dead-zero score alongside well-formed output is a scoring-path bug, not
    #    a model result. A model that emits parseable citations scores > 0 somewhere.
    f1 = summary.get("avg_reported", {}).get("citation_f1", 0.0)
    adherence = summary.get("format_adherence") or {}
    if f1 == 0.0 and adherence.get("format_ok_rate", 0.0) > 0.5:
        problems.append(
            f"F1 is exactly 0.000 while {adherence['format_ok_rate']:.0%} of "
            "responses were well-formed — that shape has been a bug every time"
        )

    if problems:
        raise RuntimeError(
            f"[{name}] sanity check failed:\n  - " + "\n  - ".join(problems[:8])
        )


def _attach_format_adherence(summary: dict, cited_path: str, name: str,
                             prefix: str, only_idx: set | None = None) -> None:
    """Fold per-record format diagnostics into the row's summary, on disk.

    Reported as a column, never silently absorbed into the score: a model that
    cannot hold the citation format is failing differently from one that cites the
    wrong sentence, and the distinction is what keeps this comparison honest.

    aggregate() has already written the scores file by the time we get here, so
    this re-writes it — mutating the returned dict alone would leave the published
    artifact without the diagnostics.
    """
    recs = read_jsonl_latest(cited_path)
    if only_idx is not None:
        recs = [r for r in recs if r["idx"] in only_idx]
    diags = [r.get("extra", {}) for r in recs]
    if not diags:
        return
    # Only generated-text conditions have a format to hold. The attention condition
    # emits no text, so it has no format_ok field — reporting 0/N for it would read
    # as "failed to format 100% of the time", which is meaningless, not a result.
    if not any("format_ok" in d for d in diags):
        return
    n = len(diags)
    adherence = {"n": n,
                 "format_ok_rate": round(sum(1 for d in diags if d.get("format_ok")) / n, 4),
                 "truncated_rate": round(sum(1 for d in diags if d.get("truncated")) / n, 4)}
    nonverbatim = sum(int(d.get("n_quotes_nonverbatim", 0)) for d in diags)
    quotes = sum(int(d.get("n_quotes", 0)) for d in diags)
    if quotes:
        adherence["nonverbatim_quote_rate"] = round(nonverbatim / quotes, 4)
    hallucinated = sum(int(d.get("dropped_indices", 0)) for d in diags)
    if hallucinated:
        adherence["hallucinated_index_total"] = hallucinated
    summary["format_adherence"] = adherence
    write_json(os.path.join(RESULTS_DIR, f"{prefix}_scores_{name}.json"), summary)


# --------------------------------------------------------------------------- #
# Arm 2 — inline citations in one pass, vs attention over that same answer     #
# --------------------------------------------------------------------------- #
def run_arm2(examples: list[dict], args, cfg: config.RunConfig) -> dict:
    only = {e["idx"] for e in examples}
    client = env.openrouter_client()
    # Thinking rescued the prompted condition in Arm 1 (0.00 -> 0.80 once the
    # thought channel was parsed correctly), so the inline condition gets the same
    # chance rather than being reported at whichever setting we happened to try.
    inline = (
        InlineCiteMethod(client, model=args.model, max_tokens=args.thinking_max_tokens,
                         enable_thinking=True)
        if args.inline_thinking
        else InlineCiteMethod(client, model=args.model, max_tokens=args.max_tokens)
    )

    # Stage 1: the model answers and cites in one call. Its own answer text (and
    # its own statement boundaries) become the shared ground for both conditions.
    # `frozen` is empty here because this arm generates its answer inside the
    # method; cite_stage still handles caching/retries/failure accounting for us.
    empty: dict[str, str] = {e["idx"]: "" for e in examples}
    inline_cited = exp1.cite_stage("inline", inline, examples, empty, prefix=INLINE_PREFIX)

    inline_records = {r["idx"]: r for r in read_jsonl_latest(inline_cited)}

    # Stage 2: attention over the SAME answer text, on the SAME statement spans.
    class TokenPathOnInline(TokenPathMethod):
        """Attention condition pinned to the inline answer's own statements."""

        name = "tokenpath"

        def cite(self, example, answer, statement_spans_override=None):
            rec = inline_records[example["idx"]]
            spans = [s["span"] for s in rec["statements"]]
            return super().cite(example, rec["prediction"], statement_spans_override=spans)

        def cache_signature_for(self, example, answer):
            rec = inline_records.get(example["idx"], {})
            return {
                **super().cache_signature_for(example, rec.get("prediction", "")),
                "paired_with": "inline",
                "n_statements": len(rec.get("statements", [])),
            }

    tp_on_inline = TokenPathOnInline(
        env.tokenpath_client(), agg_cfg=agg_cfg(args.threshold)
    )

    answers = {idx: r["prediction"] for idx, r in inline_records.items()}
    usable = [e for e in examples if answers.get(e["idx"])]
    tp_cited = exp1.cite_stage(
        "tokenpath", tp_on_inline, usable, answers, prefix=INLINE_PREFIX
    )

    # Stage 3: a SECOND-PASS citer over those same answers. Without this, a weak
    # inline score cannot be interpreted: emitting <statement>/<cite> tags mid
    # generation and identifying the right source sentence are different skills,
    # and inline failure could be entirely the former. The second pass asks only
    # for the latter — same model, same answer, same statements, citations in a
    # separate call — so inline vs 2-pass isolates format-following, and 2-pass vs
    # attention isolates the citation source.
    class PromptedOnInline(PromptedMethod):
        """Second-pass citer pinned to the inline answer's own statements."""

        name = "prompted_2pass"

        def cite(self, example, answer, statement_spans_override=None):
            rec = inline_records[example["idx"]]
            spans = [s["span"] for s in rec["statements"]]
            return super().cite(example, rec["prediction"],
                                statement_spans_override=spans)

        @property
        def cache_signature(self) -> dict:
            return {**super().cache_signature, "paired_with": "inline"}

    two_pass = PromptedOnInline(client, model=args.model, enable_thinking=False,
                                max_tokens=args.prompted_max_tokens,
                                name="prompted_2pass")
    two_pass_cited = exp1.cite_stage(
        "prompted_2pass", two_pass, usable, answers, prefix=INLINE_PREFIX
    )

    summaries = {}
    for name, cited in (("inline", inline_cited), ("prompted_2pass", two_pass_cited),
                        ("tokenpath", tp_cited)):
        judged = exp1.judge_stage(name, cited, cfg, prefix=INLINE_PREFIX)
        summaries[name] = exp1.aggregate(name, judged, cited, cfg, prefix=INLINE_PREFIX,
                                         only_idx=only)
        _attach_format_adherence(summaries[name], cited, name, INLINE_PREFIX, only_idx=only)
        _sanity_check(name, summaries[name], cited, judged, only_idx=only)

    return summaries


def main():
    ap = argparse.ArgumentParser(description="Exp 4 — within-model citation comparison")
    ap.add_argument("--arm", choices=["1", "2", "both"], default="both")
    ap.add_argument("--model", default=config.EXP4_MODEL,
                    help="the one model under test, filling every LLM role")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--judge-model", default=config.JUDGE_MODEL)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the attention mass threshold (else exp4_threshold.json)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--prompted-max-tokens", type=int, default=16384,
                    help="completion budget for the prompted row. JSON carrying a "
                         "verbatim quote per answer sentence is token-hungry: at 2048 "
                         "85%% of failures were truncation on gov_report, not the model "
                         "failing to format. Starving this row silently favours attention.")
    ap.add_argument("--inline-thinking", action="store_true",
                    help="Arm 2 steelman: let the inline condition use the thought channel")
    ap.add_argument("--thinking-max-tokens", type=int, default=40960,
                    help="completion budget for the thinking-enabled prompted row "
                         "(reasoning shares the window with the citation JSON)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-languages", action="store_true")
    ap.add_argument("--limit-per-dataset", type=int, default=None)
    ap.add_argument("--run-date", default=config.DEFAULT_RUN_DATE)
    args = ap.parse_args()

    examples = load_data.load_split(
        args.split,
        seed=args.seed,
        english_only=not args.all_languages,
        limit_per_dataset=args.limit_per_dataset,
    )
    cfg = _cfg(args)
    print(f"exp4: model={args.model} split={args.split} n={len(examples)} arm={args.arm}")

    out: dict[str, dict] = {}
    if args.arm in ("1", "both"):
        out["arm1_frozen"] = run_arm1(examples, args, cfg)
    if args.arm in ("2", "both"):
        out["arm2_inline"] = run_arm2(examples, args, cfg)

    for arm, summaries in out.items():
        print(f"\n=== {arm} ===")
        for name, s in summaries.items():
            if "avg_reported" not in s:  # bookkeeping entries (e.g. _condition_a_best)
                print(f"  {name}: {s}")
                continue
            a = s["avg_reported"]
            print(f"  {name:10s} F1={a['citation_f1']:.3f} "
                  f"R={a['citation_recall']:.3f} P={a['citation_precision']:.3f} "
                  f"p50={s['cost_latency']['latency_p50_s']:.2f}s "
                  f"${s['cost_latency']['usd_per_query_mean']:.5f}/q")


if __name__ == "__main__":
    main()
