
import pandas as pd
import pytest

from ep_voting.export import COLUMNS, build_table, write_excel
from ep_voting.llm import PREFILL, build_messages, evaluate_amendments, parse_model_output
from ep_voting.pipeline import run_pipeline
from ep_voting.schemas import Amendment
from ep_voting.screening import score_amendments, select_relevant, split_priorities
from ep_voting.stubs import StubEmbedder, stub_generate


# ----- screening -------------------------------------------------------------------

def test_split_priorities_strips_numbering_and_bullets():
    text = "1. Protect youth jobs\n- Keep NGO access\n\n  • No defence transfers"
    assert split_priorities(text) == ["Protect youth jobs", "Keep NGO access", "No defence transfers"]


def make(n, text, location="Paragraph 1"):
    return Amendment(number=n, authors="A", doc_type="Motion", location=location, amendment_text=text)


def test_scoring_uses_best_priority_and_ranks():
    amendments = [
        make(1, "apprenticeships for young people"),
        make(2, "weather forecasts for sailors"),
        make(3, "grants for youth NGOs"),
    ]
    prio = "Quality apprenticeships for young people\nGrants for youth NGOs"
    score_amendments(amendments, prio, StubEmbedder())
    assert amendments[0].matched_priority == 1
    assert amendments[2].matched_priority == 2
    assert amendments[1].relevance_score < amendments[0].relevance_score
    kept = select_relevant(amendments, threshold=0.3, max_items=10)
    assert [a.number for a in kept][-1:] != [2]
    assert 2 not in [a.number for a in kept]
    assert len(select_relevant(amendments, 0.0, 2)) == 2  # cap applies


# ----- model output parsing ---------------------------------------------------------

@pytest.mark.parametrize(
    "completion",
    [
        'Supports youth jobs.", "recommended_vote": "+"}',  # continuation of the prefill
        '{"policy_rationale": "Supports youth jobs.", "recommended_vote": "+"}',
        '```json\n{"policy_rationale": "Supports youth jobs.", "recommended_vote": "+"}\n```',
        'Supports youth jobs.", "recommended_vote": "+"}\nExtra chatter after the JSON.',
        'Supports youth jobs.", "recommended_vote": "for"}',  # alias
    ],
)
def test_parse_model_output_accepts_valid_variants(completion):
    result = parse_model_output(completion)
    assert result is not None and result.recommended_vote == "+"
    assert result.policy_rationale == "Supports youth jobs."


def test_parse_model_output_rejects_garbage_and_bad_votes():
    assert parse_model_output("I think you should vote yes.") is None
    assert parse_model_output('x", "recommended_vote": "maybe"}') is None
    assert parse_model_output("") is None


def test_malformed_json_with_readable_vote_is_flagged():
    result = parse_model_output('Weakens the budget line", "recommended_vote": "-"')
    assert result.recommended_vote == "-"
    assert result.policy_rationale.startswith("[Check manually")


def test_answer_cut_off_before_the_vote_is_unusable():
    assert parse_model_output('Weakens the budget line for yo') is None


def test_rationale_is_capped():
    long = " ".join(["word"] * 100)
    result = parse_model_output(f'{long}", "recommended_vote": "+"}}')
    assert len(result.policy_rationale.split()) <= 46


def test_prompt_tells_the_model_what_was_removed_and_added():
    a = make(7, "3. Calls for **new youth** funds;")
    a.original_text = "3. Calls for **existing** funds;"
    a.doc_type, a.location = "Motion", "Paragraph 3"
    user = build_messages(a, "Protect youth jobs")[1]["content"]
    assert "Protect youth jobs" in user
    assert "REMOVED by the amendment: existing" in user
    assert "ADDED by the amendment: new youth" in user
    assert "**" not in user
    assert PREFILL.startswith('{"policy_rationale"')


def test_prompt_for_new_insertion():
    a = make(8, "Ba. whereas something new;")
    a.is_new = True
    user = build_messages(a, "p")[1]["content"]
    assert "NEW INSERTION" in user and "whereas something new" in user and "REMOVED" not in user


def test_failed_answers_become_review_never_a_made_up_vote():
    amendments = [make(i, "text") for i in range(1, 4)]

    def generate(prompts):
        return ['ok", "recommended_vote": "+"}', "nonsense", 'no", "recommended_vote": "-"}']

    decisions, errors = evaluate_amendments(amendments, "p", generate)
    assert [d.recommended_vote for d in decisions] == ["+", "REVIEW", "-"]
    assert errors == []


def test_gpu_failure_keeps_finished_chunks_and_marks_the_rest():
    amendments = [make(i, "text") for i in range(1, 6)]
    calls = {"n": 0}

    def generate(prompts):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("GPU quota exceeded")
        return ['ok", "recommended_vote": "+"}'] * len(prompts)

    decisions, errors = evaluate_amendments(amendments, "p", generate, chunk_size=2)
    assert [d.recommended_vote for d in decisions] == ["+", "+", "REVIEW", "REVIEW", "REVIEW"]
    assert "GPU quota exceeded" in errors[0] and "amendment 3" in errors[0]
    assert "GPU quota exceeded" in decisions[2].policy_rationale


# ----- table, Excel, full pipeline --------------------------------------------------

def test_table_is_in_voting_order_without_markers():
    a1, a2 = make(10, "x **bold** y"), make(2, "z")
    decisions = evaluate_amendments([a1, a2], "p", stub_generate)[0]
    df = build_table([a1, a2], decisions)
    assert list(df.columns) == COLUMNS
    assert list(df["Amendment"]) == ["AM 2", "AM 10"]
    assert df.loc[1, "Amendment text"] == "x bold y"


def test_excel_round_trip(tmp_path):
    df = pd.DataFrame([{c: "v" for c in COLUMNS}])
    path = write_excel(df, str(tmp_path))
    back = pd.read_excel(path)
    assert list(back.columns) == COLUMNS and len(back) == 1


def test_full_pipeline_on_fixture(fixture_pdf):
    out = run_pipeline(fixture_pdf, "Youth Guarantee and youth instruments", 0.0, 10,
                       StubEmbedder(), stub_generate)
    assert len(out.table) == 6
    assert list(out.table["Amendment"]) == [f"AM {i}" for i in range(1, 7)]
    assert "6" in out.status and "Check every recommendation" in out.status


def test_pipeline_messages_for_bad_input(fixture_pdf):
    assert "upload" in run_pipeline(None, "p", 0.2, 5, StubEmbedder(), stub_generate).status.lower()
    assert "priorities" in run_pipeline(fixture_pdf, "  ", 0.2, 5, StubEmbedder(), stub_generate).status
    high = run_pipeline(fixture_pdf, "zzz qqq", 0.99, 5, StubEmbedder(), stub_generate)
    assert len(high.table) == 0 and "threshold" in high.status.lower()


def test_pipeline_cap_message(fixture_pdf):
    out = run_pipeline(fixture_pdf, "youth", 0.0, 2, StubEmbedder(), stub_generate)
    assert len(out.table) == 2 and "left out by the limit" in out.status


def test_real_document_pipeline(real_pdf):
    out = run_pipeline(real_pdf, "youth employment apprenticeships\ngrants for NGOs", 0.2, 60,
                       StubEmbedder(), stub_generate, chunk_size=24)
    assert 0 < len(out.table) <= 60
    assert not (out.table["Vote"] == "REVIEW").any()


def test_transient_gpu_failure_is_retried():
    amendments = [make(i, "text") for i in range(1, 3)]
    calls = {"n": 0}

    def generate(prompts):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("No CUDA GPUs are available")
        return ['ok", "recommended_vote": "+"}'] * len(prompts)

    decisions, errors = evaluate_amendments(amendments, "p", generate, retry_wait=0)
    assert [d.recommended_vote for d in decisions] == ["+", "+"]
    assert errors == [] and calls["n"] == 3


def test_persistent_gpu_failure_names_the_error_type():
    amendments = [make(1, "text")]

    def generate(prompts):
        raise RuntimeError("No CUDA GPUs are available")

    decisions, errors = evaluate_amendments(amendments, "p", generate, retries=1, retry_wait=0)
    assert decisions[0].recommended_vote == "REVIEW"
    assert "RuntimeError: No CUDA GPUs are available" in errors[0]


def test_review_row_shows_start_of_raw_answer():
    decisions, _ = evaluate_amendments([make(1, "t")], "p", lambda prompts: ["I would vote yes because it is good"])
    assert decisions[0].recommended_vote == "REVIEW"
    assert "I would vote yes" in decisions[0].policy_rationale


@pytest.mark.parametrize(
    ("word", "symbol"),
    [("SUPPORT", "+"), ("OPPOSE", "-"), ("oppose", "-"), ("ABSTAIN", "ABSTAIN"), ("SPLIT", "SPLIT")],
)
def test_vote_words_are_mapped_to_table_symbols(word, symbol):
    result = parse_model_output(f'Some reason.", "recommended_vote": "{word}"}}')
    assert result.recommended_vote == symbol


def test_prompt_asks_for_words_not_symbols():
    from ep_voting.llm import SYSTEM_PROMPT

    assert "SUPPORT, OPPOSE, ABSTAIN, SPLIT" in SYSTEM_PROMPT
    assert "speculative" in SYSTEM_PROMPT
