"""The planner against a scripted model: no network, but the same request
and response shapes as the Anthropic SDK."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

import syrup
from syrup.planner import plan


class ScriptedClaude:
    def __init__(self, answer, stop_reason="end_turn"):
        self.answer, self.stop_reason, self.requests = answer, stop_reason, []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        text = SimpleNamespace(type="text", text=json.dumps(self.answer))
        return SimpleNamespace(stop_reason=self.stop_reason, content=[text])


def spec(**fields):
    keys = ["find", "color", "region", "order", "limit", "min_area_pct", "max_area_pct", "measure"]
    return {"spec": {key: fields.get(key) for key in keys}, "refusal": None}


def test_a_planned_name_is_checked_and_compiled_like_any_other():
    claude = ScriptedClaude(spec(find="bar", color="red", region=[0, 0.8, 1, 0.2], order="size", limit=1))
    op = plan("find_status_bar", "the big red bar along the bottom fifth", client=claude)
    assert op.intent == "find red bar in (0, 4/5, 1, 1/5), by area, largest first, at most 1"

    request = claude.requests[0]
    assert request["model"] == "claude-opus-5-5"
    schema = request["output_config"]["format"]["schema"]
    assert "bar" in schema["properties"]["spec"]["anyOf"][0]["properties"]["find"]["enum"]
    assert "status bar" not in request["system"] and "find_status_bar" in request["messages"][0]["content"]

    page = np.full((100, 200, 3), 30, np.uint8)
    page[85:95, 10:150] = (210, 40, 40)
    (bar,) = op(page)
    assert (bar.box.x, bar.box.y) == (10, 85)


def test_names_the_grammar_knows_never_reach_the_model():
    claude = ScriptedClaude(None)
    assert plan("find_largest_face", client=claude).name == "find_largest_face"
    with pytest.raises(syrup.IntentError) as e:
        plan("find_best_face", client=claude)
    assert e.value.kind == "ambiguous"
    assert claude.requests == []

    # A description settles what the grammar found ambiguous.
    claude = ScriptedClaude(spec(find="face", order="size", limit=2))
    assert plan("find_big_faces", "the two largest faces", client=claude).intent.endswith("at most 2")


def test_refusals_and_bad_proposals_are_intent_errors():
    claude = ScriptedClaude({"spec": None, "refusal": {"kind": "unsupported", "reason": "Syrup cannot tell who someone is"}})
    with pytest.raises(syrup.IntentError) as e:
        plan("find_my_boss", client=claude)
    assert (e.value.kind, e.value.reason) == ("unsupported", "Syrup cannot tell who someone is")

    # A colour on faces is refused by the same check a hand-written define gets.
    with pytest.raises(syrup.IntentError):
        plan("find_tanned_faces", client=ScriptedClaude(spec(find="face", color="orange")))

    with pytest.raises(syrup.IntentError):
        plan("find_anything", client=ScriptedClaude(None, stop_reason="refusal"))
