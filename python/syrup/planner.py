"""Names outside the grammar, planned by a language model.

    from syrup.planner import plan

    find_header_faces = plan("find_header_faces", "faces in the top fifth, biggest first")

The model never writes code and never picks a function. It may only answer
with the arguments of `syrup.define` (a target, colour, region, order,
limit, area bounds or measurement from Syrup's own vocabulary) or refuse
because the request is ambiguous or unsupported. Its answer then goes
through the same checks as any other definition: the intent and plan are
type-checked, the module is generated, compiled and validated against the
interpreter. A name the grammar already understands never reaches the model.

Needs the `anthropic` package (`pip install syrup-cv[planner]`) and an API
key or `ant auth login` profile.
"""

import json

from . import _call, _native, define, resolve
from .errors import IntentError

MODEL = "claude-opus-5-5"

SYSTEM = """You translate requests for computer-vision operations into a fixed
specification for the Syrup library. You cannot add capabilities: use only
the values the schema allows. If the request leaves a choice open (which
order, how many, what counts as large), or needs something no allowed value
expresses (identities, emotions, objects Syrup has no target for), refuse
with kind "ambiguous" or "unsupported" and a one-sentence reason, rather
than guessing. Regions are fractions (x, y, w, h) of the image, or a named
region. Area bounds are whole percentages of the image area. For reference,
this is the grammar of names Syrup understands without you:

"""


def _nullable(schema):
    return {"anyOf": [schema, {"type": "null"}]}


def _schema(vocabulary):
    fields = {
        "find": {"type": "string", "enum": vocabulary["targets"]},
        "color": _nullable({"type": "string", "enum": vocabulary["colors"]}),
        "region": _nullable(
            {
                "anyOf": [
                    {"type": "string", "enum": vocabulary["regions"]},
                    {"type": "array", "items": {"type": "number"}},
                ]
            }
        ),
        "order": _nullable({"type": "string", "enum": vocabulary["orders"]}),
        "limit": _nullable({"type": "integer"}),
        "min_area_pct": _nullable({"type": "integer"}),
        "max_area_pct": _nullable({"type": "integer"}),
        "measure": _nullable({"type": "string", "enum": vocabulary["quantities"]}),
    }
    refusal = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["ambiguous", "unsupported"]},
            "reason": {"type": "string"},
        },
        "required": ["kind", "reason"],
        "additionalProperties": False,
    }
    spec = {
        "type": "object",
        "properties": fields,
        "required": list(fields),
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"spec": _nullable(spec), "refusal": _nullable(refusal)},
        "required": ["spec", "refusal"],
        "additionalProperties": False,
    }


def _ask(client, model, name, description, vocabulary):
    request = f"Operation name: {name}"
    if description:
        request += f"\nWhat it should do: {description}"
    response = client.beta.messages.create(
        model=model,
        max_tokens=4000,
        # A short structured answer; low effort is plenty.
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": _schema(vocabulary)}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM + vocabulary["grammar"],
        messages=[{"role": "user", "content": request}],
    )
    if response.stop_reason == "refusal":
        raise IntentError("resolve", "unsupported", "the planner declined this request", name)
    if response.stop_reason == "max_tokens":
        raise IntentError("resolve", "malformed", "the planner's answer was cut off", name)
    text = next(block.text for block in response.content if block.type == "text")
    return json.loads(text)


def plan(name, description=None, *, client=None, model=MODEL):
    """The operation `name` means, planned by a model when the grammar does
    not cover it. With a `description`, that is what gets planned, and the
    name is only its label. Raises IntentError when the model finds the
    request ambiguous or unsupported, or proposes something Syrup rejects."""
    try:
        return resolve(name)
    except IntentError as unresolved:
        if description is None and unresolved.kind in ("ambiguous", "conflicting"):
            # The grammar understood the words and found them wanting; with
            # nothing more to go on, a model would only be guessing.
            raise
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    vocabulary = json.loads(_call(_native.vocabulary))
    answer = _ask(client, model, name, description, vocabulary)
    if answer.get("refusal"):
        refusal = answer["refusal"]
        raise IntentError("resolve", refusal["kind"], refusal["reason"], name, hint="planned by " + model)
    spec = {key: value for key, value in (answer.get("spec") or {}).items() if value is not None}
    if "find" not in spec:
        raise IntentError("resolve", "malformed", "the planner returned neither a plan nor a refusal", name)
    return define(name, **spec)
