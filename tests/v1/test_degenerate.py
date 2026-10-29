"""The degenerate-turn rule (live stop + post-hoc scan) and the `repetition` finish reason."""

from types import SimpleNamespace

import httpx
import pytest

import verifiers.v1 as vf
from verifiers.v1 import graph
from verifiers.v1.clients import ModelContext
from verifiers.v1.clients.train import response_from_generate, serialize_completion
from verifiers.v1.configs.agent import DegenerateTurnLimit
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.degenerate import find_degenerate_turn, turn_mean_nll
from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.session import RolloutLimits, RolloutSession
from verifiers.v1.types import TurnTokens

RULE = {"min_tokens": 100, "max_mean_nll": 3.0}


def test_rule_thresholds_and_repetition_exemption():
    assert turn_mean_nll([-8.0] * 4000, "length", **RULE) == (4000, 8.0)
    assert turn_mean_nll([-8.0] * 4000, "stop", **RULE) is not None
    assert turn_mean_nll([-8.0] * 4000, None, **RULE) is not None
    # A repetition-stopped turn is never judged, however high its NLL.
    assert turn_mean_nll([-8.0] * 4000, "repetition", **RULE) is None
    assert turn_mean_nll([-9.0] * 99, "stop", **RULE) is None  # too short
    assert turn_mean_nll([-3.0] * 500, "stop", **RULE) is None  # strictly greater
    assert turn_mean_nll([None] * 500, "stop", **RULE) is None
    assert turn_mean_nll(None, "stop", **RULE) is None


def _trace_with(*turns):
    """Nodes + calls shaped like a recorded trace: (sampled, logprobs, finish_reason)."""
    nodes, calls = [], []
    for sampled, logprobs, finish in turns:
        nodes.append(SimpleNamespace(sampled=sampled, logprobs=logprobs))
        if sampled:
            calls.append(
                SimpleNamespace(node=len(nodes) - 1, finish_reason=finish, error=None)
            )
    return SimpleNamespace(nodes=nodes, calls=calls)


def test_find_skips_repetition_turns_and_reports_index():
    trace = _trace_with(
        (False, [], None),
        (True, [-0.5] * 300, "tool_calls"),
        (True, [-8.0] * 300, "repetition"),
        (True, [-6.0] * 200, "length"),
    )
    turn = find_degenerate_turn(trace, **RULE)
    assert (turn.index, turn.node, turn.tokens, turn.mean_nll) == (2, 3, 200, 6.0)
    trace.calls[-1].finish_reason = "repetition"
    assert find_degenerate_turn(trace, **RULE) is None


def test_limits_degenerate_matches_helper():
    assert not RolloutLimits().degenerate([-8.0] * 500, "stop")
    limits = RolloutLimits(degenerate_turn=DegenerateTurnLimit(**RULE))
    assert limits.degenerate([-8.0] * 500, "stop")
    assert not limits.degenerate([-8.0] * 500, "repetition")
    assert not limits.degenerate([-0.5] * 500, "stop")
    assert vf.AgentConfig(degenerate_turn=RULE).degenerate_turn == DegenerateTurnLimit(
        **RULE
    )


def test_repetition_is_recorded_but_served_as_length():
    response = response_from_generate(
        {
            "finish_reason": "repetition",
            "content": "a a a",
            "prompt_ids": [1, 2],
            "completion_ids": [3, 4],
            "completion_logprobs": [-0.1, -0.1],
        },
        "m",
    )
    assert response.finish_reason == "repetition"
    assert (
        serialize_completion(response, "m")["choices"][0]["finish_reason"] == "length"
    )
    trace = _new_trace()
    trace.calls.append(vf.ModelCall(node=0, finish_reason="repetition"))
    assert trace.is_truncated


def _new_trace() -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="x")),
    )


class _FakeClient:
    """Serves scripted generate results, like the renderer TrainClient."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = 0

    async def get_response(self, dialect, body, sampling, **kwargs):
        logprobs, finish = self.turns[self.calls]
        self.calls += 1
        n = len(logprobs)
        response = vf.Response(
            id=f"r{self.calls}",
            created=0,
            model="m",
            message=vf.AssistantMessage(content=f"turn {self.calls}"),
            finish_reason=finish,
            usage=vf.Usage(prompt_tokens=3, completion_tokens=n),
            tokens=TurnTokens(
                prompt_ids=[1, 2, 3],
                completion_ids=list(range(10, 10 + n)),
                completion_logprobs=logprobs,
            ),
        )
        response.raw = serialize_completion(response, "m")
        return response

    async def close(self):
        pass


async def _run_session(turns, limits):
    session = RolloutSession(
        ctx=ModelContext(model="m", client=EvalClientConfig()),
        trace=_new_trace(),
        limits=limits,
    )
    client = _FakeClient(turns)
    statuses, finishes = [], []
    async with InterceptionServer() as server:
        secret, _ = server.register(session)
        session.client = client
        messages = [{"role": "user", "content": "go"}]
        async with httpx.AsyncClient(base_url=server.base_url) as http:
            for _ in turns:
                r = await http.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": messages},
                    headers={"Authorization": f"Bearer {secret}"},
                )
                statuses.append(r.status_code)
                if r.status_code != 200:
                    break
                choice = r.json()["choices"][0]
                finishes.append(choice["finish_reason"])
                messages = [
                    *messages,
                    {"role": "assistant", "content": choice["message"]["content"]},
                    {"role": "user", "content": "more"},
                ]
    return session.trace, client, statuses, finishes


LIMITS = RolloutLimits(degenerate_turn=DegenerateTurnLimit(**RULE))


async def test_server_stops_after_degenerate_turn():
    turns = [([-0.5] * 200, "stop"), ([-8.0] * 300, "length"), ([-0.5] * 200, "stop")]
    trace, client, statuses, _ = await _run_session(turns, LIMITS)
    assert statuses == [200, 400]
    assert client.calls == 2
    assert trace.stop_condition == "degenerate_turn"
    assert trace.num_turns == 2  # the degenerate turn itself is committed
    assert [c.finish_reason for c in trace.calls] == ["stop", "length"]
    # The post-hoc scan finds the same turn on the recorded trace.
    turn = find_degenerate_turn(trace, **RULE)
    assert turn is not None and turn.index == 1 and turn.tokens == 300


@pytest.mark.parametrize(
    "limits",
    [RolloutLimits(), LIMITS],
    ids=["no-limit", "repetition-exempt"],
)
async def test_server_continues(limits):
    turns = [([-8.0] * 300, "repetition"), ([-0.5] * 200, "stop")]
    if limits.degenerate_turn is None:
        turns[0] = ([-8.0] * 300, "length")
    trace, _, statuses, finishes = await _run_session(turns, limits)
    assert statuses == [200, 200]
    assert trace.stop_condition is None
    # A repetition stop reaches the harness as `length` but stays `repetition` on record.
    assert finishes[0] == "length"
    assert trace.calls[0].finish_reason == turns[0][1]
    found = find_degenerate_turn(trace, **RULE)
    if limits.degenerate_turn is None:  # no live limit: only the post-hoc scan flags it
        assert found is not None and found.index == 0
    else:  # exempt both live and post hoc
        assert found is None


def test_graph_commit_keeps_logprobs_for_live_check():
    trace = _new_trace()
    node = graph.prepare_turn(trace, [vf.UserMessage(content="u")]).commit(
        vf.Response(
            id="a",
            created=0,
            model="m",
            message=vf.AssistantMessage(content="a"),
            finish_reason="stop",
            tokens=TurnTokens(
                prompt_ids=[1], completion_ids=[2, 3], completion_logprobs=[-1.0, -2.0]
            ),
        )
    )
    assert trace.nodes[node].sampled and trace.nodes[node].logprobs == [-1.0, -2.0]
