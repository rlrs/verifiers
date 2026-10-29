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
from verifiers.v1.degenerate import find_degenerate_turn, loop_start, turn_mean_nll
from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.session import RolloutLimits, RolloutSession
from verifiers.v1.types import TurnTokens

RULE = {"min_tokens": 100, "max_mean_nll": 3.0}


def _turn(prefix: int, prefix_lp: float, loop: int = 0, period: int = 20):
    """Sampled tokens + logprobs: `prefix` distinct tokens, then a `period`-token loop
    (logprob -0.01) for `loop` tokens, ending mid-period."""
    tokens = list(range(1000, 1000 + prefix)) + [7 + i % period for i in range(loop)]
    return tokens, [prefix_lp] * prefix + [-0.01] * loop


def test_rule_thresholds():
    assert turn_mean_nll([-8.0] * 4000, "length", **RULE) == (4000, 8.0)
    assert turn_mean_nll([-8.0] * 4000, "stop", **RULE) is not None
    assert turn_mean_nll([-8.0] * 4000, None, **RULE) is not None
    assert turn_mean_nll([-9.0] * 99, "stop", **RULE) is None  # too short
    assert turn_mean_nll([-3.0] * 500, "stop", **RULE) is None  # strictly greater
    assert turn_mean_nll([None] * 500, "stop", **RULE) is None
    assert turn_mean_nll(None, "stop", **RULE) is None


def test_loop_start_strips_full_and_partial_repetitions():
    tokens, _ = _turn(300, -1.0, loop=170, period=20)  # 8.5 periods of 20
    assert loop_start(tokens, 16, 64) == 300
    # A short base period is found through its multiple in the range (3 -> 18).
    short = list(range(1000, 1300)) + [1, 2, 3] * 60
    assert loop_start(short, 16, 64) == 300
    # No multiple of the period in range, or no repetition at all: nothing is stripped.
    assert loop_start(tokens, 41, 59) == len(tokens)
    assert loop_start(list(range(500)), 16, 64) == 500


def test_repetition_turn_judged_on_tokens_before_the_loop():
    # Gibberish then a loop: the whole-turn mean (~2.4) passes, the prefix (8.0) does not.
    tokens, lps = _turn(300, -8.0, loop=700)
    assert turn_mean_nll(lps, "length", **RULE) is None
    assert turn_mean_nll(lps, "repetition", token_ids=tokens, **RULE) == (300, 8.0)
    # A healthy prefix stays healthy however the cut changed the mean.
    tokens, lps = _turn(300, -0.5, loop=200)
    assert turn_mean_nll(lps, "repetition", token_ids=tokens, **RULE) is None
    # A prefix shorter than min_tokens is not judged.
    tokens, lps = _turn(80, -8.0, loop=400)
    assert turn_mean_nll(lps, "repetition", token_ids=tokens, **RULE) is None
    # The pattern range decides what counts as the loop.
    tokens, lps = _turn(300, -8.0, loop=700, period=100)
    assert turn_mean_nll(lps, "repetition", token_ids=tokens, **RULE) is None
    assert turn_mean_nll(
        lps, "repetition", token_ids=tokens, pattern_sizes=(16, 128), **RULE
    ) == (300, 8.0)


def _node(tokens, logprobs, sampled=True):
    """A message node: 2 scaffold tokens, then the sampled ones under `mask`."""
    return SimpleNamespace(
        sampled=sampled,
        token_ids=[0, 1, *tokens],
        mask=[False, False, *([True] * len(tokens))],
        logprobs=logprobs,
    )


def _trace_with(*turns):
    """Nodes + calls shaped like a recorded trace: (sampled, (tokens, logprobs), finish)."""
    nodes, calls = [], []
    for sampled, (tokens, logprobs), finish in turns:
        nodes.append(_node(tokens, logprobs, sampled))
        if sampled:
            calls.append(
                SimpleNamespace(node=len(nodes) - 1, finish_reason=finish, error=None)
            )
    return SimpleNamespace(nodes=nodes, calls=calls)


def test_find_judges_repetition_turns_on_their_prefix():
    trace = _trace_with(
        (False, ([], []), None),
        (True, _turn(300, -0.5), "tool_calls"),
        (True, _turn(300, -0.5, loop=700), "repetition"),
        (True, _turn(300, -8.0, loop=700), "repetition"),
        (True, _turn(200, -6.0), "length"),
    )
    turn = find_degenerate_turn(trace, **RULE)
    assert (turn.index, turn.node, turn.tokens, turn.mean_nll) == (2, 3, 300, 8.0)
    trace.calls[2].finish_reason = "length"  # judged whole: mean ~2.4, passes
    turn = find_degenerate_turn(trace, **RULE)
    assert (turn.index, turn.node, turn.tokens) == (3, 4, 200)


def test_limits_degenerate_matches_helper():
    assert not RolloutLimits().degenerate(_node(*_turn(500, -8.0)), "stop")
    limits = RolloutLimits(degenerate_turn=DegenerateTurnLimit(**RULE))
    assert limits.degenerate(_node(*_turn(500, -8.0)), "stop")
    assert not limits.degenerate(_node(*_turn(500, -0.5)), "stop")
    assert limits.degenerate(_node(*_turn(300, -8.0, loop=700)), "repetition")
    assert not limits.degenerate(_node(*_turn(300, -8.0, loop=700)), "length")
    wide = RolloutLimits(
        degenerate_turn=DegenerateTurnLimit(**RULE, max_pattern_size=128)
    )
    looped = _node(*_turn(300, -8.0, loop=700, period=100))
    assert not limits.degenerate(looped, "repetition")
    assert wide.degenerate(looped, "repetition")
    assert vf.AgentConfig(degenerate_turn=RULE).degenerate_turn == DegenerateTurnLimit(
        **RULE
    )
    with pytest.raises(ValueError):
        DegenerateTurnLimit(min_pattern_size=65)


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
        tokens, logprobs, finish = self.turns[self.calls]
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
                completion_ids=tokens,
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


@pytest.mark.parametrize(
    "turn",
    [(*_turn(300, -8.0), "length"), (*_turn(300, -8.0, loop=700), "repetition")],
    ids=["length", "repetition-prefix"],
)
async def test_server_stops_after_degenerate_turn(turn):
    turns = [(*_turn(200, -0.5), "stop"), turn, (*_turn(200, -0.5), "stop")]
    trace, client, statuses, _ = await _run_session(turns, LIMITS)
    assert statuses == [200, 400]
    assert client.calls == 2
    assert trace.stop_condition == "degenerate_turn"
    assert trace.num_turns == 2  # the degenerate turn itself is committed
    assert [c.finish_reason for c in trace.calls] == ["stop", turn[2]]
    # The post-hoc scan finds the same turn on the recorded trace.
    found = find_degenerate_turn(trace, **RULE)
    assert found is not None and found.index == 1 and found.tokens == 300


@pytest.mark.parametrize(
    ("limits", "turn"),
    [
        (RolloutLimits(), (*_turn(300, -8.0), "length")),
        (LIMITS, (*_turn(300, -0.5, loop=700), "repetition")),
    ],
    ids=["no-limit", "repetition-healthy-prefix"],
)
async def test_server_continues(limits, turn):
    turns = [turn, (*_turn(200, -0.5), "stop")]
    trace, _, statuses, finishes = await _run_session(turns, limits)
    assert statuses == [200, 200]
    assert trace.stop_condition is None
    # A repetition stop reaches the harness as `length` but stays `repetition` on record.
    assert finishes[0] == "length"
    assert trace.calls[0].finish_reason == turn[2]
    found = find_degenerate_turn(trace, **RULE)
    if limits.degenerate_turn is None:  # no live limit: only the post-hoc scan flags it
        assert found is not None and found.index == 0
    else:  # healthy before the loop: neither check flags it
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
