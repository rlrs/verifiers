from copy import deepcopy

from pydantic import Field

import verifiers.v1 as vf

PROMPT = (
    "Use web search to inspect the front page of example.com. "
    "Answer in one sentence and include a markdown source link."
)
GERMAN_EXAMPLE_URL = "https://de.wikipedia.org/wiki/Example.com"


class GermanRewrite(vf.StrictBaseModel):
    text_de: str
    citation_url_de: str
    citation_title_de: str


class InterceptWebSearchTask(vf.Task):
    expected_url: str


class InterceptWebSearchConfig(vf.TasksetConfig):
    rewrite: vf.JudgeConfig = Field(
        default_factory=lambda: vf.JudgeConfig(
            model="gpt-4.1-mini",
            base_url="https://api.openai.com/v1",
            api_key_var="OPENAI_API_KEY",
        )
    )
    """Model config for the interceptor's German rewrite call."""


class InterceptWebSearchTaskset(
    vf.Taskset[InterceptWebSearchTask, InterceptWebSearchConfig]
):
    def load_tasks(self) -> list[InterceptWebSearchTask]:
        return [
            InterceptWebSearchTask(
                idx=0,
                prompt=PROMPT,
                expected_url=GERMAN_EXAMPLE_URL,
            )
        ]

    @vf.intercept
    async def german_rewrite(
        self,
        response: vf.Response,
        trace: vf.Trace,
    ) -> vf.Response:
        original_text = response.message.content or ""
        provider_state = response.message.provider_state or []
        judge = vf.Judge(self.config.rewrite)
        result = await judge.complete(
            (
                "Rewrite the assistant answer into natural German. Return JSON with "
                "text_de, citation_url_de, and citation_title_de. Keep the factual meaning, "
                "include exactly one markdown source link in text_de, and choose a "
                f"German-language citation URL when available. Prefer {GERMAN_EXAMPLE_URL} "
                "for Example.com.\n\n"
                f"Original answer:\n{original_text}\n\nProvider state:\n{provider_state}"
            ),
            trace=trace,
            schema=GermanRewrite,
            temperature=0,
        )
        rewrite = result.parsed
        if rewrite is None:
            raise RuntimeError("rewrite model returned no GermanRewrite object")
        response.message.content = rewrite.text_de
        output = deepcopy(provider_state)
        start = rewrite.text_de.find(rewrite.citation_url_de)
        annotation = {
            "type": "url_citation",
            "start_index": max(start, 0),
            "end_index": max(start, 0) + len(rewrite.citation_url_de),
            "title": rewrite.citation_title_de,
            "url": rewrite.citation_url_de,
        }
        for item in output:
            if item.get("type") == "web_search_call":
                item.setdefault("action", {})["sources"] = [
                    {"type": "url", "url": rewrite.citation_url_de}
                ]
            if item.get("type") == "message":
                for part in item.get("content") or []:
                    if part.get("type") == "output_text":
                        part["text"] = rewrite.text_de
                        part["annotations"] = [annotation]
        response.message.provider_state = output
        trace.info["intercept_rewrite"] = rewrite.model_dump()
        return response

    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def saw_german_rewrite(
        self, task: InterceptWebSearchTask, trace: vf.Trace
    ) -> float:
        reply = trace.last_reply or ""
        return float(task.expected_url in reply and "Die " in reply)
