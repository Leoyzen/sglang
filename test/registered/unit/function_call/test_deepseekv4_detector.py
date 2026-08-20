"""Unit tests for DeepSeekV4Detector DSML streaming — no server, no model loading."""

from unittest.mock import patch

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

DSML = "｜DSML｜"


def _wrapped(invoke: str) -> str:
    return f"<{DSML}tool_calls>\n{invoke}\n</{DSML}tool_calls>"


def _invoke(name: str, params: str = "") -> str:
    return f'<{DSML}invoke name="{name}">\n{params}\n</{DSML}invoke>'


def _param(name: str, is_string: str, value: str) -> str:
    return (
        f'<{DSML}parameter name="{name}" string="{is_string}">{value}</{DSML}parameter>'
    )


def _weather_call(city: str = "SF") -> str:
    return _wrapped(_invoke("get_weather", _param("city", "true", city)))


class TestDeepSeekV4Streaming(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            )
        ]

    def _feed(self, chunks):
        """Returns (normal_text, calls) accumulated over the chunks."""
        detector = DeepSeekV4Detector()
        normal, calls = "", []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal += result.normal_text
            calls.extend(result.calls)
        return normal, calls

    def test_preamble_in_same_delta_as_tool_call(self):
        """Prose sharing a delta with the tool call must not be dropped, and the
        streaming and one-shot paths must agree on it."""
        text = "Let me check.\n" + _weather_call()
        normal, calls = self._feed([text])

        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])
        self.assertEqual(
            normal, DeepSeekV4Detector().detect_and_parse(text, self.tools).normal_text
        )

    def test_preamble_before_bare_invoke_without_wrapper(self):
        """The bare `<｜DSML｜invoke …>` form has no tool_calls wrapper to walk
        back to, so the preamble is computed from the invoke itself."""
        text = "Checking.\n" + _invoke("get_weather", _param("city", "true", "SF"))
        normal, calls = self._feed([text])

        self.assertIn("Checking.", normal)
        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])

    def test_no_dsml_markers_leak_into_normal_text(self):
        text = "Prose.\n" + _weather_call()
        normal, _ = self._feed([text[i : i + 4] for i in range(0, len(text), 4)])

        self.assertNotIn(DSML, normal)

    def test_malformed_partial_json_falls_back_to_raw_value(self):
        """A partial non-string parameter must not escape as MalformedJSON."""
        detector = DeepSeekV4Detector()
        result = detector.parse_streaming_increment(
            f'<{DSML}tool_calls>\n<{DSML}invoke name="get_weather">\n'
            f'<{DSML}parameter name="city" string="false">{{"a"',
            self.tools,
        )

        self.assertEqual([c.name for c in result.calls if c.name], ["get_weather"])

    def test_non_streaming_parses_every_tool_calls_section(self):
        """A turn with two tool_calls sections must yield both calls."""
        result = DeepSeekV4Detector().detect_and_parse(
            f"{_weather_call('SF')}\n{_weather_call('NY')}", self.tools
        )

        self.assertEqual(len(result.calls), 2)

    def test_parse_error_neither_swallows_nor_duplicates(self):
        """An unexpected parse error must retain the buffer for retry; only
        the preamble (text before the first DSML tag) is emitted as
        normal_text so the tool-call text is neither swallowed permanently
        nor duplicated across deltas."""
        detector = DeepSeekV4Detector()

        with patch.object(
            DeepSeekV4Detector,
            "_parse_parameters_from_xml",
            side_effect=RuntimeError("boom"),
        ):
            first = detector.parse_streaming_increment(_weather_call(), self.tools)
            # Buffer is retained for retry — NOT cleared
            self.assertNotEqual(detector._buffer, "")

        # Mock removed — the retained buffer should now parse successfully
        # on the next delta, proving the retry works.
        second = detector.parse_streaming_increment(" tail", self.tools)

        # _weather_call() has no preamble, so first.normal_text is empty.
        self.assertEqual(first.calls, [])
        # The tool call is emitted as a call, NOT as normal_text
        self.assertNotIn("get_weather", second.normal_text)
        self.assertTrue(any(c.name == "get_weather" for c in second.calls))


class TestDeepSeekV4FinishLeaks(CustomTestCase):
    """Bug regression: closing DSML sub-tags that are NOT in _DSML_TOOL_TAGS
    leak from finish() when the detector state was reset (e.g. after a parse
    error or when closing tags arrive without their opening counterparts).

    Root cause: _DSML_TOOL_TAGS includes </invoke>, </tool_calls>,
    </function_calls> but omits </parameter>.  When finish() strips from the
    earliest known tag, any </parameter> before that position survives as
    normal_text."""

    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            )
        ]

    def test_closing_parameter_tag_not_in_tool_tags(self):
        """</parameter> must be in _DSML_TOOL_TAGS so finish() can strip it."""
        detector = DeepSeekV4Detector()
        self.assertIn(
            f"</{DSML}parameter",
            detector._DSML_TOOL_TAGS,
            "</｜DSML｜parameter is missing from _DSML_TOOL_TAGS; "
            "finish() cannot strip it and it leaks as normal_text",
        )

    def test_finish_strips_closing_parameter_tag(self):
        """When the detector buffer contains only closing tags (e.g. after the
        reasoning parser split at </think> inside a tool call), finish() must
        strip ALL DSML closing tags including </parameter>."""
        detector = DeepSeekV4Detector()
        # Simulate closing tags arriving without their opening counterparts
        closing_tags = f"</{DSML}parameter>\n</{DSML}invoke>\n</{DSML}tool_calls>"
        detector._buffer = closing_tags
        # State reset as if no tool call was successfully parsed
        detector.current_tool_id = -1
        detector.prev_tool_call_arr = []

        result = detector.finish(self.tools)
        self.assertNotIn(
            DSML,
            result.normal_text,
            "DSML closing tags leaked from finish() — </parameter> is not "
            "in _DSML_TOOL_TAGS so it survives the strip",
        )

    def test_finish_strips_isolated_closing_parameter_tag(self):
        """When </parameter> arrives in a chunk WITHOUT </invoke> or
        </tool_calls>, potentially_dsml is False (</parameter> not in
        _DSML_TOOL_TAGS) and the tag leaks as normal_text during streaming."""
        detector = DeepSeekV4Detector()
        # Feed only the closing parameter tag
        result = detector.parse_streaming_increment(f"</{DSML}parameter>\n", self.tools)
        self.assertNotIn(
            DSML,
            result.normal_text,
            "Isolated </parameter> leaked as normal_text because it is not "
            "in _DSML_TOOL_TAGS and potentially_dsml is False",
        )


class TestDeepSeekV4ParameterTruncation(CustomTestCase):
    """Bug regression: when the reasoning parser splits a tool call at
    </think> (which appears inside a parameter value), the parameter content
    is truncated — text before </think> goes to reasoning_text, text after
    goes to normal_text, and the tool call is never reassembled."""

    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="task",
                    description="Run a subagent task",
                    parameters={
                        "type": "object",
                        "properties": {"prompt": {"type": "string"}},
                        "required": ["prompt"],
                    },
                ),
            )
        ]

    def test_parameter_value_preserved_across_closing_tag_chunks(self):
        """The full parameter value must survive when closing tags arrive in
        separate chunks.  No content should be lost between the parameter
        value and the closing </parameter> tag."""
        detector = DeepSeekV4Detector()
        chunks = [
            f'<{DSML}tool_calls>\n<{DSML}invoke name="task">\n'
            f'<{DSML}parameter name="prompt" string="true">Find how X works',
            f"</{DSML}parameter>\n",
            f"</{DSML}invoke>\n</{DSML}tool_calls>",
        ]
        normal, calls = "", []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal += result.normal_text
            calls.extend(result.calls)
        # Flush
        result = detector.finish(self.tools)
        normal += result.normal_text

        # The tool call should be parsed with the full parameter value
        tool_calls_with_name = [c for c in calls if c.name]
        self.assertEqual(len(tool_calls_with_name), 1)
        self.assertEqual(tool_calls_with_name[0].name, "task")

        # Collect all parameter fragments
        param_fragments = [c.parameters for c in calls if c.parameters]
        full_params = "".join(param_fragments)
        self.assertIn("Find how X works", full_params)
        self.assertNotIn(DSML, normal)


class TestDeepSeekV4MultiToolCallStreaming(CustomTestCase):
    """Multi-tool-call streaming: verify the detector correctly separates
    arguments for multiple tool calls using tool_index.

    Regression: the concurrent live test reported "multi-tool-call args
    concatenation" as a bug, but the detector was returning correct
    tool_index values. The bug was in the test script not using the
    index field. These tests confirm the DETECTOR handles it correctly.
    """

    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="save_text",
                    description="Save text",
                    parameters={
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                ),
            ),
        ]

    def _feed(self, chunks):
        """Returns (normal_text, calls) accumulated over the chunks."""
        detector = DeepSeekV4Detector()
        normal, calls = "", []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal += result.normal_text
            calls.extend(result.calls)
        return normal, calls

    def test_two_tool_calls_single_chunk(self):
        """Two invoke blocks in a single chunk must produce separate
        ToolCallItems with correct tool_index values."""
        text = _wrapped(
            _invoke("get_weather", '{"city": "Shanghai"}')
            + "\n"
            + _invoke("save_text", '{"text": "Meeting at 3pm"}')
        )
        normal, calls = self._feed([text])

        # Should have 4 call items: name+args for each tool
        names = [c.name for c in calls if c.name]
        self.assertEqual(names, ["get_weather", "save_text"])

        # Group args by tool_index
        args_by_idx: dict[int, str] = {}
        for c in calls:
            if c.parameters:
                idx = c.tool_index
                args_by_idx[idx] = args_by_idx.get(idx, "") + c.parameters

        # Each tool index should have valid, separate JSON
        self.assertEqual(len(args_by_idx), 2)
        import json as _json

        args_0 = _json.loads(args_by_idx[0])
        args_1 = _json.loads(args_by_idx[1])
        self.assertEqual(args_0, {"city": "Shanghai"})
        self.assertEqual(args_1, {"text": "Meeting at 3pm"})

    def test_two_tool_calls_streamed_separately(self):
        """Two invoke blocks arriving in separate chunks must also produce
        separate ToolCallItems with correct tool_index."""
        chunk1 = _wrapped(_invoke("get_weather", '{"city": "Shanghai"}'))
        chunk2 = (
            "\n"
            + _invoke("save_text", '{"text": "Meeting"}')
            + f"\n</{DSML}tool_calls>"
        )
        normal, calls = self._feed([chunk1, chunk2])

        names = [c.name for c in calls if c.name]
        self.assertEqual(names, ["get_weather", "save_text"])

        args_by_idx: dict[int, str] = {}
        for c in calls:
            if c.parameters:
                idx = c.tool_index
                args_by_idx[idx] = args_by_idx.get(idx, "") + c.parameters

        import json as _json

        self.assertEqual(len(args_by_idx), 2)
        self.assertEqual(_json.loads(args_by_idx[0]), {"city": "Shanghai"})
        self.assertEqual(_json.loads(args_by_idx[1]), {"text": "Meeting"})

    def test_two_tool_calls_interleaved_chunks(self):
        """Second invoke's JSON body split across chunks — args must
        still be routed to the correct tool_index."""
        # Construct chunks manually to avoid _wrapped adding closing tags
        # to incomplete invoke blocks.
        chunk1 = (
            f"<{DSML}tool_calls>\n"
            + _invoke("get_weather", '{"city": "Shanghai"}')
            + f'\n<{DSML}invoke name="save_text">\n'
            + '{"text": "Meet'
        )
        chunk2 = 'ing at 3pm"}\n' + f"</{DSML}invoke>\n</{DSML}tool_calls>"
        normal, calls = self._feed([chunk1, chunk2])

        names = [c.name for c in calls if c.name]
        self.assertEqual(names, ["get_weather", "save_text"])

        args_by_idx: dict[int, str] = {}
        for c in calls:
            if c.parameters:
                idx = c.tool_index
                args_by_idx[idx] = args_by_idx.get(idx, "") + c.parameters

        import json as _json

        self.assertEqual(len(args_by_idx), 2)
        self.assertEqual(_json.loads(args_by_idx[0]), {"city": "Shanghai"})
        self.assertEqual(_json.loads(args_by_idx[1]), {"text": "Meeting at 3pm"})


if __name__ == "__main__":
    import unittest

    unittest.main()
