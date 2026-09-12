from functools import lru_cache
from typing import List, Literal, Optional, Union

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.function_call.base_format_detector import (
    BaseFormatDetector,
    StructuralTag,
    get_model_structural_tag,
)
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector

# Registered by xgrammar >= 0.2.6 builds that carry the spaced DSML grammar
# (upstream mlc-ai/xgrammar#885). Older xgrammar raises on unknown model names.
_DSV41_XGRAMMAR_MODEL = "deepseek_v4_1"


@lru_cache(maxsize=1)
def _dsv41_native_structural_tag_available() -> bool:
    """Probe once for xgrammar's builtin deepseek_v4_1 structural tag.

    The import can succeed while the model name stays unknown on older
    xgrammar, so the registration itself must be probed; fall back if absent.
    """
    if get_model_structural_tag is None:
        return False
    try:
        get_model_structural_tag(
            model=_DSV41_XGRAMMAR_MODEL, tools=[], tool_choice="auto"
        )
        return True
    except Exception:
        return False


class DeepSeekV41Detector(DeepSeekV32Detector):
    """DeepSeek V4.1 DSML detector.

    Tag names have a leading space: " calls", " invoke", and " parameter".
    """

    tool_calls_block_name = " calls"
    invoke_tag_name = " invoke"
    parameter_tag_name = " parameter"

    # The encoder joins an assistant turn's content and its calls block with a
    # blank line, and renders it even when there is no content.
    tool_calls_prefix = "\n\n"
    think_end_token = "</think>"

    def get_structural_tag_name(self) -> Optional[str]:
        # xgrammar's builtin "deepseek_v4" tag hardcodes the unspaced names;
        # the spaced "deepseek_v4_1" grammar (style deepseek_v4_1_xml, matching
        # the prompt-trained format) only exists in newer xgrammar builds.
        if _dsv41_native_structural_tag_available():
            return _DSV41_XGRAMMAR_MODEL
        return None

    def supports_structural_tag(self) -> bool:
        return True

    def get_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> Optional[StructuralTag]:
        if _dsv41_native_structural_tag_available():
            # Delegate to xgrammar's builtin deepseek_v4_1 tag: its
            # deepseek_v4_1_xml body constrains string values with a
            # TagDispatch that forbids "</｜DSML｜ parameter>", structurally
            # preventing DSML fragments leaking into tool arguments.
            return BaseFormatDetector.get_structural_tag(
                self,
                tools=tools,
                tool_choice=tool_choice,
                thinking_mode=thinking_mode,
                parallel_tool_calls=parallel_tool_calls,
            )
        return self._build_spaced_json_body_structural_tag(
            tools=tools,
            tool_choice=tool_choice,
            thinking_mode=thinking_mode,
            parallel_tool_calls=parallel_tool_calls,
        )

    def _build_spaced_json_body_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> Optional[StructuralTag]:
        """Fallback for older xgrammar: the builtin "deepseek_v4" shape with
        the spaced tag names.

        Bodies are JSON: xgrammar's "deepseek_xml" body style also hardcodes
        the unspaced " parameter" name, and the V3.2-lineage parser accepts a
        JSON body inside an invoke.

        Known weakness: a raw-JSON body cannot forbid DSML-looking text inside
        string values, so under strict constraints the model may leak
        "</｜DSML｜ parameter>" fragments into arguments (the model is trained
        to close XML-style parameters). Prefer the native deepseek_v4_1 tag.
        """
        try:
            from xgrammar.structural_tag import (
                AnyTextFormat,
                ConstStringFormat,
                JSONSchemaFormat,
                OrFormat,
                SequenceFormat,
                StructuralTag,
                TagFormat,
                TagsWithSeparatorFormat,
                TriggeredTagsFormat,
            )
        except ImportError:
            return None

        tools = list(tools or [])
        if isinstance(tool_choice, ToolChoice):
            tools = [
                tool
                for tool in tools
                if tool.function.name == tool_choice.function.name
            ]
            if len(tools) != 1:
                return None
        if not tools:
            return None

        def invoke_tag(tool: Tool) -> TagFormat:
            function = tool.function
            schema = function.parameters if function.strict else True
            if schema is None:
                schema = True
            return TagFormat(
                begin=f'{self.invoke_start_token} name="{function.name}">',
                content=JSONSchemaFormat(json_schema=schema),
                end=f"{self.invoke_end_token}\n",
            )

        tags = [invoke_tag(tool) for tool in tools]
        if isinstance(tool_choice, ToolChoice):
            calls = tags[0]
        elif parallel_tool_calls:
            calls = TagsWithSeparatorFormat(tags=tags, separator="", at_least_one=True)
        else:
            calls = OrFormat(elements=tags)
        block_begin = f"{self.bot_token}\n"

        if tool_choice == "auto":
            body = TriggeredTagsFormat(
                triggers=[self.bot_token],
                tags=[TagFormat(begin=block_begin, content=calls, end=self.eot_token)],
                excludes=["<think>", self.think_end_token],
            )
        else:
            body = SequenceFormat(
                elements=[
                    ConstStringFormat(value=self.tool_calls_prefix + block_begin),
                    calls,
                    ConstStringFormat(value=self.eot_token),
                ]
            )
        if not thinking_mode:
            return StructuralTag(format=body)
        reasoning = TagFormat(
            begin="", content=AnyTextFormat(), end=self.think_end_token
        )
        return StructuralTag(format=SequenceFormat(elements=[reasoning, body]))
