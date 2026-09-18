from sglang.srt.configs.deepseek_v41 import (
    DeepseekV41Config,
    dsv41_vision_enabled,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _vision_config(**overrides):
    """A V4.1 config carrying a vision tower, as produced from the HF schema."""
    values = {
        "model_type": "deepseek_v41",
        "vision_n_layers": 32,
    }
    values.update(overrides)
    return DeepseekV41Config(**values)


def test_vision_tower_defaults_to_enabled():
    config = _vision_config()
    assert dsv41_vision_enabled(config)
    assert not getattr(config, "language_model_only", False)


def test_language_model_only_disables_vision():
    config = _vision_config(language_model_only=True)
    assert not dsv41_vision_enabled(config)


def test_text_only_checkpoint_has_no_vision():
    config = DeepseekV41Config(model_type="deepseek_v41")
    assert config.vision_n_layers == 0
    assert not dsv41_vision_enabled(config)


def test_other_model_types_never_build_a_tower():
    class OtherConfig:
        model_type = "deepseek_v4"
        vision_n_layers = 32
        language_model_only = False

    assert not dsv41_vision_enabled(OtherConfig())


def test_missing_attributes_fail_closed():
    class Bare:
        model_type = "deepseek_v41"

    assert not dsv41_vision_enabled(Bare())
