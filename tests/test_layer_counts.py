"""Test that get_layer_count returns correct values for known models."""

import pytest

from nl_probes.utils.common import get_layer_count

# Known layer counts to verify against
KNOWN_LAYER_COUNTS = {
    "Qwen/Qwen3-8B": 36,
}


@pytest.mark.parametrize("model_name,expected_layers", KNOWN_LAYER_COUNTS.items())
def test_get_layer_count(model_name: str, expected_layers: int):
    """Verify get_layer_count matches known layer counts from HuggingFace configs."""
    actual_layers = get_layer_count(model_name)
    assert actual_layers == expected_layers, f"{model_name}: expected {expected_layers}, got {actual_layers}"
