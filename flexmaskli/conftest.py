import pytest
import torch


@pytest.fixture(autouse=True)
def reset_dynamo():
    """Reset torch._dynamo state and config between tests."""
    saved = {
        "recompile_limit": torch._dynamo.config.recompile_limit,
        "fail_on_recompile_limit_hit": torch._dynamo.config.fail_on_recompile_limit_hit,
        "accumulated_recompile_limit": torch._dynamo.config.accumulated_recompile_limit,
    }
    torch._dynamo.reset()
    torch._dynamo.config.accumulated_recompile_limit = 10_000_000
    yield
    torch._dynamo.reset()
    for k, v in saved.items():
        setattr(torch._dynamo.config, k, v)
