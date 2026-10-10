import jax.numpy as jnp
import numpy as np
from brax.training.acme import running_statistics, specs

from g1_wbc.ppo_teacher import unfreeze_constant_dims


def test_frozen_dims_reset_to_unit_std():
    """M12: a dim that never varied must not normalise a first real value to 1e5."""
    st = running_statistics.init_state({"p": specs.Array((2,), jnp.float32)})
    x = {"p": jnp.stack([jnp.zeros(1000), jnp.linspace(-1, 1, 1000)], -1)}
    st = running_statistics.update(st, x)
    assert float(st.std["p"][0]) < 1e-3
    st2 = unfreeze_constant_dims(st)
    assert np.isclose(float(st2.std["p"][0]), 1.0)
    assert np.isclose(float(st2.std["p"][1]), float(st.std["p"][1]))
