"""Measure per-sample and mean gradient norms at initialisation, per target.

Tells us whether the non-private control's divergence is a real property of the
unclipped gradient (huge per-sample norms) or an artefact of the harness.
"""

import sys

sys.path.insert(0, "/home/psaunder/Documents/Masters/differentiable-privacy-percentages/src")


import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


def probe(target_name, batch_size=250, T=5000):
    from conf.scope import RunContext, using
    from conf.singleton_conf import SingletonConfig
    from environments.dp_params import DPTrainingParams
    from util.dataloaders import get_dataset_shapes
    from util.transfer import TargetSpec, build_target_config

    spec = TargetSpec(name=target_name, eps=10.0, delta=1e-7, T=T, arch="")
    config = build_target_config(spec, batch_size)

    with SingletonConfig.override(config), using(RunContext(config)):
        get_dataset_shapes()
        env_params = DPTrainingParams.create_direct_from_config()
        loader = env_params.loader

        from environments.dp import per_example_loss_and_grads
        from util.util import reinit_model

        model = reinit_model(env_params.network, jr.PRNGKey(0))
        n_params = sum(x.size for x in jax.tree.leaves(eqx_arrays(model)))

        idx = np.sort(
            np.random.default_rng(0).choice(loader.n_train, size=batch_size, replace=False)
        )
        bx, by = loader.load_train_batch(idx)
        bx, by = jnp.asarray(bx), jnp.asarray(by)

        per_losses, grads = per_example_loss_and_grads(model, bx, by)
        leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(grads)]
        # per-sample global norm (leading axis is the sample axis)
        sq = np.zeros(batch_size)
        for leaf in leaves:
            sq += (leaf.reshape(batch_size, -1) ** 2).sum(axis=1)
        per_norms = np.sqrt(sq)

        mean_g = [leaf.sum(axis=0) / batch_size for leaf in leaves]
        mean_norm = float(np.sqrt(sum((g**2).sum() for g in mean_g)))

        x = np.asarray(bx)
        print(f"\n=== {target_name} ===")
        print(f"  params={n_params:,}  sample_shape={loader.sample_shape}")
        print(f"  x: min={x.min():.3f} max={x.max():.3f} mean={x.mean():.3f} std={x.std():.3f}")
        print(f"  initial per-sample loss: mean={float(per_losses.mean()):.4f}")
        print(
            f"  per-sample grad norm: min={per_norms.min():.4g} median={np.median(per_norms):.4g} "
            f"max={per_norms.max():.4g}"
        )
        print(
            f"  ||mean gradient|| (unclipped, the non-private update direction) = {mean_norm:.6g}"
        )
        for C in (0.1, 1.0, 5.0):
            factor = np.minimum(1.0, C / np.maximum(per_norms, 1e-12))
            cg = [
                (leaf.reshape(batch_size, -1) * factor[:, None]).sum(axis=0) / batch_size
                for leaf in leaves
            ]
            cn = float(np.sqrt(sum((g**2).sum() for g in cg)))
            print(f"  ||mean gradient|| clipped at C={C:<5g} = {cn:.6g}")


def eqx_arrays(model):
    import equinox as eqx

    arrays, _ = eqx.partition(model, eqx.is_array)
    return arrays


if __name__ == "__main__":
    for t in sys.argv[1:] or ["imagenet", "eyepacs", "chexpert"]:
        probe(t)
