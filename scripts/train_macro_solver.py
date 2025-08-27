"""Train MacroFinanceNet (Try.md idea) with Euler scan and Optax.

Usage:
  python scripts/train_macro_solver.py --epochs 2000 --paths 4096 --dt 0.001

Notes:
- This script assumes JAX + Equinox + Optax are installed.
- It uses the equations sketch from bsde_dsgE.models.probab01_equations.
"""

from __future__ import annotations

import argparse
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from bsde_dsgE.models.macro_solver import Config as NetCfg, MacroFinanceNet
from bsde_dsgE.models.probab01_equations import Config as EqCfg, compute_dynamics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--paths", type=int, default=4096)
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--steps", type=int, default=0, help="If >0, use multi-step scan of this length")
    p.add_argument("--J", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=str, default="", help="Optional path to save trained model via Equinox")
    p.add_argument("--eval-symmetric", action="store_true", help="Print symmetric-state q after training")
    return p.parse_args()


def symmetric_omegas(cfg: EqCfg, etas=(0.5,)) -> jnp.ndarray:
    J = cfg.J
    z = jnp.ones((len(etas), J)) / float(J)
    z = z[:, : J - 1]
    e = jnp.stack([jnp.ones((cfg.N_ETA,)) * float(val) for val in etas], axis=0)
    return jnp.hstack([e, z])


def loss_fn(model: MacroFinanceNet, key: jax.Array, eq_cfg: EqCfg, *, paths: int, dt: float, steps: int) -> jax.Array:
    B = paths
    # Init Omega from uniform ranges (eta in [0.2,0.8], zeta ~ simplex)
    key_eta, key_dir, key_dW = jax.random.split(key, 3)
    eta = jax.random.uniform(key_eta, (B, eq_cfg.N_ETA), minval=0.2, maxval=0.8)
    raw = jax.random.uniform(key_dir, (B, eq_cfg.J))
    zeta_full = raw / jnp.sum(raw, axis=1, keepdims=True)
    zeta = zeta_full[:, : eq_cfg.N_ZETA]
    Omega = jnp.hstack([eta, zeta])

    if steps <= 1:
        # One-step consistency
        dW = jax.random.normal(key_dW, (B, eq_cfg.J)) * jnp.sqrt(dt)
        q, sigma_q, r = model(Omega)
        drift_X, vol_X, h, Z = compute_dynamics(eq_cfg, Omega, q, sigma_q, r)
        stoch_X = jnp.einsum("bij,bi->bj", vol_X, dW)
        stoch_Y = jnp.einsum("bij,bi->bj", Z, dW)
        Omega_1 = Omega + drift_X * dt + stoch_X
        q_1 = q - h * dt + stoch_Y
        q_hat_1, _, _ = model(Omega_1)
        return jnp.mean(jnp.sum((q_hat_1 - q_1) ** 2, axis=1))
    else:
        # Multi-step scan
        dWs = jax.random.normal(key_dW, (steps, B, eq_cfg.J)) * jnp.sqrt(dt)
        def scan_fn(carry, dW_i):
            Om, _ = carry
            q_i, sig_i, r_i = model(Om)
            drift_X, vol_X, h, Z = compute_dynamics(eq_cfg, Om, q_i, sig_i, r_i)
            stoch_X = jnp.einsum("bij,bi->bj", vol_X, dW_i)
            stoch_Y = jnp.einsum("bij,bi->bj", Z, dW_i)
            Om1 = Om + drift_X * dt + stoch_X
            q1 = q_i - h * dt + stoch_Y
            q_hat1, _, _ = model(Om1)
            loss_i = jnp.mean(jnp.sum((q_hat1 - q1) ** 2, axis=1))
            return (Om1, q1), loss_i
        (_, _), losses = jax.lax.scan(scan_fn, (Omega, jnp.zeros((B, eq_cfg.J))), dWs)
        return jnp.mean(losses)


def main() -> None:
    args = parse_args()
    key = jax.random.PRNGKey(args.seed)
    net_cfg = NetCfg(J=args.J)
    eq_cfg = EqCfg(J=args.J)
    model = MacroFinanceNet(net_cfg, key)

    schedule = optax.exponential_decay(init_value=args.lr, transition_steps=1000, decay_rate=0.95, end_value=1e-6)
    opt = optax.adam(schedule)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step(model, opt_state, key):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, key, eq_cfg, paths=args.paths, dt=args.dt, steps=args.steps)
        updates, opt_state = opt.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    print({"epochs": args.epochs, "paths": args.paths, "dt": args.dt, "J": args.J})
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        key, k = jax.random.split(key)
        model, opt_state, loss = train_step(model, opt_state, k)
        if ep == 1:
            print(f"Compiled in {time.time() - t0:.2f}s")
            t0 = time.time()
        if ep % 200 == 0 or ep == args.epochs:
            print(f"epoch {ep}: loss={float(loss):.6e} (elapsed {time.time() - t0:.2f}s)")
            t0 = time.time()

    if args.save:
        eqx.tree_serialise_leaves(args.save, model)
        print({"saved": args.save})
    if args.eval_symmetric:
        etas = (0.3, 0.4, 0.5, 0.6, 0.7)
        Omega_sym = symmetric_omegas(eq_cfg, etas)
        q, sigma_q, r = model(Omega_sym)
        print("q@sym:", np.array(q))


if __name__ == "__main__":
    main()
