# Copyright 2025 TOYOTA MOTOR CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MNIST DP-SGD expressed with ordinary JAX transformations.

This is an executable design probe for ``privjail.jax``.  In particular:

* ``jax.grad(loss_one)`` differentiates one record's scalar loss.
* ``jax.vmap`` batches that gradient to produce per-example gradients.
* ``clip_norm`` globally clips each example across the whole parameter PyTree.
* ``gaussian_mechanism`` adds noise to the clipped sum.
* ``dp_sgd_step`` puts clipping, noise, and the update in one jitted graph.

The functions named ``clip_norm`` and ``gaussian_mechanism`` are written in raw
JAX here, but are intended to become trusted ``privjail.jax`` primitives.  The
raw implementation is expanded into lower-level primitives while tracing; use
``--print-jaxpr`` to see that expansion.

JAX random keys are explicit inputs because ordinary JAX functions are pure.
In PrivJail, the sampling and noise keys must instead be owned and injected by
the trusted server; accepting client-controlled keys would invalidate the DP
guarantee.

This example demonstrates the numerical computation and accounting shape.  Raw
JAX does not enforce PrivJail metadata rules or a privacy boundary, so this file
must not itself be treated as a DP enforcement mechanism.

Examples:

    uv run python examples/dpsgd_jax.py --jaxpr-only --print-jaxpr
    uv run python examples/dpsgd_jax.py --num-steps 100
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np


Array = jax.Array
Params = dict[str, Array]
Gradients = dict[str, Array]
PyTree = Any


class NoiseState(NamedTuple):
    """Explicit raw-JAX state; a PrivJail implementation must keep it trusted."""

    key: Array


class SamplingState(NamedTuple):
    """Explicit raw-JAX state; a PrivJail implementation must keep it trusted."""

    key: Array


class PrivacyBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class GaussianEvent:
    """Accounting description of one (possibly sampled) Gaussian release."""

    sensitivity: float
    noise_scale: float
    sampling_rate: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.sensitivity) or self.sensitivity <= 0:
            raise ValueError("sensitivity must be finite and > 0")
        if not math.isfinite(self.noise_scale) or self.noise_scale <= 0:
            raise ValueError("noise_scale must be finite and > 0")
        if not 0 < self.sampling_rate <= 1:
            raise ValueError("sampling_rate must be in (0, 1]")

    def rdp_epsilon(self, order: int) -> float:
        """Return the integer-order RDP cost used by PrivJail today."""

        if order < 2:
            raise ValueError("RDP order must be an integer >= 2")

        c = self.sensitivity**2 / (2.0 * self.noise_scale**2)
        if self.sampling_rate == 1.0:
            return order * c

        q = self.sampling_rate
        log_terms = []
        for k in range(order + 1):
            log_binomial = (
                math.lgamma(order + 1) - math.lgamma(k + 1) - math.lgamma(order - k + 1)
            )
            log_terms.append(
                log_binomial
                + (order - k) * math.log1p(-q)
                + k * math.log(q)
                + (k * k - k) * c
            )
        maximum = max(log_terms)
        log_a = maximum + math.log(sum(math.exp(term - maximum) for term in log_terms))
        return log_a / (order - 1)


@dataclass
class RdpAccountant:
    """Small host-side accountant component for this standalone example.

    A future remote execution path should authorize and compose the event on the
    server before returning a public result.  It should not trust this Python
    object merely because the numerical computation is jitted.
    """

    orders: tuple[int, ...]
    epsilon_limit: float
    delta: float
    _spent: dict[int, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.orders or len(set(self.orders)) != len(self.orders):
            raise ValueError("orders must be non-empty and unique")
        if any(order < 2 for order in self.orders):
            raise ValueError("orders must contain only integers >= 2")
        if not math.isfinite(self.epsilon_limit) or self.epsilon_limit <= 0:
            raise ValueError("epsilon_limit must be finite and > 0")
        if not 0 < self.delta < 1:
            raise ValueError("delta must be in (0, 1)")
        self._spent = dict.fromkeys(self.orders, 0.0)

    @property
    def epsilon(self) -> float:
        if not any(value > 0 for value in self._spent.values()):
            return 0.0
        return min(
            self._spent[order] + math.log(1.0 / self.delta) / (order - 1)
            for order in self.orders
        )

    def compose(self, event: GaussianEvent) -> None:
        """Charge one event, refusing it before its public result is produced."""

        candidate = {
            order: self._spent[order] + event.rdp_epsilon(order)
            for order in self.orders
        }
        epsilon = min(
            candidate[order] + math.log(1.0 / self.delta) / (order - 1)
            for order in self.orders
        )
        if epsilon > self.epsilon_limit:
            raise PrivacyBudgetExceeded(
                f"next event would spend epsilon={epsilon:.6g}, "
                f"above the limit {self.epsilon_limit:.6g}"
            )
        self._spent = candidate


def load_mnist_data(
    data_dir: Path,
) -> tuple[Array, Array, Array, Array, int]:
    """Load public files, then move the numerical arrays to JAX."""

    train_data_path = data_dir / "train.npz"
    test_data_path = data_dir / "test.npz"
    if not train_data_path.exists() or not test_data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {train_data_path} / {test_data_path}. "
            "Run: uv run python examples/download_dataset_mnist.py"
        )

    with np.load(train_data_path) as train:
        train_images = train["x"]
        x_train = jnp.asarray(train_images).reshape(train_images.shape[0], -1)
        y_train = jnp.asarray(train["y"], dtype=jnp.int32)
        n = int(train["n"])
    with np.load(test_data_path) as test:
        test_images = test["x"]
        x_test = jnp.asarray(test_images).reshape(test_images.shape[0], -1)
        y_test = jnp.asarray(test["y"], dtype=jnp.int32)

    return (
        x_train.astype(jnp.float32) / 255.0,
        y_train,
        x_test.astype(jnp.float32) / 255.0,
        y_test,
        n,
    )


def normalize_records(x: Array) -> Array:
    """Normalize every record to unit L2 norm (future trusted primitive)."""

    norms = jnp.linalg.norm(x, axis=1, keepdims=True)
    return cast(Array, x / (norms + jnp.asarray(1e-12, dtype=x.dtype)))


def gaussian_mechanism(
    state: NoiseState,
    value: PyTree,
    *,
    scale: Array | float,
) -> tuple[PyTree, NoiseState]:
    """Add noise to one joint release; its state must be server-owned in PrivJail."""

    leaves, treedef = jax.tree.flatten(value)
    split_keys = jax.random.split(state.key, len(leaves) + 1)
    next_state = NoiseState(split_keys[0])
    noisy_leaves = [
        leaf
        + jax.random.normal(key, leaf.shape, dtype=leaf.dtype)
        * jnp.asarray(scale, dtype=leaf.dtype)
        for leaf, key in zip(leaves, split_keys[1:], strict=True)
    ]
    return jax.tree.unflatten(treedef, noisy_leaves), next_state


def fit_dp_pca(
    x: Array,
    noise_state: NoiseState,
    *,
    k: int,
    noise_scale: Array | float,
) -> tuple[Array, NoiseState]:
    """Compute the same noisy-covariance PCA used by ``examples/dpsgd.py``."""

    normalized = normalize_records(x)
    covariance = normalized.T @ normalized
    noisy_covariance, noise_state = gaussian_mechanism(
        noise_state, covariance, scale=noise_scale
    )
    symmetric_covariance = 0.5 * (noisy_covariance + noisy_covariance.T)
    _, eigenvectors = jnp.linalg.eigh(symmetric_covariance)
    components = eigenvectors[:, -k:][:, ::-1]
    return components, noise_state


def pca_transform(x: Array, components: Array) -> Array:
    return x @ components


def init_params(key: Array, in_dim: int, hidden_dim: int, out_dim: int) -> Params:
    key1, key2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(key1, (in_dim, hidden_dim))
        / jnp.sqrt(jnp.asarray(in_dim, dtype=jnp.float32)),
        "w2": jax.random.normal(key2, (hidden_dim, out_dim))
        / jnp.sqrt(jnp.asarray(hidden_dim, dtype=jnp.float32)),
    }


def logits(params: Params, x: Array) -> Array:
    hidden = jax.nn.relu(x @ params["w1"])
    return hidden @ params["w2"]


def loss_one(params: Params, x_one: Array, y_one: Array) -> Array:
    """Scalar loss for exactly one record: the input required by ``grad``."""

    return -jax.nn.log_softmax(logits(params, x_one))[y_one]


grad_one: Callable[[Params, Array, Array], Gradients] = jax.grad(loss_one)
per_example_gradients: Callable[[Params, Array, Array], Gradients] = jax.vmap(
    grad_one,
    in_axes=(None, 0, 0),
)


def clip_norm(per_example_grads: PyTree, *, bound: Array | float) -> PyTree:
    """Globally clip each example across all leaves of a gradient PyTree."""

    leaves, treedef = jax.tree.flatten(per_example_grads)
    if not leaves:
        return per_example_grads

    batch_size = leaves[0].shape[0]
    squared_norms = jnp.zeros((batch_size,), dtype=leaves[0].dtype)
    for leaf in leaves:
        parameter_axes = tuple(range(1, leaf.ndim))
        squared_norms = squared_norms + jnp.sum(jnp.square(leaf), axis=parameter_axes)

    norms = jnp.sqrt(squared_norms)
    bound_array = jnp.asarray(bound, dtype=norms.dtype)
    scales = bound_array / jnp.maximum(norms, bound_array)
    clipped_leaves = [
        leaf * scales.reshape((batch_size,) + (1,) * (leaf.ndim - 1)) for leaf in leaves
    ]
    return jax.tree.unflatten(treedef, clipped_leaves)


def clipped_grad_sum(
    params: Params,
    x_batch: Array,
    y_batch: Array,
    *,
    clip_bound: Array | float,
) -> Gradients:
    per_example = per_example_gradients(params, x_batch, y_batch)
    clipped = clip_norm(per_example, bound=clip_bound)
    return cast(
        Gradients,
        jax.tree.map(lambda leaf: jnp.sum(leaf, axis=0), clipped),
    )


def dp_sgd_step(
    params: Params,
    noise_state: NoiseState,
    x_batch: Array,
    y_batch: Array,
    *,
    clip_bound: Array | float,
    noise_multiplier: Array | float,
    expected_batch_size: Array | float,
    learning_rate: Array | float,
) -> tuple[Params, NoiseState]:
    """One pure DP-SGD step, including clipping and Gaussian noise."""

    gradient_sums = clipped_grad_sum(params, x_batch, y_batch, clip_bound=clip_bound)
    noisy_sums, noise_state = gaussian_mechanism(
        noise_state,
        gradient_sums,
        scale=jnp.asarray(noise_multiplier) * jnp.asarray(clip_bound),
    )
    gradients = jax.tree.map(
        lambda gradient: gradient / expected_batch_size, noisy_sums
    )
    params = jax.tree.map(
        lambda parameter, gradient: parameter - learning_rate * gradient,
        params,
        gradients,
    )
    return params, noise_state


jitted_dp_sgd_step = jax.jit(dp_sgd_step)
jitted_dp_pca = jax.jit(fit_dp_pca, static_argnames=("k",))


def poisson_sample(
    state: SamplingState,
    x: Array,
    y: Array,
    *,
    sampling_rate: float,
) -> tuple[Array, Array, SamplingState]:
    """Poisson-sample records outside jit because the result shape is dynamic."""

    next_key, sample_key = jax.random.split(state.key)
    selected = jax.random.bernoulli(sample_key, sampling_rate, (x.shape[0],))
    return x[selected], y[selected], SamplingState(next_key)


def vary_learning_rate(
    start: float,
    end: float,
    saturate_epochs: int,
    epoch: int,
) -> float:
    if saturate_epochs <= 1:
        return end
    step = (start - end) / float(saturate_epochs - 1)
    if epoch < saturate_epochs:
        return start - step * epoch
    return end


@jax.jit
def _batch_metrics(params: Params, x: Array, y: Array) -> tuple[Array, Array]:
    predictions = logits(params, x)
    losses = -jax.nn.log_softmax(predictions)[jnp.arange(y.shape[0]), y]
    correct = jnp.argmax(predictions, axis=1) == y
    return jnp.sum(losses), jnp.sum(correct)


def evaluate(
    params: Params,
    x: Array,
    y: Array,
    *,
    batch_size: int = 2048,
) -> tuple[float, float]:
    total_loss = 0.0
    total_correct = 0
    for start in range(0, x.shape[0], batch_size):
        batch_loss, batch_correct = _batch_metrics(
            params,
            x[start : start + batch_size],
            y[start : start + batch_size],
        )
        total_loss += float(batch_loss)
        total_correct += int(batch_correct)
    return total_loss / x.shape[0], total_correct / x.shape[0]


def _nested_jaxprs(value: Any) -> Iterable[Any]:
    if hasattr(value, "jaxpr") and hasattr(value.jaxpr, "eqns"):
        yield value.jaxpr
        yield from _nested_jaxprs(value.jaxpr)
    elif hasattr(value, "eqns"):
        for equation in value.eqns:
            yield from _nested_jaxprs(equation.params)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _nested_jaxprs(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _nested_jaxprs(item)


def _all_primitive_names(closed_jaxpr: Any) -> list[str]:
    names = [equation.primitive.name for equation in closed_jaxpr.jaxpr.eqns]
    for nested in _nested_jaxprs(closed_jaxpr.jaxpr):
        names.extend(equation.primitive.name for equation in nested.eqns)
    return names


def print_jaxpr_summary(
    name: str,
    function: Callable[..., Any],
    *args: Any,
    full: bool,
    **kwargs: Any,
) -> None:
    closed_jaxpr = jax.make_jaxpr(function)(*args, **kwargs)
    top_level = [equation.primitive.name for equation in closed_jaxpr.jaxpr.eqns]
    all_names = _all_primitive_names(closed_jaxpr)
    histogram = Counter(all_names)

    print(f"\n=== {name} ===")
    print("inputs: ", ", ".join(map(str, closed_jaxpr.in_avals)))
    print("outputs:", ", ".join(map(str, closed_jaxpr.out_avals)))
    print(f"equations: {len(top_level)} top-level / {len(all_names)} recursive")
    print("top-level primitives:", " -> ".join(top_level))
    print(
        "primitive histogram:",
        ", ".join(
            f"{primitive}={count}" for primitive, count in histogram.most_common()
        ),
    )
    if full:
        print(closed_jaxpr)


def inspect_jaxprs(
    *,
    in_dim: int,
    hidden_dim: int,
    batch_size: int,
    full: bool,
) -> None:
    params = init_params(jax.random.key(0), in_dim, hidden_dim, 10)
    x_one = jnp.ones((in_dim,), dtype=jnp.float32)
    y_one = jnp.asarray(1, dtype=jnp.int32)
    x_batch = jnp.ones((batch_size, in_dim), dtype=jnp.float32)
    y_batch = jnp.arange(batch_size, dtype=jnp.int32) % 10
    noise_state = NoiseState(jax.random.key(1))

    print_jaxpr_summary("loss_one", loss_one, params, x_one, y_one, full=full)
    print_jaxpr_summary("grad(loss_one)", grad_one, params, x_one, y_one, full=full)
    print_jaxpr_summary(
        "vmap(grad(loss_one))",
        per_example_gradients,
        params,
        x_batch,
        y_batch,
        full=full,
    )
    print_jaxpr_summary(
        "clipped_grad_sum",
        clipped_grad_sum,
        params,
        x_batch,
        y_batch,
        clip_bound=1.0,
        full=full,
    )
    print_jaxpr_summary(
        "dp_sgd_step",
        dp_sgd_step,
        params,
        noise_state,
        x_batch,
        y_batch,
        clip_bound=1.0,
        noise_multiplier=1.0,
        expected_batch_size=float(batch_size),
        learning_rate=0.1,
        full=full,
    )


def _parse_orders(value: str) -> tuple[int, ...]:
    try:
        orders = tuple(int(part) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "RDP orders must be comma-separated integers"
        ) from error
    if not orders or any(order < 2 for order in orders):
        raise argparse.ArgumentTypeError("RDP orders must be integers >= 2")
    return orders


def _validate_args(args: argparse.Namespace) -> None:
    if args.hidden_dim < 1:
        raise ValueError("--hidden-dim must be >= 1")
    if args.pca_dim < 0:
        raise ValueError("--pca-dim must be >= 0")
    if args.num_steps < 0:
        raise ValueError("--num-steps must be >= 0")
    if args.lot_size < 1:
        raise ValueError("--lot-size must be >= 1")
    if args.clip_norm <= 0:
        raise ValueError("--clip-norm must be > 0")
    if args.pca_sigma <= 0:
        raise ValueError("--pca-sigma must be > 0")
    if args.sgd_sigma <= 0:
        raise ValueError("--sgd-sigma must be > 0")
    if args.eval_batch_size < 1:
        raise ValueError("--eval-batch-size must be >= 1")


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Raw-JAX counterpart of examples/dpsgd.py"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--hidden-dim", type=int, default=1000)
    parser.add_argument("--pca-dim", type=int, default=60)

    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--lot-size", type=int, default=600)
    parser.add_argument("--lr-begin", type=float, default=0.1)
    parser.add_argument("--lr-end", type=float, default=0.052)
    parser.add_argument("--lr-decay-epochs", type=int, default=10)

    parser.add_argument("--clip-norm", type=float, default=4.0)
    parser.add_argument("--pca-sigma", type=float, default=16.0)
    parser.add_argument("--sgd-sigma", type=float, default=8.0)
    parser.add_argument("--eps", type=float, default=2.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument(
        "--rdp-orders",
        type=_parse_orders,
        default=_parse_orders("2,3,4,5,6,7,8,9,10,12,14,16,20,24,32,48,64,96,128"),
    )
    parser.add_argument("--eval-batch-size", type=int, default=2048)

    parser.add_argument(
        "--print-jaxpr",
        action="store_true",
        help="print graph summaries for loss, grad, vmap, clipping, and one step",
    )
    parser.add_argument(
        "--full-jaxpr",
        action="store_true",
        help="also print the complete ClosedJaxpr for each graph",
    )
    parser.add_argument(
        "--jaxpr-only",
        action="store_true",
        help="inspect small graphs without loading MNIST or training",
    )
    parser.add_argument("--jaxpr-in-dim", type=int, default=8)
    parser.add_argument("--jaxpr-hidden-dim", type=int, default=4)
    parser.add_argument("--jaxpr-batch-size", type=int, default=3)
    return parser


def main() -> None:
    args = _argument_parser().parse_args()
    _validate_args(args)

    if args.print_jaxpr or args.full_jaxpr or args.jaxpr_only:
        inspect_jaxprs(
            in_dim=args.jaxpr_in_dim,
            hidden_dim=args.jaxpr_hidden_dim,
            batch_size=args.jaxpr_batch_size,
            full=args.full_jaxpr,
        )
    if args.jaxpr_only:
        return

    params_key, sampling_key, noise_key = jax.random.split(jax.random.key(args.seed), 3)
    sampling_state = SamplingState(sampling_key)
    noise_state = NoiseState(noise_key)

    x_train, y_train, x_test, y_test, n = load_mnist_data(args.data_dir)
    if n != x_train.shape[0]:
        raise ValueError(f"MNIST n={n} does not match x.shape[0]={x_train.shape[0]}")
    if args.pca_dim > x_train.shape[1]:
        raise ValueError("--pca-dim cannot exceed the flattened input dimension")

    accountant = RdpAccountant(
        orders=args.rdp_orders,
        epsilon_limit=args.eps,
        delta=args.delta,
    )

    try:
        if args.pca_dim > 0:
            accountant.compose(
                GaussianEvent(sensitivity=1.0, noise_scale=args.pca_sigma)
            )
            components, noise_state = jitted_dp_pca(
                x_train,
                noise_state,
                k=args.pca_dim,
                noise_scale=args.pca_sigma,
            )
            x_train = pca_transform(x_train, components)
            x_test = pca_transform(x_test, components)

        params = init_params(
            params_key,
            in_dim=x_train.shape[1],
            hidden_dim=args.hidden_dim,
            out_dim=10,
        )

        sampling_rate = min(1.0, args.lot_size / float(n))
        expected_batch_size = sampling_rate * n
        lots_per_epoch = max(1.0, n / float(args.lot_size))
        sgd_event = GaussianEvent(
            sensitivity=args.clip_norm,
            noise_scale=args.sgd_sigma * args.clip_norm,
            sampling_rate=sampling_rate,
        )

        test_loss, test_acc = evaluate(
            params, x_test, y_test, batch_size=args.eval_batch_size
        )
        print(
            f"[init] loss={test_loss:.4f} acc={test_acc:.4f} "
            f"eps={accountant.epsilon:.4f}"
        )

        for step in range(1, args.num_steps + 1):
            epoch = int((step - 1) // lots_per_epoch)
            learning_rate = vary_learning_rate(
                args.lr_begin,
                args.lr_end,
                args.lr_decay_epochs,
                epoch,
            )
            x_lot, y_lot, sampling_state = poisson_sample(
                sampling_state,
                x_train,
                y_train,
                sampling_rate=sampling_rate,
            )

            # Charge before executing the mechanism and exposing its result.
            accountant.compose(sgd_event)
            params, noise_state = jitted_dp_sgd_step(
                params,
                noise_state,
                x_lot,
                y_lot,
                clip_bound=args.clip_norm,
                noise_multiplier=args.sgd_sigma,
                expected_batch_size=expected_batch_size,
                learning_rate=learning_rate,
            )

            next_epoch = int(step // lots_per_epoch)
            is_epoch_end = next_epoch != epoch or step == args.num_steps
            if is_epoch_end:
                test_loss, test_acc = evaluate(
                    params,
                    x_test,
                    y_test,
                    batch_size=args.eval_batch_size,
                )
                print(
                    f"epoch={epoch:03d} step={step:05d} "
                    f"batch={x_lot.shape[0]:04d} "
                    f"loss={test_loss:.4f} acc={test_acc:.4f} "
                    f"eps={accountant.epsilon:.4f}"
                )

    except PrivacyBudgetExceeded as error:
        print(f"Budget exceeded: {error}")

    finally:
        if "params" in locals():
            test_loss, test_acc = evaluate(
                params,
                x_test,
                y_test,
                batch_size=args.eval_batch_size,
            )
            print(
                f"[final] loss={test_loss:.4f} acc={test_acc:.4f} "
                f"eps={accountant.epsilon:.4f}"
            )


if __name__ == "__main__":
    main()
