# Copyright 2025 TOYOTA MOTOR CORPORATION.
# Copyright 2026 Shumpei Shiina.
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

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import privjail as pj
import privjail.jax as pjx
import privjail.numpy as pnp

def load_mnist_data():
    train_data_path = Path("data/mnist/train.npz")
    train_schema_path = Path("examples/schema/mnist.json")
    test_data_path = Path("data/mnist/test.npz")

    if not train_data_path.exists() or not test_data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {train_data_path} / {test_data_path}. Run: uv run python examples/download_dataset_mnist.py"
        )

    train = pnp.load(str(train_data_path), schema_path=str(train_schema_path))
    test = np.load(test_data_path)
    n = int(train["n"])
    return pjx.asarray(train["x"], dtype=jnp.float32), pjx.asarray(train["y"], dtype=jnp.int32), jnp.asarray(test["x"]), jnp.asarray(test["y"]), n

def fit_dp_pca(x, k, sigma):
    x_norm = pj.normalize(x)
    cov = x_norm.T @ x_norm
    cov = pj.gaussian_mechanism(cov, scale=sigma)
    cov = 0.5 * (cov + cov.T)
    _, vecs = jnp.linalg.eigh(cov)
    return vecs[:, -k:][:, ::-1]

def pca_transform(x, components):
    return x @ components

def init_params(rng, in_dim, hidden_dim, out_dim):
    rng1, rng2 = jax.random.split(rng)
    w1 = jax.random.normal(rng1, shape=(in_dim, hidden_dim)) / jnp.sqrt(in_dim)
    w2 = jax.random.normal(rng2, shape=(hidden_dim, out_dim)) / jnp.sqrt(hidden_dim)
    return {"w1": w1, "w2": w2}

def forward(params, x):
    w1, w2 = params["w1"], params["w2"]
    h1 = jax.nn.relu(x @ w1)
    return h1 @ w2

def loss_fn(params, x, y):
    log_probs = jax.nn.log_softmax(forward(params, x))
    losses = -jax.vmap(lambda log_prob, label: log_prob[label])(log_probs, y)
    return losses.mean()

@pjx.jit(static_argnames=("clip_norm", "sigma"))
def dp_sgd_step(params, x, y, clip_norm, sigma, lot_size, lr):
    grad_sums = pjx.clipped_grad(loss_fn, l2_clip_norm=clip_norm, batch_argnums=(1, 2))(params, x, y)
    grad_sums = pj.gaussian_mechanism(grad_sums, scale=sigma * clip_norm)
    return update_params(params, grad_sums, lot_size, lr)

def update_params(params, grad_sums, lot_size, lr):
    return {k: v - lr * grad_sums[k] / lot_size for k, v in params.items()}

def vary_learning_rate(start, end, saturate_epochs, epoch):
    if saturate_epochs <= 0:
        return start
    step = (start - end) / float(saturate_epochs - 1)
    if epoch < saturate_epochs:
        return start - step * epoch
    return end

def evaluate(params, x, y, batch_size=2048):
    losses = []
    accs = []
    for i in range(0, x.shape[0], batch_size):
        xb = x[i : i + batch_size]
        yb = y[i : i + batch_size]
        loss = float(loss_fn(params, xb, yb))
        acc = float(jnp.mean(forward(params, xb).argmax(axis=1) == yb))
        losses.append(loss)
        accs.append(acc)
    return float(np.mean(losses)), float(np.mean(accs))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--hidden-dim", type=int, default=1000)
    p.add_argument("--pca-dim", type=int, default=60)

    p.add_argument("--num-steps", type=int, default=10000)
    p.add_argument("--lot-size", type=int, default=600)
    p.add_argument("--lr-begin", type=float, default=0.1)
    p.add_argument("--lr-end", type=float, default=0.052)
    p.add_argument("--lr-decay-epochs", type=int, default=10)

    p.add_argument("--clip-norm", type=float, default=4.0)
    p.add_argument("--pca-sigma", type=float, default=16.0)
    p.add_argument("--sgd-sigma", type=float, default=8.0)
    p.add_argument("--eps", type=float, default=2.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--rdp-orders", type=str, default="2,3,4,5,6,7,8,9,10,12,14,16,20,24,32,48,64,96,128")

    p.add_argument("--remote", action="store_true")
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--port", type=int, default=12345)

    args = p.parse_args()

    if args.remote:
        pj.connect(args.host, args.port)

    rng = jax.random.key(args.seed)

    x_train, y_train, x_test, y_test, n = load_mnist_data()
    x_train = x_train.reshape(n, -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    alpha = [int(s) for s in args.rdp_orders.split(",") if s.strip()]

    with pj.RDP([x_train, y_train], alpha=alpha, budget_limit=(args.eps, args.delta), prepaid=True):
        if args.pca_dim > 0:
            components = fit_dp_pca(x_train, args.pca_dim, sigma=args.pca_sigma)
            x_train = pca_transform(x_train, components)
            x_test = pca_transform(x_test, components)

        params = init_params(rng, x_test.shape[1], args.hidden_dim, 10)

        if args.lot_size < 1:
            raise ValueError("--lot-size must be >= 1")
        q = min(1.0, args.lot_size / float(n))
        lots_per_epoch = max(1.0, n / float(args.lot_size))

        test_loss, test_acc = evaluate(params, x_test, y_test)
        eps, _ = x_train.accountant.parent.budget_spent
        print(f"[init] loss={test_loss:.4f} acc={test_acc:.4f} eps={eps:.4f}")

        try:
            for step in range(1, args.num_steps + 1):
                epoch = int((step - 1) // lots_per_epoch)
                lr = vary_learning_rate(args.lr_begin, args.lr_end, args.lr_decay_epochs, epoch)

                x_lot, y_lot = pj.sample(x_train, y_train, q=q)
                params = dp_sgd_step(params, x_lot, y_lot, args.clip_norm, args.sgd_sigma, args.lot_size, lr)

                next_epoch = int(step // lots_per_epoch)
                is_epoch_end = (next_epoch != epoch) or (step == args.num_steps)
                if is_epoch_end:
                    test_loss, test_acc = evaluate(params, x_test, y_test)
                    eps, _ = x_train.accountant.parent.budget_spent
                    print(f"epoch={epoch:03d} step={step:05d} loss={test_loss:.4f} acc={test_acc:.4f} eps={eps:.4f}")
                    # for name, state in pj.accountant_state().items():
                    #     print(f"=== {name} ===")
                    #     print(state)

        except pj.BudgetExceededError:
            print("Budget exceeded.")

        finally:
            test_loss, test_acc = evaluate(params, x_test, y_test)
            eps, _ = x_train.accountant.parent.budget_spent
            print(f"[final] loss={test_loss:.4f} acc={test_acc:.4f} eps={eps:.4f}")
            print(pj.budgets_spent())

if __name__ == "__main__":
    main()
