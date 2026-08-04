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

from __future__ import annotations

import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

import privjail as pj
import privjail.jax as pjx
import privjail.numpy as pnp


def _unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_server(process: subprocess.Popen[str], port: int) -> None:
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output, _ = process.communicate()
            raise RuntimeError(f"PrivJail server exited during startup:\n{output}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError("Timed out waiting for the PrivJail server.")


def test_jitted_dp_sgd_step_over_egrpc(tmp_path: Path) -> None:
    data_path = tmp_path / "private.npz"
    schema_path = tmp_path / "private.json"
    np.savez(
        data_path,
        x=np.arange(12, dtype=np.float32).reshape(4, 3) / 12.0,
        y=np.asarray([0.0, 0.5, 1.0, 1.5], dtype=np.float32),
    )
    schema_path.write_text(
        json.dumps(
            {
                "x": {"alignment_signature": "records"},
                "y": {"alignment_signature": "records"},
            }
        )
    )

    port = _unused_local_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "privjail",
            "serve",
            "127.0.0.1",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    connected = False
    try:
        _wait_for_server(process, port)
        pj.connect("127.0.0.1", port)
        connected = True

        loaded = pnp.load(str(data_path), str(schema_path))
        private_x = pjx.asarray(loaded["x"], dtype=jnp.float32)
        private_y = pjx.asarray(loaded["y"], dtype=jnp.float32)
        assert isinstance(private_x, pjx.PrivArray)
        assert isinstance(private_y, pjx.PrivArray)
        assert private_x.alignment_signature == private_y.alignment_signature

        params = {
            "w": jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float32),
            "b": jnp.asarray(0.0, dtype=jnp.float32),
        }

        def loss_fn(
            current_params: dict[str, jax.Array],
            x_batch: jax.Array,
            y_batch: jax.Array,
        ) -> jax.Array:
            predictions = x_batch @ current_params["w"] + current_params["b"]
            errors = predictions - y_batch
            return jnp.mean(errors * errors)

        def dp_sgd_step(
            current_params: dict[str, jax.Array],
            x_batch: jax.Array,
            y_batch: jax.Array,
            clip_bound: float,
            noise_multiplier: float,
            expected_batch_size: float,
            learning_rate: float,
            delta: float,
        ) -> dict[str, jax.Array]:
            gradient_sums = pjx.clipped_grad(
                loss_fn,
                l2_clip_norm=clip_bound,
                batch_argnums=(1, 2),
            )(
                current_params,
                x_batch,
                y_batch,
            )
            noisy_sums = pj.gaussian_mechanism(
                gradient_sums,
                scale=noise_multiplier * clip_bound,
                delta=delta,
            )
            gradients = jax.tree.map(
                lambda leaf: leaf / expected_batch_size,
                noisy_sums,
            )
            return cast(
                dict[str, jax.Array],
                jax.tree.map(
                    lambda parameter, gradient: (
                        parameter - learning_rate * gradient
                    ),
                    current_params,
                    gradients,
                ),
            )

        step = pjx.jit(
            dp_sgd_step,
            static_argnames=("clip_bound", "noise_multiplier", "delta"),
        )
        updated = step(
            params,
            private_x,
            private_y,
            clip_bound=1.0,
            noise_multiplier=2.0,
            expected_batch_size=4.0,
            learning_rate=0.1,
            delta=1e-5,
        )

        assert all(isinstance(leaf, jax.Array) for leaf in updated.values())
        assert updated["w"].shape == params["w"].shape
        assert updated["b"].shape == params["b"].shape
        assert all(
            bool(jnp.all(jnp.isfinite(leaf))) for leaf in updated.values()
        )
        family, spent = pj.budgets_spent()[str(data_path)]
        assert family == "approx"
        assert isinstance(spent, tuple)
        assert spent[0] > 0
        assert spent[1] == 1e-5

        del step, updated, private_y, private_x, loaded
    finally:
        if connected:
            pj.disconnect()
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
