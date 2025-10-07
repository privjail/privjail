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

from __future__ import annotations
import uuid
import math
import pytest
import pandas as pd
import privjail as pj
from privjail import pandas as ppd

def load_dataframe() -> tuple[ppd.PrivDataFrame, pd.DataFrame]:
    data = [
        ["U001", 1, 0.3, "x"],
        ["U001", 1, 0.2, "y"],
        ["U001", 8, 0.4, "y"],
        ["U001", 1, 0.3, "z"],
        ["U001", 1, 0.3, "z"],
        ["U001", 9, 0.3, "x"],
        ["U001", 1, 0.1, "z"],
        ["U001", 1, 0.0, "z"],
        ["U001", 4, 0.3, "y"],
        ["U001", 1, 0.3, "z"],

        ["U002", 1, 0.2, "x"],
        ["U002", 0, 0.3, "x"],
        ["U002", 2, 0.3, "x"],
        ["U002", 1, 0.4, "y"],
        ["U002", 1, 0.3, "y"],
        ["U002", 3, 0.3, "x"],
        ["U002", 1, 0.3, "x"],
        ["U002", 5, 0.8, "z"],
        ["U002", 5, 0.3, "z"],
        ["U002", 0, 1.0, "x"],

        ["U003", 1, 0.3, "y"],
        ["U003", 1, 0.7, "z"],
        ["U003", 3, 0.9, "x"],
        ["U003", 1, 0.2, "x"],
        ["U003", 1, 0.3, "x"],
        ["U003", 5, 0.9, "z"],
        ["U003", 1, 0.3, "z"],
        ["U003", 1, 0.0, "y"],
        ["U003", 1, 0.1, "x"],
        ["U003", 2, 0.6, "x"],
    ]

    columns = ["uid", "a", "b", "c"]

    domains = {
        "uid" : ppd.StrDomain(),
        "a"   : ppd.RealDomain(dtype="int64", range=(None, None)),
        "b"   : ppd.RealDomain(dtype="float64", range=(0.0, 1.0)),
        "c"   : ppd.CategoryDomain(categories=["x", "y", "z", "w"]), # "w" does not appear in data
    }

    accountant = pj.PureAccountant()
    accountant.set_as_root(name=str(uuid.uuid4()))
    pdf = ppd.PrivDataFrame(data,
                            columns=columns,
                            domains=domains,
                            distance=pj.RealExpr(1),
                            accountant=accountant,
                            user_key="uid")

    df = pd.DataFrame(data, columns=columns)

    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()
    return pdf, df

def test_uldp() -> None:
    pdf, df = load_dataframe()

    assert pdf.user_key == "uid"
    assert pdf.user_max_freq is None

    assert pdf.shape[0].max_distance == math.inf

    with pytest.raises(pj.DPError):
        pdf.shape[0].reveal(eps=0.1)

    # bound user contribution
    k = 5
    pdf = pdf.groupby("uid").head(k) # type: ignore
    df = df.groupby("uid").head(k)

    assert (pdf._value == df).all().all()

    assert pdf.user_max_freq == k
    assert pdf.shape[0].max_distance == k

    # prohibited operations
    with pytest.raises(Exception):
        pdf.groupby("uid").get_group("U001")

    with pytest.raises(Exception):
        pdf.groupby("uid").groups # type: ignore

    with pytest.raises(Exception):
        len(pdf.groupby("uid"))

    with pytest.raises(Exception):
        for _ in pdf.groupby("uid"):
            pass

    # group by non-user-key columns
    s = pj.SensitiveInt(0)
    for c, pdf_c in pdf.groupby("c"):
        assert pdf_c.user_key == "uid"
        assert pdf_c.user_max_freq == k
        assert pdf_c.shape[0].max_distance == k

        s += pdf_c.shape[0]

    assert s.max_distance == k

    s = pj.SensitiveInt(0)
    # FIXME: remove ["b"] from the PrivJail version (by returning SensitiveSeries rather than SensitiveDataFrame)
    for pv, v in zip(pdf.groupby("c")["b"].sum()["b"], df.groupby("c")["b"].sum()): # type: ignore
        assert pv.max_distance == k
        assert pv._value == v

        s += pv

    assert s.max_distance == k

    assert pdf.groupby("c")["b"].sum()["b"]["w"]._value == 0 # type: ignore
