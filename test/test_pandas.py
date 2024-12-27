from __future__ import annotations
from typing import Any
import uuid
import pytest
import pandas as pd
import numpy as np
import privjail as pj
from privjail import pandas as ppd

def load_dataframe() -> tuple[ppd.PrivDataFrame, pd.DataFrame]:
    data = {
        "a": [1, 2, 3, 4, 5],
        "b": [2, 4, 4, 4, 3],
    }
    domains = {
        "a": ppd.RealDomain(type="int64", range=(None, None)),
        "b": ppd.RealDomain(type="int64", range=(None, None)),
    }
    pdf = ppd.PrivDataFrame(data, domains=domains, distance=pj.Distance(1), root_name=str(uuid.uuid4()))
    df = pd.DataFrame(data)
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()
    return pdf, df

def assert_equal_sensitive_series(sensitive_ser: ppd.SensitiveSeries[Any], expected_ser: pd.Series[Any]) -> None:
    assert isinstance(sensitive_ser, ppd.SensitiveSeries)
    assert (sensitive_ser.index == expected_ser.index).all()
    for idx in sensitive_ser.index:
        assert sensitive_ser.loc[idx]._value == expected_ser.loc[idx]

def assert_equal_sensitive_dataframes(sensitive_df: ppd.SensitiveDataFrame, expected_df: pd.DataFrame) -> None:
    assert isinstance(sensitive_df, ppd.SensitiveDataFrame)
    assert (sensitive_df.index == expected_df.index).all()
    assert (sensitive_df.columns == expected_df.columns).all()
    for idx in sensitive_df.index:
        for col in sensitive_df.columns:
            assert sensitive_df.loc[idx, col]._value == expected_df.loc[idx, col]

def test_priv_dataframe_size() -> None:
    pdf, df = load_dataframe()

    # The `shape` member should be a pair of a sensitive value (row) and a non-sensitive value (column)
    assert isinstance(pdf.shape, tuple)
    assert len(pdf.shape) == len(df.shape) == 2
    assert isinstance(pdf.shape[0], pj.Prisoner)
    assert pdf.shape[0]._value == df.shape[0]
    assert pdf.shape[0].distance.max() == 1
    assert pdf.shape[1] == len(pdf.columns) == len(df.columns)

    # The `size` member should be a sensitive value
    assert isinstance(pdf.size, pj.Prisoner)
    assert pdf.size._value == df.size
    assert pdf.size.distance.max() == len(pdf.columns) == len(df.columns)

    # Builtin `len()` function should raise an error because it must be an integer value
    with pytest.raises(pj.DPError):
        len(pdf)

def test_priv_dataframe_comp() -> None:
    pdf, df = load_dataframe()

    # A non-sensitive value should be successfully compared against a private dataframe
    assert isinstance(pdf == 3, ppd.PrivDataFrame)
    assert isinstance(pdf != 3, ppd.PrivDataFrame)
    assert isinstance(pdf <  3, ppd.PrivDataFrame)
    assert isinstance(pdf <= 3, ppd.PrivDataFrame)
    assert isinstance(pdf >  3, ppd.PrivDataFrame)
    assert isinstance(pdf >= 3, ppd.PrivDataFrame)
    assert ((pdf == 3)._value == (df == 3)).all().all()
    assert ((pdf != 3)._value == (df != 3)).all().all()
    assert ((pdf <  3)._value == (df <  3)).all().all()
    assert ((pdf <= 3)._value == (df <= 3)).all().all()
    assert ((pdf >  3)._value == (df >  3)).all().all()
    assert ((pdf >= 3)._value == (df >= 3)).all().all()

    # A non-sensitive value should be successfully compared against a private series
    assert isinstance(pdf["a"] == 3, ppd.PrivSeries)
    assert isinstance(pdf["a"] != 3, ppd.PrivSeries)
    assert isinstance(pdf["a"] <  3, ppd.PrivSeries)
    assert isinstance(pdf["a"] <= 3, ppd.PrivSeries)
    assert isinstance(pdf["a"] >  3, ppd.PrivSeries)
    assert isinstance(pdf["a"] >= 3, ppd.PrivSeries)
    assert ((pdf["a"] == 3)._value == (df["a"] == 3)).all()
    assert ((pdf["a"] != 3)._value == (df["a"] != 3)).all()
    assert ((pdf["a"] <  3)._value == (df["a"] <  3)).all()
    assert ((pdf["a"] <= 3)._value == (df["a"] <= 3)).all()
    assert ((pdf["a"] >  3)._value == (df["a"] >  3)).all()
    assert ((pdf["a"] >= 3)._value == (df["a"] >= 3)).all()

    # An irrelevant, non-private dataframe should not be compared against a private dataframe
    with pytest.raises(pj.DPError): pdf == df
    with pytest.raises(pj.DPError): pdf != df
    with pytest.raises(pj.DPError): pdf <  df
    with pytest.raises(pj.DPError): pdf <= df
    with pytest.raises(pj.DPError): pdf >  df
    with pytest.raises(pj.DPError): pdf >= df

    # An irrelevant, non-private 2d array should not be compared against a private dataframe
    with pytest.raises(pj.DPError): pdf == [[list(range(len(pdf.columns)))] for x in range(5)]
    with pytest.raises(pj.DPError): pdf != [[list(range(len(pdf.columns)))] for x in range(5)]
    with pytest.raises(pj.DPError): pdf <  [[list(range(len(pdf.columns)))] for x in range(5)]
    with pytest.raises(pj.DPError): pdf <= [[list(range(len(pdf.columns)))] for x in range(5)]
    with pytest.raises(pj.DPError): pdf >  [[list(range(len(pdf.columns)))] for x in range(5)]
    with pytest.raises(pj.DPError): pdf >= [[list(range(len(pdf.columns)))] for x in range(5)]

    # An irrelevant, non-private array should not be compared against a private series
    with pytest.raises(pj.DPError): pdf["a"] == [0, 1, 2, 3, 4]
    with pytest.raises(pj.DPError): pdf["a"] != [0, 1, 2, 3, 4]
    with pytest.raises(pj.DPError): pdf["a"] <  [0, 1, 2, 3, 4]
    with pytest.raises(pj.DPError): pdf["a"] <= [0, 1, 2, 3, 4]
    with pytest.raises(pj.DPError): pdf["a"] >  [0, 1, 2, 3, 4]
    with pytest.raises(pj.DPError): pdf["a"] >= [0, 1, 2, 3, 4]

    x = pj.Prisoner(value=0, distance=pj.Distance(1), root_name=str(uuid.uuid4()))

    # A sensitive value should not be compared against a private dataframe
    with pytest.raises(pj.DPError): pdf == x
    with pytest.raises(pj.DPError): pdf != x
    with pytest.raises(pj.DPError): pdf <  x
    with pytest.raises(pj.DPError): pdf <= x
    with pytest.raises(pj.DPError): pdf >  x
    with pytest.raises(pj.DPError): pdf >= x

    # A sensitive value should not be compared against a private series
    with pytest.raises(pj.DPError): pdf["a"] == x
    with pytest.raises(pj.DPError): pdf["a"] != x
    with pytest.raises(pj.DPError): pdf["a"] <  x
    with pytest.raises(pj.DPError): pdf["a"] <= x
    with pytest.raises(pj.DPError): pdf["a"] >  x
    with pytest.raises(pj.DPError): pdf["a"] >= x

    # Sensitive dataframes of potentially different size should not be compared
    pdf_ = pdf[pdf["a"] >= 0]
    with pytest.raises(pj.DPError): pdf == pdf_
    with pytest.raises(pj.DPError): pdf != pdf_
    with pytest.raises(pj.DPError): pdf <  pdf_
    with pytest.raises(pj.DPError): pdf <= pdf_
    with pytest.raises(pj.DPError): pdf >  pdf_
    with pytest.raises(pj.DPError): pdf >= pdf_

    # Sensitive series of potentially different size should not be compared
    with pytest.raises(pj.DPError): pdf["a"] == pdf_["a"]
    with pytest.raises(pj.DPError): pdf["a"] != pdf_["a"]
    with pytest.raises(pj.DPError): pdf["a"] <  pdf_["a"]
    with pytest.raises(pj.DPError): pdf["a"] <= pdf_["a"]
    with pytest.raises(pj.DPError): pdf["a"] >  pdf_["a"]
    with pytest.raises(pj.DPError): pdf["a"] >= pdf_["a"]

def test_priv_dataframe_getitem() -> None:
    pdf, df = load_dataframe()

    # A single-column view should be successfully retrieved from a private dataframe
    assert isinstance(pdf["a"], ppd.PrivSeries)
    assert (pdf["a"]._value == df["a"]).all()

    # A multi-column view should be successfully retrieved from a private dataframe
    assert isinstance(pdf[["a", "b"]], ppd.PrivDataFrame)
    assert (pdf[["a", "b"]]._value == df[["a", "b"]]).all().all()

    # An irrelevant, non-sensitve bool vector should not be accepted for filtering (dataframe)
    with pytest.raises(pj.DPError):
        pdf[[True, True, False, False, True]] # type: ignore

    # An irrelevant, non-sensitve bool vector should not be accepted for filtering (series)
    with pytest.raises(pj.DPError):
        pdf["a"][[True, True, False, False, True]] # type: ignore

    # A bool-filtered view should be successfully retrieved from a private dataframe
    assert isinstance(pdf[pdf["a"] > 3], ppd.PrivDataFrame)
    assert (pdf[pdf["a"] > 3]._value == df[df["a"] > 3]).all().all()

    # A bool-filtered view should be successfully retrieved from a private series
    assert isinstance(pdf["a"][pdf["a"] > 3], ppd.PrivSeries)
    assert (pdf["a"][pdf["a"] > 3]._value == df["a"][df["a"] > 3]).all()

    # A sensitve bool vector of potentially different size should not be accepted for filtering (dataframe)
    pdf_ = pdf[pdf["a"] >= 0]
    with pytest.raises(pj.DPError):
        pdf[pdf_["a"] > 3]

    # A sensitve bool vector of potentially different size should not be accepted for filtering (series)
    with pytest.raises(pj.DPError):
        pdf["a"][pdf_["a"] > 3]

    x = pj.Prisoner(value=0, distance=pj.Distance(1), root_name=str(uuid.uuid4()))

    # A sensitive value should not be used as a column name
    with pytest.raises(pj.DPError):
        pdf[x] # type: ignore

    # A slice should not be used for selecting rows of a dataframe
    # TODO: it might be legal to accept a slice
    with pytest.raises(pj.DPError):
        pdf[2:5] # type: ignore

    # A slice should not be used for selecting rows of a series
    # TODO: it might be legal to accept a slice
    with pytest.raises(pj.DPError):
        pdf["a"][2:5] # type: ignore

def test_priv_dataframe_setitem() -> None:
    pdf, df = load_dataframe()

    # A non-sensitive value should be successfully assigned to a single-column view
    pdf["c"] = 10
    df["c"] = 10
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    # A private series should be successfully assigned to a single-column view
    pdf["d"] = pdf["a"]
    df["d"] = df["a"]
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    # A non-sensitive value should be successfully assigned to a multi-column view
    pdf[["e", "f"]] = 100
    df[["e", "f"]] = 100
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    # A private dataframe should be successfully assigned to a multi-column view
    pdf[["g", "h"]] = pdf[["a", "b"]]
    df[["g", "h"]] = df[["a", "b"]]
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    # A non-sensitive value should be successfully assigned to a bool-filtered view
    pdf[pdf["a"] < 3] = 8
    df[df["a"] < 3] = 8
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    # A non-sensitive array should be successfully assigned to a bool-filtered view
    pdf[pdf["a"] == 8] = list(range(len(pdf.columns)))
    df[df["a"] == 8] = list(range(len(df.columns)))
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    # A private dataframe should be successfully assigned to a bool-filtered view
    pdf[pdf["a"] < 3] = pdf[pdf["a"] < 4]
    df[df["a"] < 3] = df[df["a"] < 4]
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    # A non-private series can be assigned to a private single-column view
    # (because this operation succeeds regardless of the row number)
    # TODO: this may be disallowed if we track the value domain for each column
    pdf["i"] = df["c"]
    df["i"] = df["c"]
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    # A non-private dataframe can be assigned to a private multi-column view
    # (because this operation succeeds regardless of the row number)
    # TODO: this may be disallowed if we track the value domain for each column
    pdf[["j", "k"]] = df[["a", "b"]]
    df[["j", "k"]] = df[["a", "b"]]
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    # An irrelevant, non-private dataframe can be assigned to a bool-filtered view
    # (because this operation succeeds regardless of the row number)
    # TODO: this may be disallowed if we track the value domain for each column
    pdf[pdf["a"] > 3] = df
    df[df["a"] > 3] = df
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    x = pj.Prisoner(value=0, distance=pj.Distance(1), root_name=str(uuid.uuid4()))

    # A sensitive value should not be assigned to a single-column view
    with pytest.raises(pj.DPError):
        pdf["x"] = x

    # A sensitive value should not be assigned to a multi-column view
    with pytest.raises(pj.DPError):
        pdf[["x", "y"]] = x

    # A sensitive value should not be assigned to a bool-filtered view
    with pytest.raises(pj.DPError):
        pdf[pdf["a"] > 3] = x

    # An irrelevant, non-private array should not be assigned to a column view
    with pytest.raises(pj.DPError):
        pdf["x"] = [1, 2, 3, 4, 5]

    # An irrelevant, non-private 2d array should not be assigned to a bool-filtered view
    with pytest.raises(pj.DPError):
        pdf[pdf["a"] > 3] = [[list(range(len(pdf.columns)))] for x in range(5)]

    # A sensitive value should not be used as a column name
    with pytest.raises(pj.DPError):
        pdf[x] = 10 # type: ignore

    # A sensitve bool vector of potentially different size should not be accepted for filtering (dataframe)
    pdf_ = pdf[pdf["a"] >= 0]
    with pytest.raises(pj.DPError):
        pdf[pdf_["a"] > 3] = 10

    # A sensitve bool vector of potentially different size should not be accepted for filtering (series)
    with pytest.raises(pj.DPError):
        pdf["a"][pdf_["a"] > 3] = 10

    # A slice should not be used for selecting rows
    # TODO: it might be legal to accept a slice
    with pytest.raises(pj.DPError):
        pdf[2:5] = 0 # type: ignore

def test_priv_dataframe_replace() -> None:
    pdf, df = load_dataframe()

    # Default behaviour
    assert (pdf.replace(4, 10)._value == df.replace(4, 10)).all().all()

    # Special behaviour when `inplace=True`
    assert pdf.replace(3, 10, inplace=True) == None
    assert (pdf._value == df.replace(3, 10)).all().all()

def test_priv_series_replace() -> None:
    pdf, df = load_dataframe()

    # Default behaviour
    assert (pdf["b"].replace(4, 10)._value == df["b"].replace(4, 10)).all()

    # Special behaviour when `inplace=True`
    assert pdf["b"].replace(3, 10, inplace=True) == None
    assert (pdf["b"]._value == df["b"].replace(3, 10)).all()

def test_priv_dataframe_dropna() -> None:
    pdf, df = load_dataframe()

    pdf.replace(3, np.nan, inplace=True)
    df.replace(3, np.nan, inplace=True)

    # Default behaviour
    assert (pdf.dropna()._value == df.dropna()).all().all()

    # Should return an error with ignore_index=True
    with pytest.raises(pj.DPError):
        pdf.dropna(ignore_index=True)

    # Special behaviour when `inplace=True`
    assert pdf.dropna(inplace=True) == None
    assert (pdf._value == df.dropna()).all().all()

def test_priv_series_dropna() -> None:
    pdf, df = load_dataframe()

    pdf.replace(3, np.nan, inplace=True)
    df.replace(3, np.nan, inplace=True)

    # Default behaviour
    assert (pdf["b"].dropna()._value == df["b"].dropna()).all()

    # Should return an error with ignore_index=True
    with pytest.raises(pj.DPError):
        pdf["b"].dropna(ignore_index=True)

    # TODO: this fails because of the original pandas bug?
    # # Special behaviour when `inplace=True`
    # assert pdf["b"].dropna(inplace=True) == None
    # assert (pdf["b"]._value == df["b"].dropna()).all()

def test_priv_series_value_counts() -> None:
    pdf, df = load_dataframe()

    # Should return an error without arguments
    with pytest.raises(pj.DPError):
        pdf["b"].value_counts()

    # Should return an error with `sort=True`
    with pytest.raises(pj.DPError):
        pdf["b"].value_counts(values=[1, 2, 3, 4, 5])

    # Should return an error without specifying values
    with pytest.raises(pj.DPError):
        pdf["b"].value_counts(sort=False)

    # Should return correct counts when all possible values are provided
    values = [2, 3, 4]
    counts = pdf["b"].value_counts(sort=False, values=values)
    assert_equal_sensitive_series(counts, pd.Series({2: 1, 3: 1, 4: 3}))
    assert counts.iloc[0].distance.max() == 1

    # Should return correct counts when only a part of possible values are provided
    values = [3, 4]
    counts = pdf["b"].value_counts(sort=False, values=values)
    assert_equal_sensitive_series(counts, pd.Series({3: 1, 4: 3}))
    assert counts.iloc[0].distance.max() == 1

    # Should return correct counts when non-existent values are provided
    values = [1, 3, 4, 5]
    counts = pdf["b"].value_counts(sort=False, values=values)
    assert_equal_sensitive_series(counts, pd.Series({1: 0, 3: 1, 4: 3, 5: 0}))
    assert counts.iloc[0].distance.max() == 1

    # Should be able to get a sensitive value from a sensitive series
    c4 = counts[4]
    assert isinstance(c4, pj.Prisoner)
    assert c4.distance.max() == 1
    assert c4._value == 3

    # Should be able to get a sensitive view from a sensitive series
    c3 = counts[1:3][3]
    c4 = counts[1:3][4]
    assert isinstance(c3, pj.Prisoner)
    assert isinstance(c4, pj.Prisoner)
    assert c3.distance.max() == c4.distance.max() == 1
    assert c3._value == 1
    assert c4._value == 3

def test_crosstab() -> None:
    pdf, df = load_dataframe()

    rowvalues = [1, 2, 3, 4, 5]
    colvalues = [1, 2, 3, 4, 5]

    # Should raise an error without rowvalues/colvalues
    with pytest.raises(pj.DPError): ppd.crosstab(pdf["a"], pdf["b"])
    with pytest.raises(pj.DPError): ppd.crosstab(pdf["a"], pdf["b"], rowvalues=rowvalues)
    with pytest.raises(pj.DPError): ppd.crosstab(pdf["a"], pdf["b"], colvalues=colvalues)

    # Should raise an error with margins=True
    with pytest.raises(pj.DPError):
        ppd.crosstab(pdf["a"], pdf["b"], rowvalues=rowvalues, colvalues=colvalues, margins=True)

    # Should raise an error with series of potentially different size
    pdf_ = pdf[pdf["a"] >= 0]
    with pytest.raises(pj.DPError):
        ppd.crosstab(pdf["a"], pdf_["b"], rowvalues=rowvalues, colvalues=colvalues)

    # Should return correct counts when all possible values are provided
    counts = ppd.crosstab(pdf["a"], pdf["b"], rowvalues=rowvalues, colvalues=colvalues)
    ans = pd.DataFrame([[0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 0]],
                       index=rowvalues, columns=colvalues)
    assert_equal_sensitive_dataframes(counts, ans)

    pj.laplace_mechanism(counts, eps=1.0)

def test_cut() -> None:
    pdf, df = load_dataframe()

    # Default behaviour
    assert isinstance(ppd.cut(pdf["a"], bins=[0, 3, 6]), ppd.PrivSeries)
    assert (ppd.cut(pdf["a"], bins=[0, 3, 6])._value == pd.cut(df["a"], bins=[0, 3, 6])).all()

    # Should raise an error with a scalar bins
    with pytest.raises(pj.DPError):
        ppd.cut(pdf["a"], bins=2) # type: ignore

def test_dataframe_groupby() -> None:
    pdf, df = load_dataframe()

    # Should raise an error without `keys`
    with pytest.raises(pj.DPError):
        pdf.groupby("b")

    # Should be able to group by a single column
    keys = [1, 2, 4]
    groups = pdf.groupby("b", keys=keys)
    assert len(groups) == len(keys)

    # Should be able to loop over groups
    for i, (key, pdf_) in enumerate(groups):
        assert key == keys[i]
        assert isinstance(pdf_, ppd.PrivDataFrame)

    # A group with key=1 should be empty but its columns and dtypes should match the original
    assert isinstance(groups.get_group(1), ppd.PrivDataFrame)
    assert len(groups.get_group(1)._value) == 0
    assert (groups.get_group(1).columns == pdf.columns).all()
    assert (groups.get_group(1).dtypes == pdf.dtypes).all()

    # Check for a group with key=2
    assert isinstance(groups.get_group(2), ppd.PrivDataFrame)
    assert (groups.get_group(2)._value == pd.DataFrame({"a": [1], "b": [2]}, index=[0])).all().all()
    assert (groups.get_group(2).columns == pdf.columns).all()
    assert (groups.get_group(2).dtypes == pdf.dtypes).all()

    # Check for a group with key=4
    assert isinstance(groups.get_group(4), ppd.PrivDataFrame)
    assert (groups.get_group(4)._value == pd.DataFrame({"a": [2, 3, 4], "b": [4, 4, 4]}, index=[1, 2, 3])).all().all()
    assert (groups.get_group(4).columns == pdf.columns).all()
    assert (groups.get_group(4).dtypes == pdf.dtypes).all()

def test_privacy_budget() -> None:
    pdf, df = load_dataframe()

    eps = 0.1
    pdf1 = pdf[pdf["b"] >= 3]
    counts = pdf1["b"].value_counts(sort=False, values=[3, 4, 5])

    pj.laplace_mechanism(counts, eps=eps)
    assert pj.consumed_privacy_budget()[pdf.root_name()] == eps

    pj.laplace_mechanism(counts[4], eps=eps)
    assert pj.consumed_privacy_budget()[pdf.root_name()] == eps * 2

    pdf2 = pdf[pdf["a"] >= 3]

    pj.laplace_mechanism(pdf2.shape[0], eps=eps)
    assert pj.consumed_privacy_budget()[pdf.root_name()] == eps * 3

    # Privacy budget for different data sources should be managed independently
    pdf_, df_ = load_dataframe()

    pj.laplace_mechanism(pdf_.shape[0], eps=eps)
    assert pj.consumed_privacy_budget()[pdf.root_name()] == eps * 3
    assert pj.consumed_privacy_budget()[pdf_.root_name()] == eps

def test_privacy_budget_parallel_composition() -> None:
    pdf, df = load_dataframe()

    eps = 0.1

    # value_counts()
    counts = pdf["b"].value_counts(sort=False, values=[2, 3, 4])

    pj.laplace_mechanism(counts, eps=eps)
    assert pj.consumed_privacy_budget()[pdf.root_name()] == eps

    pj.laplace_mechanism(counts[2], eps=eps)
    assert pj.consumed_privacy_budget()[pdf.root_name()] == eps * 2

    pj.laplace_mechanism(counts[3], eps=eps)
    assert pj.consumed_privacy_budget()[pdf.root_name()] == eps * 2

    pj.laplace_mechanism(counts[3] + counts[4], eps=eps)
    assert pj.consumed_privacy_budget()[pdf.root_name()] == eps * 3

    pj.laplace_mechanism(counts[2] - counts[4], eps=eps)
    assert pj.consumed_privacy_budget()[pdf.root_name()] == eps * 3

    # crosstab()
    crosstab = ppd.crosstab(pdf["a"], pdf["b"], rowvalues=[1, 2, 3, 4, 5], colvalues=[1, 2, 3, 4, 5])

    for idx in crosstab.index:
        for col in crosstab.columns:
            pj.laplace_mechanism(crosstab.loc[idx, col], eps=eps)

    assert pj.consumed_privacy_budget()[pdf.root_name()] == eps * 4

    s = pj.SensitiveInt(0)
    for idx in crosstab.index:
        for col in crosstab.columns:
            s += crosstab.loc[idx, col]
    assert s.distance.max() == 1
    pj.laplace_mechanism(s, eps=eps)
    assert pj.consumed_privacy_budget()[pdf.root_name()] == eps * 5

    assert crosstab.loc[1, 5].distance.max() == 1 # type: ignore
    pj.laplace_mechanism(crosstab.loc[1, 5], eps=eps) # type: ignore
    assert pj.consumed_privacy_budget()[pdf.root_name()] == eps * 6

    # groupby()
    s = pj.SensitiveInt(0)
    for key, pdf_ in pdf.groupby("b", keys=[1, 2, 3, 4, 5]):
        s += pdf_.shape[0]
    assert s.distance.max() == 1
    assert s._value == len(pdf._value)
    pj.laplace_mechanism(s, eps=eps)
    assert pj.consumed_privacy_budget()[pdf.root_name()] == eps * 7

    s = pj.SensitiveInt(0)
    for key, pdf_ in pdf.groupby("b", keys=[1, 2, 3, 4, 5]):
        s += key * pdf_.shape[0]
    assert s.distance.max() == 5
    pj.laplace_mechanism(s, eps=eps)
    assert pj.consumed_privacy_budget()[pdf.root_name()] == eps * 8
