import pytest
import pandas as pd
import pripri
from pripri import pandas as ppd

def load_dataframe():
    data = {
        "a": [1, 2, 3, 4, 5],
        "b": [2, 4, 4, 4, 3],
    }
    pdf = ppd.DataFrame(data)
    df = pd.DataFrame(data)
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()
    return pdf, df

def test_dataframe_size():
    pdf, df = load_dataframe()

    # The `shape` member should be a pair of a sensitive value (row) and a non-sensitive value (column)
    assert isinstance(pdf.shape, tuple)
    assert len(pdf.shape) == len(df.shape) == 2
    assert isinstance(pdf.shape[0], pripri.Prisoner)
    assert pdf.shape[0]._value == df.shape[0]
    assert pdf.shape[0].sensitivity == 1
    assert pdf.shape[1] == len(pdf.columns) == len(df.columns)

    # The `size` member should be a sensitive value
    assert isinstance(pdf.size, pripri.Prisoner)
    assert pdf.size._value == df.size
    assert pdf.size.sensitivity == len(pdf.columns) == len(df.columns)

    # Builtin `len()` function should raise an error because it must be an integer value
    with pytest.raises(pripri.DPError):
        len(pdf)

def test_dataframe_comp():
    pdf, df = load_dataframe()

    # A non-sensitive value should be successfully compared against a sensitive dataframe
    assert isinstance(pdf == 3, ppd.DataFrame)
    assert isinstance(pdf != 3, ppd.DataFrame)
    assert isinstance(pdf <  3, ppd.DataFrame)
    assert isinstance(pdf <= 3, ppd.DataFrame)
    assert isinstance(pdf >  3, ppd.DataFrame)
    assert isinstance(pdf >= 3, ppd.DataFrame)
    assert ((pdf == 3)._value == (df == 3)).all().all()
    assert ((pdf != 3)._value == (df != 3)).all().all()
    assert ((pdf <  3)._value == (df <  3)).all().all()
    assert ((pdf <= 3)._value == (df <= 3)).all().all()
    assert ((pdf >  3)._value == (df >  3)).all().all()
    assert ((pdf >= 3)._value == (df >= 3)).all().all()

    # A non-sensitive value should be successfully compared against a sensitive series
    assert isinstance(pdf["a"] == 3, ppd.Series)
    assert isinstance(pdf["a"] != 3, ppd.Series)
    assert isinstance(pdf["a"] <  3, ppd.Series)
    assert isinstance(pdf["a"] <= 3, ppd.Series)
    assert isinstance(pdf["a"] >  3, ppd.Series)
    assert isinstance(pdf["a"] >= 3, ppd.Series)
    assert ((pdf["a"] == 3)._value == (df["a"] == 3)).all()
    assert ((pdf["a"] != 3)._value == (df["a"] != 3)).all()
    assert ((pdf["a"] <  3)._value == (df["a"] <  3)).all()
    assert ((pdf["a"] <= 3)._value == (df["a"] <= 3)).all()
    assert ((pdf["a"] >  3)._value == (df["a"] >  3)).all()
    assert ((pdf["a"] >= 3)._value == (df["a"] >= 3)).all()

    # An irrelevant, non-sensitive dataframe should not be compared against a sensitive dataframe
    with pytest.raises(pripri.DPError): pdf == df
    with pytest.raises(pripri.DPError): pdf != df
    with pytest.raises(pripri.DPError): pdf <  df
    with pytest.raises(pripri.DPError): pdf <= df
    with pytest.raises(pripri.DPError): pdf >  df
    with pytest.raises(pripri.DPError): pdf >= df

    # An irrelevant, non-sensitive 2d array should not be compared against a sensitive dataframe
    with pytest.raises(pripri.DPError): pdf == [[list(range(len(pdf.columns)))] for x in range(5)]
    with pytest.raises(pripri.DPError): pdf != [[list(range(len(pdf.columns)))] for x in range(5)]
    with pytest.raises(pripri.DPError): pdf <  [[list(range(len(pdf.columns)))] for x in range(5)]
    with pytest.raises(pripri.DPError): pdf <= [[list(range(len(pdf.columns)))] for x in range(5)]
    with pytest.raises(pripri.DPError): pdf >  [[list(range(len(pdf.columns)))] for x in range(5)]
    with pytest.raises(pripri.DPError): pdf >= [[list(range(len(pdf.columns)))] for x in range(5)]

    # An irrelevant, non-sensitive array should not be compared against a sensitive series
    with pytest.raises(pripri.DPError): pdf["a"] == [0, 1, 2, 3, 4]
    with pytest.raises(pripri.DPError): pdf["a"] != [0, 1, 2, 3, 4]
    with pytest.raises(pripri.DPError): pdf["a"] <  [0, 1, 2, 3, 4]
    with pytest.raises(pripri.DPError): pdf["a"] <= [0, 1, 2, 3, 4]
    with pytest.raises(pripri.DPError): pdf["a"] >  [0, 1, 2, 3, 4]
    with pytest.raises(pripri.DPError): pdf["a"] >= [0, 1, 2, 3, 4]

    x = pripri.Prisoner(value=0, sensitivity=1)

    # A sensitive value should not be compared against a sensitive dataframe
    with pytest.raises(pripri.DPError): pdf == x
    with pytest.raises(pripri.DPError): pdf != x
    with pytest.raises(pripri.DPError): pdf <  x
    with pytest.raises(pripri.DPError): pdf <= x
    with pytest.raises(pripri.DPError): pdf >  x
    with pytest.raises(pripri.DPError): pdf >= x

    # A sensitive value should not be compared against a sensitive series
    with pytest.raises(pripri.DPError): pdf["a"] == x
    with pytest.raises(pripri.DPError): pdf["a"] != x
    with pytest.raises(pripri.DPError): pdf["a"] <  x
    with pytest.raises(pripri.DPError): pdf["a"] <= x
    with pytest.raises(pripri.DPError): pdf["a"] >  x
    with pytest.raises(pripri.DPError): pdf["a"] >= x

def test_dataframe_getitem():
    pdf, df = load_dataframe()

    # A single-column view should be successfully retrieved from a sensitive dataframe
    assert isinstance(pdf["a"], ppd.Series)
    assert (pdf["a"]._value == df["a"]).all()

    # A multi-column view should be successfully retrieved from a sensitive dataframe
    assert isinstance(pdf[["a", "b"]], ppd.DataFrame)
    assert (pdf[["a", "b"]]._value == df[["a", "b"]]).all().all()

    # An irrelevant, non-sensitve bool vector should not be accepted for filtering (dataframe)
    with pytest.raises(pripri.DPError):
        pdf[[True, True, False, False, True]]

    # An irrelevant, non-sensitve bool vector should not be accepted for filtering (series)
    with pytest.raises(pripri.DPError):
        pdf["a"][[True, True, False, False, True]]

    # A bool-filtered view should be successfully retrieved from a sensitive dataframe
    assert isinstance(pdf[pdf["a"] > 3], ppd.DataFrame)
    assert (pdf[pdf["a"] > 3]._value == df[df["a"] > 3]).all().all()

    # A bool-filtered view should be successfully retrieved from a sensitive series
    assert isinstance(pdf["a"][pdf["a"] > 3], ppd.Series)
    assert (pdf["a"][pdf["a"] > 3]._value == df["a"][df["a"] > 3]).all()

    x = pripri.Prisoner(value=0, sensitivity=1)

    # A sensitive value should not be used as a column name
    with pytest.raises(pripri.DPError):
        pdf[x]

    # A slice should not be used for selecting rows of a dataframe
    # TODO: it might be legal to accept a slice
    with pytest.raises(pripri.DPError):
        pdf[2:5]

    # A slice should not be used for selecting rows of a series
    # TODO: it might be legal to accept a slice
    with pytest.raises(pripri.DPError):
        pdf["a"][2:5]

def test_dataframe_setitem():
    pdf, df = load_dataframe()

    # A non-sensitive value should be successfully assigned to a single-column view
    pdf["c"] = 10
    df["c"] = 10
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    # A sensitive series should be successfully assigned to a single-column view
    pdf["d"] = pdf["a"]
    df["d"] = df["a"]
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    # A non-sensitive value should be successfully assigned to a multi-column view
    pdf[["e", "f"]] = 100
    df[["e", "f"]] = 100
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    # A sensitive dataframe should be successfully assigned to a multi-column view
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

    # A sensitive dataframe should be successfully assigned to a bool-filtered view
    pdf[pdf["a"] < 3] = pdf[pdf["a"] > 3].reset_index(drop=True)
    df[df["a"] < 3] = df[df["a"] > 3].reset_index(drop=True)
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    # A non-sensitive series can be assigned to a sensitive single-column view
    # (because this operation succeeds regardless of the row number)
    # TODO: this may be disallowed if we track the value domain for each column
    pdf["i"] = df["c"]
    df["i"] = df["c"]
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    # A non-sensitive dataframe can be assigned to a sensitive multi-column view
    # (because this operation succeeds regardless of the row number)
    # TODO: this may be disallowed if we track the value domain for each column
    pdf[["j", "k"]] = df[["a", "b"]]
    df[["j", "k"]] = df[["a", "b"]]
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    # An irrelevant, non-sensitive dataframe can be assigned to a bool-filtered view
    # (because this operation succeeds regardless of the row number)
    # TODO: this may be disallowed if we track the value domain for each column
    pdf[pdf["a"] > 3] = df
    df[df["a"] > 3] = df
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    x = pripri.Prisoner(value=0, sensitivity=1)

    # A sensitive value should not be assigned to a single-column view
    with pytest.raises(pripri.DPError):
        pdf["x"] = x

    # A sensitive value should not be assigned to a multi-column view
    with pytest.raises(pripri.DPError):
        pdf[["x", "y"]] = x

    # A sensitive value should not be assigned to a bool-filtered view
    with pytest.raises(pripri.DPError):
        pdf[pdf["a"] > 3] = x

    # An irrelevant, non-sensitive array should not be assigned to a column view
    with pytest.raises(pripri.DPError):
        pdf["x"] = [1, 2, 3, 4, 5]

    # An irrelevant, non-sensitive 2d array should not be assigned to a bool-filtered view
    with pytest.raises(pripri.DPError):
        pdf[pdf["a"] > 3] = [[list(range(len(pdf.columns)))] for x in range(5)]

    # A sensitive value should not be used as a column name
    with pytest.raises(pripri.DPError):
        pdf[x] = 10

    # A slice should not be used for selecting rows
    # TODO: it might be legal to accept a slice
    with pytest.raises(pripri.DPError):
        pdf[2:5] = 0

def test_dataframe_reset_index():
    pdf, df = load_dataframe()

    # Default behaviour
    assert (pdf.reset_index()._value == df.reset_index()).all().all()

    # Special behaviour when `inplace=True`
    assert pdf.reset_index(inplace=True) == None
    assert (pdf._value == df.reset_index()).all().all()

def test_series_reset_index():
    pdf, df = load_dataframe()

    # Default behaviour
    assert isinstance(pdf["a"].reset_index(), ppd.DataFrame)
    assert (pdf["a"].reset_index()._value == df["a"].reset_index()).all().all()

    # Special behaviour when `drop=True`
    assert isinstance(pdf["a"].reset_index(drop=True), ppd.Series)
    assert (pdf["a"].reset_index(drop=True)._value == df["a"].reset_index(drop=True)).all().all()

    # Special behaviour when `drop=True` and `inplace=True`
    assert pdf["a"].reset_index(drop=True, inplace=True) == None
    assert df["a"].reset_index(drop=True, inplace=True) == None
    assert (pdf._value == df).all().all()
