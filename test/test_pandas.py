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

    assert isinstance(pdf.shape, tuple)
    assert len(pdf.shape) == len(df.shape) == 2
    assert isinstance(pdf.shape[0], pripri.Prisoner)
    assert pdf.shape[0]._value == df.shape[0]
    assert pdf.shape[0].sensitivity == 1
    assert pdf.shape[1] == len(pdf.columns) == len(df.columns)

    assert isinstance(pdf.size, pripri.Prisoner)
    assert pdf.size._value == df.size
    assert pdf.size.sensitivity == len(pdf.columns) == len(df.columns)

    with pytest.raises(Exception): len(pdf)

def test_dataframe_comp():
    pdf, df = load_dataframe()

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

    with pytest.raises(Exception): pdf == df
    with pytest.raises(Exception): pdf != df
    with pytest.raises(Exception): pdf <  df
    with pytest.raises(Exception): pdf <= df
    with pytest.raises(Exception): pdf >  df
    with pytest.raises(Exception): pdf >= df

    with pytest.raises(Exception): pdf["a"] == [0, 1, 2, 3, 4]
    with pytest.raises(Exception): pdf["a"] != [0, 1, 2, 3, 4]
    with pytest.raises(Exception): pdf["a"] <  [0, 1, 2, 3, 4]
    with pytest.raises(Exception): pdf["a"] <= [0, 1, 2, 3, 4]
    with pytest.raises(Exception): pdf["a"] >  [0, 1, 2, 3, 4]
    with pytest.raises(Exception): pdf["a"] >= [0, 1, 2, 3, 4]

    x = pripri.Prisoner(value=0, sensitivity=1)

    with pytest.raises(Exception): pdf == x
    with pytest.raises(Exception): pdf != x
    with pytest.raises(Exception): pdf <  x
    with pytest.raises(Exception): pdf <= x
    with pytest.raises(Exception): pdf >  x
    with pytest.raises(Exception): pdf >= x

    with pytest.raises(Exception): pdf["a"] == x
    with pytest.raises(Exception): pdf["a"] != x
    with pytest.raises(Exception): pdf["a"] <  x
    with pytest.raises(Exception): pdf["a"] <= x
    with pytest.raises(Exception): pdf["a"] >  x
    with pytest.raises(Exception): pdf["a"] >= x

def test_dataframe_getitem():
    pdf, df = load_dataframe()

    assert isinstance(pdf["a"], ppd.Series)
    assert (pdf["a"]._value == df["a"]).all()

    assert isinstance(pdf[["a", "b"]], ppd.DataFrame)
    assert (pdf[["a", "b"]]._value == df[["a", "b"]]).all().all()

    x = pripri.Prisoner(value=0, sensitivity=1)
    with pytest.raises(Exception): pdf[x]

    with pytest.raises(Exception): pdf[2:5]
    with pytest.raises(Exception): pdf[[True, True, False, False, True]]

    assert isinstance(pdf[pdf["a"] > 3], ppd.DataFrame)
    assert (pdf[pdf["a"] > 3]._value == df[df["a"] > 3]).all().all()

    with pytest.raises(Exception): pdf["a"][2:5]
    with pytest.raises(Exception): pdf["a"][[True, True, False, False, True]]

    assert isinstance(pdf["a"][pdf["a"] > 3], ppd.Series)
    assert (pdf["a"][pdf["a"] > 3]._value == df["a"][df["a"] > 3]).all()

def test_dataframe_setitem():
    pdf, df = load_dataframe()

    pdf["c"] = 10
    df["c"] = 10
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    pdf["d"] = pdf["a"]
    df["d"] = df["a"]
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    pdf[["e", "f"]] = 100
    df[["e", "f"]] = 100
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    pdf[["g", "h"]] = pdf[["a", "b"]]
    df[["g", "h"]] = df[["a", "b"]]
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    pdf[pdf["a"] < 3] = 8
    df[df["a"] < 3] = 8
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    pdf[pdf["a"] == 8] = list(range(len(pdf.columns)))
    df[df["a"] == 8] = list(range(len(df.columns)))
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    pdf[pdf["a"] < 3] = pdf[pdf["a"] > 3].reset_index(drop=True)
    df[df["a"] < 3] = df[df["a"] > 3].reset_index(drop=True)
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    pdf["i"] = df["c"]
    df["i"] = df["c"]
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    pdf[["j", "k"]] = df[["a", "b"]]
    df[["j", "k"]] = df[["a", "b"]]
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    pdf[pdf["a"] > 3] = df
    df[df["a"] > 3] = df
    assert (pdf.columns == df.columns).all()
    assert (pdf._value == df).all().all()

    x = pripri.Prisoner(value=0, sensitivity=1)
    with pytest.raises(Exception): pdf["x"] = x
    with pytest.raises(Exception): pdf["x"] = [1, 2, 3, 4, 5]
    with pytest.raises(Exception): pdf[["x", "y"]] = x
    with pytest.raises(Exception): pdf[pdf["a"] > 3] = x
    with pytest.raises(Exception): pdf[pdf["a"] > 3] = [1, 2, 3, 4, 5]
    with pytest.raises(Exception): pdf[x] = 10
    with pytest.raises(Exception): pdf[2:5] = 0

def test_dataframe_reset_index():
    pdf, df = load_dataframe()

    assert (pdf.reset_index()._value == df.reset_index()).all().all()

    assert pdf.reset_index(inplace=True) == None
    assert (pdf._value == df.reset_index()).all().all()

def test_series_reset_index():
    pdf, df = load_dataframe()

    assert isinstance(pdf["a"].reset_index(), ppd.DataFrame)
    assert (pdf["a"].reset_index()._value == df["a"].reset_index()).all().all()
    assert isinstance(pdf["a"].reset_index(drop=True), ppd.Series)
    assert (pdf["a"].reset_index(drop=True)._value == df["a"].reset_index(drop=True)).all().all()
    assert pdf["a"].reset_index(drop=True, inplace=True) == None
    assert df["a"].reset_index(drop=True, inplace=True) == None
    assert (pdf._value == df).all().all()
