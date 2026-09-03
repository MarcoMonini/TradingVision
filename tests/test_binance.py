"""Offline checks for the Binance dump parser: python tests/test_binance.py"""

import io
import zipfile

import pandas as pd

from tradingvision.data.binance import parse

ROW = "{t},4261.48,4280.56,4200.00,4270.00,2.189,{c},9333.62,9,0.489,2089.10,0"
HEADER = "open_time,open,high,low,close,volume,close_time,quote_volume,trades,a,b,ignore"


def zipped(*lines: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.csv", "\n".join(lines))
    return buf.getvalue()


def test_timestamp_unit_is_inferred():
    """Binance switched open_time from ms to us during 2025, so both must land on the same date."""
    ms = parse(zipped(ROW.format(t=1502942400000, c=1502942699999)))
    us = parse(zipped(ROW.format(t=1502942400000000, c=1502942699999999)))
    assert ms.index[0] == us.index[0] == pd.Timestamp("2017-08-17 04:00:00", tz="UTC")


def test_header_row_is_dropped():
    df = parse(zipped(HEADER, ROW.format(t=1502942400000, c=1502942699999)))
    assert len(df) == 1 and df.close.iloc[0] == 4270.0


def test_columns_and_dtypes():
    df = parse(zipped(ROW.format(t=1502942400000, c=1502942699999)))
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "quote_volume", "trades"]
    assert df.close.dtype == "float64" and df.trades.dtype == "int32"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
