import pandas as pd
import pytest
from analysis.statistics import _summary_stats
from data.exceptions import InvalidDataError


def test_summary_stats_numeric_check():
    df = pd.DataFrame({
        'Sleep Duration': [6.0, 8.0, 10.0],
        'Age': [20, 30, 40],
        'Daily Steps': [1000, 2000, 3000]
    })

    stats = _summary_stats(df, ['Sleep Duration', 'Age'])

    assert stats.loc['Sleep Duration', 'mean'] == 8.0

    assert stats.loc['Age', 'max'] == 40.0

    assert stats.loc['Sleep Duration', 'min'] == 6.0

def test_summary_stats_raises_on_non_numeric():
    df = pd.DataFrame({
        'Sleep Duration': [6.0, 7.0],
        'Name': ['Alice', 'Bob']
    })
    with pytest.raises(InvalidDataError):
        _summary_stats(df, ['Name'])

# #Jak to działa:
# Pytest „wchodzi” w blok with.
# Uruchamia kod w bloku (summary_stats(df, ['Name'])).
# Jeśli kod rzuci dokładnie wyjątek InvalidDataError → test przechodzi.
# Jeśli kod nie rzuci wyjątku → test nie przechodzi, pytest zgłasza błąd.
# Jeśli kod rzuci inny wyjątek → test też nie przechodzi, pytest pokazuje inny błąd.