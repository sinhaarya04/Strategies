"""Workaround runner: pip is broken (libexpat mismatch) so pyarrow can't install.
Monkey-patch DataFrame.to_parquet to a no-op so fetch_es_data() doesn't crash on cache write."""
import pandas as pd
pd.DataFrame.to_parquet = lambda self, *a, **kw: None
import backtest_fisher_3lot
backtest_fisher_3lot.main()
