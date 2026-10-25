"""
Brazil DI1 futures curve history (free, no API key).

Pulls daily B3 settlement data for every DI1 contract via the PYield library
(history from 2 Jan 2018), caches it locally, and writes:

  1. di1_contracts.csv          one row per date x contract (settlement rate,
                                business days to expiry, open interest, volume)
  2. di1_constant_maturity.csv  one row per date, fixed tenors interpolated
                                flat-forward on the 252-business-day basis
                                (removes roll jumps; use for slope history)
  3. di1_slopes.csv             tenor slopes and the chosen contract-pair
                                spread, in basis points
  4. di1_slopes.png             chart of the above (if matplotlib is installed)

Rates are in % (e.g. 13.75), spreads in bp. Re-running only fetches dates not
already in the cache.

Setup:   pip install pyield polars matplotlib
Usage:   python di_curve_history.py
         python di_curve_history.py --start 2022-01-01 --pair F27 F31
         python di_curve_history.py --tenors 252 504 756 1260 --slope 252 1260
"""

import argparse
import datetime as dt
from pathlib import Path

import polars as pl
import pyield as yd

CACHE = Path("di1_cache.csv")
CHUNK = 60  # trading dates per request


def parse_args():
    p = argparse.ArgumentParser(description="Download DI1 curve history from B3 via PYield")
    p.add_argument("--start", default="2018-01-02", help="first date (YYYY-MM-DD); data begins 2018-01-02")
    p.add_argument("--end", default=None, help="last date (YYYY-MM-DD); default = latest available")
    p.add_argument("--tenors", nargs="+", type=int, default=[126, 252, 504, 756, 1260, 2520],
                   help="constant-maturity tenors in business days (252 = 1y)")
    p.add_argument("--slope", nargs=2, type=int, default=[252, 1260], metavar=("SHORT", "LONG"),
                   help="tenor pair for the headline slope, in business days")
    p.add_argument("--pair", nargs=2, default=["F27", "F31"], metavar=("SHORT", "LONG"),
                   help="actual contract pair for the steepener, e.g. F27 F31 (month code + 2-digit year)")
    p.add_argument("--outdir", default=".", help="output folder")
    return p.parse_args()


def load_cache() -> pl.DataFrame:
    if CACHE.exists():
        return pl.read_csv(CACHE, try_parse_dates=True)
    return pl.DataFrame()


def fetch(dates: list[dt.date]) -> pl.DataFrame:
    """Fetch DI1 settlement data for the given trading dates, in chunks."""
    keep = ["data_referencia", "codigo_negociacao", "data_vencimento", "dias_uteis",
            "taxa_ajuste", "taxa_fechamento", "contratos_abertos", "numero_negocios"]
    frames = []
    for i in range(0, len(dates), CHUNK):
        batch = dates[i:i + CHUNK]
        print(f"  fetching {batch[0]} to {batch[-1]} ({len(batch)} dates)")
        try:
            df = yd.di1.dados(batch)
        except Exception as e:  # B3 format changes or network errors: report and carry on
            print(f"  ! failed for this batch: {e}")
            continue
        if not df.is_empty():
            frames.append(df.select([c for c in keep if c in df.columns]))
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def update_cache(start: dt.date, end: dt.date) -> pl.DataFrame:
    cache = load_cache()
    available = [d for d in yd.di1.datas_disponiveis().to_list() if start <= d <= end]
    have = set(cache["data_referencia"].to_list()) if not cache.is_empty() else set()
    missing = [d for d in available if d not in have]
    print(f"{len(available)} trading dates in range, {len(have & set(available))} cached, "
          f"{len(missing)} to fetch")
    if missing:
        new = fetch(missing)
        if not new.is_empty():
            cache = pl.concat([cache, new], how="diagonal_relaxed") if not cache.is_empty() else new
            cache = cache.unique(["data_referencia", "codigo_negociacao"]).sort(
                ["data_referencia", "data_vencimento"])
            cache.write_csv(CACHE)
    return cache.filter(pl.col("data_referencia").is_between(start, end))


def contracts_table(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(
        pl.col("data_referencia").alias("date"),
        pl.col("codigo_negociacao").alias("ticker"),
        pl.col("codigo_negociacao").str.slice(3).alias("contract"),  # DI1F27 -> F27
        pl.col("data_vencimento").alias("expiry"),
        pl.col("dias_uteis").alias("bdays_to_expiry"),
        (pl.col("taxa_ajuste") * 100).round(4).alias("settle_rate_pct"),
        (pl.col("taxa_fechamento") * 100).round(4).alias("close_rate_pct"),
        pl.col("contratos_abertos").alias("open_interest"),
        pl.col("numero_negocios").alias("trades"),
    ).sort(["date", "expiry"])


def constant_maturity(c: pl.DataFrame, tenors: list[int]) -> pl.DataFrame:
    """Flat-forward interpolation of each day's settlement curve at fixed tenors."""
    curve = c.filter(pl.col("settle_rate_pct").is_not_null() & (pl.col("bdays_to_expiry") > 0))
    dates = curve["date"].unique().sort()
    targets = pl.DataFrame({"date": dates}).join(
        pl.DataFrame({"tenor_bd": tenors}), how="cross")
    rates = yd.di1.interpolar(
        dus_alvo=targets["tenor_bd"],
        dus_curva=curve["bdays_to_expiry"],
        taxas_curva=curve["settle_rate_pct"] / 100,
        datas_alvo=targets["date"],
        datas_curva=curve["date"],
        extrapolar=False,  # leave long tenors blank on days the curve doesn't reach them
    )
    targets = targets.with_columns((rates * 100).round(4).alias("rate_pct"))
    wide = targets.pivot(on="tenor_bd", index="date", values="rate_pct").sort("date")
    return wide.rename({str(t): f"cm_{t}bd" for t in tenors})


def slopes(cm: pl.DataFrame, c: pl.DataFrame, slope: list[int], pair: list[str]) -> pl.DataFrame:
    s, l = slope
    out = cm.select("date", ((pl.col(f"cm_{l}bd") - pl.col(f"cm_{s}bd")) * 100).round(1)
                    .alias(f"slope_{s}_{l}bd_bp"))
    short, long_ = pair
    legs = (c.filter(pl.col("contract").is_in([short, long_]))
             .pivot(on="contract", index="date", values="settle_rate_pct"))
    if short in legs.columns and long_ in legs.columns:
        legs = legs.select("date", ((pl.col(long_) - pl.col(short)) * 100).round(1)
                           .alias(f"pair_{long_}_minus_{short}_bp"))
        out = out.join(legs, on="date", how="left")
    else:
        print(f"! contract pair {short}/{long_} not found in the data range")
    return out.sort("date")


def chart(sl: pl.DataFrame, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping chart")
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    for col in sl.columns[1:]:
        ax.plot(sl["date"].to_list(), sl[col].to_list(), label=col, linewidth=1.2)
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_ylabel("bp")
    ax.set_title("Brazil DI curve slope")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main():
    a = parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    start = dt.date.fromisoformat(a.start)
    end = dt.date.fromisoformat(a.end) if a.end else dt.date.today()
    tenors = sorted(set(a.tenors) | set(a.slope))

    raw = update_cache(start, end)
    if raw.is_empty():
        raise SystemExit("No data returned. Check the date range or update PYield (pip install -U pyield).")

    c = contracts_table(raw)
    cm = constant_maturity(c, tenors)
    sl = slopes(cm, c, a.slope, a.pair)

    for name, df in [("di1_contracts.csv", c), ("di1_constant_maturity.csv", cm), ("di1_slopes.csv", sl)]:
        df.write_csv(out / name)
        print(f"wrote {out / name} ({df.height} rows)")
    chart(sl, out / "di1_slopes.png")
    print("\nLatest:")
    print(cm.tail(1))
    print(sl.tail(1))


if __name__ == "__main__":
    main()
