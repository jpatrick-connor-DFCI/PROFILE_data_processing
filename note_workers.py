"""Worker helpers for the clinical-note compilation notebooks.

These live in an importable module rather than in the notebooks because the
process pools use the ``forkserver`` start method, which pickles functions *by
name*: anything defined in a notebook cell fails to unpickle in the worker
(``attribute lookup failed on __main__``). Hence this file.

Keep this module import-safe and side-effect free: ``forkserver`` preloads it
in the server process that every worker is forked from.

Why ``forkserver`` and not the default: polars sizes its thread pool once, at
import, from the visible CPU count. Under ``fork`` every worker inherits the
parent's fully-built full-width pool, so N workers each run N threads and
oversubscribe the node. Capping it requires ``POLARS_MAX_THREADS`` to be set
*before polars is imported in the child*, which a ``ProcessPoolExecutor``
``initializer`` cannot do -- by the time it runs, the preloaded parent modules
(including polars) are already in memory. Setting the variable in the parent
environment before the forkserver starts is what actually works; see
``note_pool_context`` below.
"""

import os
from pathlib import Path

import orjson
import polars as pl


def note_pool_context(worker_threads: int = 1):
    """Build a multiprocessing context whose workers run a capped polars.

    Sets ``POLARS_MAX_THREADS`` in the *parent* environment so the forkserver
    process -- and therefore every worker forked from it -- imports polars with
    a narrow thread pool. The parent's own already-imported polars keeps its
    full width, so interactive work in the notebook is unaffected.

    Preloading only this module (rather than the default ``__main__``) keeps the
    server process from re-executing the notebook's driver code.
    """
    import multiprocessing as mp
    import sys

    # The forkserver starts a fresh interpreter, so it only finds this module if
    # its directory is importable. The notebooks are normally run from the repo
    # root, but don't rely on the cwd.
    module_dir = str(Path(__file__).resolve().parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    os.environ["POLARS_MAX_THREADS"] = str(worker_threads)
    ctx = mp.get_context("forkserver")
    ctx.set_forkserver_preload(["note_workers"])
    return ctx


def read_json_text_file(filename, schema, oncdrs_path=None, parse_event_date=False):
    """Read one OncDRS note JSON into a polars frame.

    The note dumps are Solr-style envelopes with the rows under
    ``data["response"]["docs"]``.

    This unifies the two divergent copies that previously lived in the
    notebooks. The metadata stage tags each row with its source ``FILE`` and
    leaves ``EVENT_DATE`` a string (it is parsed once after the concat); the
    text stage parses ``EVENT_DATE`` per file and adds no ``FILE`` column, since
    it joins back to metadata that already carries one.
    """
    with open(filename, "rb") as f:
        data = orjson.loads(f.read())

    df = pl.from_dicts(data["response"]["docs"], schema=schema)

    if oncdrs_path is not None:
        file_suffix = str(filename.relative_to(oncdrs_path))
        df = df.with_columns(pl.lit(file_suffix).alias("FILE"))

    if parse_event_date:
        df = df.with_columns(
            pl.col("EVENT_DATE").str.to_datetime(
                format="%Y-%m-%dT%H:%M:%SZ", time_zone="UTC"
            )
        )

    return df


def clean_path_image_notes(text_df):
    """Merge ``RPT_TEXT`` and ``NARRATIVE_TEXT`` into a single ``RPT_TEXT``.

    Pathology and imaging notes carry two text fields that are often
    duplicative: sometimes one is empty, sometimes one wholly contains the
    other, sometimes they are genuinely different and both matter.

    The equality check and the length guards are pure short-circuits for the
    two ``str.contains`` calls, which are per-row substring searches over full
    note bodies and the most expensive operation in the pipeline. A string
    cannot contain a longer string, so the guards never change the outcome;
    the equality branch returns ``RPT_TEXT``, matching the original's
    first-matching-branch behavior when both sides are equal.
    """
    rpt_len = pl.col("RPT_TEXT").str.len_bytes()
    narrative_len = pl.col("NARRATIVE_TEXT").str.len_bytes()

    cleaned_df = (
        text_df
        .with_columns(pl.col(["RPT_TEXT", "NARRATIVE_TEXT"]).replace("", None))
        .with_columns(
            pl.when(pl.col("RPT_TEXT").is_null())
            .then(pl.col("NARRATIVE_TEXT"))
            .when(pl.col("NARRATIVE_TEXT").is_null())
            .then(pl.col("RPT_TEXT"))
            .when(pl.col("RPT_TEXT") == pl.col("NARRATIVE_TEXT"))
            .then(pl.col("RPT_TEXT"))
            .when(
                (rpt_len >= narrative_len)
                & pl.col("RPT_TEXT").str.contains(
                    pl.col("NARRATIVE_TEXT"), literal=True
                )
            )
            .then(pl.col("RPT_TEXT"))
            .when(
                (narrative_len >= rpt_len)
                & pl.col("NARRATIVE_TEXT").str.contains(
                    pl.col("RPT_TEXT"), literal=True
                )
            )
            .then(pl.col("NARRATIVE_TEXT"))
            .otherwise(
                pl.concat_str(["RPT_TEXT", "NARRATIVE_TEXT"], separator="\n\n")
            )
            .alias("COMBINED_TEXT")
        )
        .drop(["RPT_TEXT", "NARRATIVE_TEXT"])
        .rename({"COMBINED_TEXT": "RPT_TEXT"})
    )
    return cleaned_df


def process_note_file(work_item, schema, join_cols, oncdrs_path, clean):
    """Read one note file, merge its text fields, and join it to its metadata.

    ``work_item`` is ``(file_name, file_meta)`` -- each worker receives only its
    own metadata slice, never the whole frame. Returns ``(file_name, frame)`` so
    the parent can assert per-file row counts without shipping the counts table
    to every worker.
    """
    file_name, file_meta = work_item

    df = read_json_text_file(
        oncdrs_path / file_name, schema, parse_event_date=True
    )

    if clean:
        df = clean_path_image_notes(df)

    joined = df.join(file_meta, on=join_cols, how="inner", nulls_equal=True)
    return file_name, joined
