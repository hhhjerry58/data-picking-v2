"""Read only assigned documents from immutable prepared RPJv2 shards."""
from collections import defaultdict
from pathlib import Path
import os
import shutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from .common import paths, read_json, write_json, file_hash


def prepared_shard(cfg, index):
    p = paths(cfg)
    source = p["data"] / "source_shards" / f"part-{index:05d}.parquet"
    expected = read_json(str(source) + ".complete.json")
    destination = p["scratch"] / "source_shards" / source.name
    marker = Path(str(destination) + ".verified.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if marker.exists() and destination.exists():
        record = read_json(marker)
        stat = destination.stat()
        if record["sha256"] == expected["sha256"] and record["size"] == stat.st_size and record["mtime_ns"] == stat.st_mtime_ns:
            return destination
    temporary = destination.with_name(destination.name + f".copy-{os.getpid()}")
    shutil.copyfile(source, temporary)
    if temporary.stat().st_size != expected["size"] or file_hash(temporary) != expected["sha256"]:
        raise RuntimeError("Corpus RAM copy does not match persistent source")
    os.replace(temporary, destination)  # POSIX scratch only; persistent source is unchanged.
    write_json(marker, {"source": str(source), "sha256": expected["sha256"], "size": destination.stat().st_size,
                        "mtime_ns": destination.stat().st_mtime_ns, "ephemeral_only": True})
    return destination


def assignment_table(cfg, partitions=None):
    p = paths(cfg)
    marker = read_json(p["data"] / "CORPUS_COMPLETED.json")
    path = p["data"] / "assignments.parquet"
    if file_hash(path) != marker["assignment_sha256"]:
        raise RuntimeError("Corpus assignment fingerprint mismatch")
    table = pq.read_table(path)
    return table.filter(pc.is_in(table["partition"], value_set=pa.array(partitions))) if partitions else table


def batches(cfg, partitions, columns=None, batch_size=1024, source_indexes=None):
    p = paths(cfg)
    table = assignment_table(cfg, partitions).select(["id", "source_index", "partition"])
    by_shard = defaultdict(dict)
    for row in table.to_pylist():
        if source_indexes is None or row["source_index"] in source_indexes:
            by_shard[row["source_index"]][row["id"]] = row["partition"]
    del table
    columns = list(columns) if columns else None
    if columns is not None and "id" not in columns:
        columns.insert(0, "id")
    for index in sorted(by_shard):
        selected = by_shard[index]
        path = prepared_shard(cfg, index)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
            ids = batch.column(batch.schema.get_field_index("id"))
            mask = pa.array([value.as_py() in selected for value in ids])
            kept = batch.filter(mask).to_pylist()
            if kept:
                yield [{**row, "partition": selected[row["id"]]} for row in kept]
