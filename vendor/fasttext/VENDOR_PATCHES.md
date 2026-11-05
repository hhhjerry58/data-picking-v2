Pinned upstream: 1142dc4c4ecbc19cc16eee5cdd28472e689267e6.

The original archive remains checksummed under the experiment artifact root.
Two local changes are required for the approved single-thread, resumable run:

- `DenseMatrix::uniformThread` partitions the matrix into ten blocks. Upstream
  `uniform(..., thread=1, ...)` initialized only block zero, leaving approximately
  90% of the aligned allocation uninitialized. This produced non-finite updates
  depending on reused allocator memory. Initialization now visits all ten blocks
  for every thread setting, and rounds block size up to include the final tail.
  Each block keeps the original `block + seed` RNG and uniform distribution.
- `Model::State` gains serialization of its accumulated loss, example count,
  and RNG. The separate adapter persists token counters, optimizer/schedule
  configuration and input cursor with the standard model binary.

No loss, SGD update, dictionary, feature, or label rule is changed. Tests compare
the adapter's uninterrupted model bytes to the CLI built from this same fixed
backend, verify all input weights are initialized, and require an interrupted
run to recover identical final model and trainer-state bytes.
