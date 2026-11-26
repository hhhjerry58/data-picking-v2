def score(rows, *, seed):
    return [(row['loss_before'] - row['loss_after']) / max(row['loss_before'], 1e-8)
            for row in rows]
