def score(rows, *, seed):
    return [row['loss_before'] - row['loss_after'] for row in rows]
