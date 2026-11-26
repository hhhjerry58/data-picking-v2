import hashlib


def score(rows, *, seed):
    return [int.from_bytes(hashlib.blake2b(f'{seed}:random:{row["id"]}'.encode(),
                                         digest_size=8).digest(), 'big') / 2**64
            for row in rows]
