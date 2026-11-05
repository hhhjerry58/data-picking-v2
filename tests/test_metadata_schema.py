import pyarrow as pa
from picking.finalize_corpus import META_SCHEMA


def test_full_unsigned_document_hash_is_preserved():
    row={'id':'document','source_index':1,'source_row':2,'domain':'example.org','crawl':'2023-14',
         'text_hash':'hash','sample_key':2**64-1,'pythia_tokens':1234}
    table=pa.Table.from_pylist([row],schema=META_SCHEMA)
    assert table['sample_key'][0].as_py()==2**64-1
