import base64
import hashlib
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import xml.etree.ElementTree as ET

from picking.blob_transport import DirectCheckpointReader
from picking.common import file_hash


def test_self_contained_blob_transport_validates_and_reads_back(tmp_path):
    blocks={};committed=[];blob=b''
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_PUT(self):
            nonlocal blob,committed
            query=urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            data=self.rfile.read(int(self.headers['Content-Length']))
            if query['comp']==['block']:
                md5=base64.b64encode(hashlib.md5(data).digest()).decode()
                assert md5==self.headers['Content-MD5']
                blocks[query['blockid'][0]]=data
                self.send_response(201);self.send_header('Content-MD5',md5);self.end_headers()
            else:
                committed=[item.text for item in ET.fromstring(data)]
                blob=b''.join(blocks[item] for item in committed)
                assert hashlib.sha256(blob).hexdigest()==self.headers['x-ms-meta-sha256']
                self.send_response(201);self.end_headers()
        def do_HEAD(self):
            self.send_response(200);self.send_header('ETag','"test-etag"');self.send_header('Content-Length',str(len(blob)));self.end_headers()
        def do_GET(self):
            query=urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            if query.get('comp')==['blocklist']:
                body=('<BlockList><CommittedBlocks>'+''.join(f'<Block><Name>{name}</Name><Size>{len(blocks[name])}</Size></Block>' for name in committed)+'</CommittedBlocks></BlockList>').encode()
                self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            else:
                assert self.headers['If-Match']=='"test-etag"'
                start,end=map(int,self.headers['Range'].removeprefix('bytes=').split('-'))
                body=blob[start:end+1]
                self.send_response(206);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        reader=DirectCheckpointReader();reader.mount=tmp_path/'mount';reader.run=reader.mount/'run';reader.root=reader.run/'checkpoints'
        reader.storage={'endpoint':f'http://127.0.0.1:{server.server_port}','container':'unit-test','sas':'unit-test-token'}
        reader.written_copies={}
        local=tmp_path/'source.pt';local.write_bytes(b'full-model-and-optimizer-state'*8192)
        record=reader.persist(local,reader.root/'step-000001/state.pt',full_readback=True)
        assert blob==local.read_bytes()
        assert record['sha256']==file_hash(local)
        assert record['full_remote_readback'] is True
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2)
