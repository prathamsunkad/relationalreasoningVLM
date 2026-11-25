"""Quota bridge must fail closed; byte, inode and metadata checks all remain."""
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from pgmvlm.storage import require_headroom, storage_report, StorageHeadroomError, parse_lustre_quota
from tools.storage_quota_bridge import collect


class StorageBridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='quota-',dir='/tmp')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.endpoint=str(self.root/'q.sock')

    def response(self,**overrides):
        return dict(path=str(self.root),uid=os.getuid(),job_id='42',checked_unix=time.time(),
            free_bytes=12*1024**3,free_files=20000,filesystem_free_bytes=100*1024**3,
            filesystem_free_files=100000,metadata_free_files=3000,**overrides)

    def request(self,response):
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
            server.bind(self.endpoint);server.listen(1)
            def answer():
                client,_=server.accept()
                with client:
                    with client.makefile('rb') as f: json.loads(f.readline())
                    client.sendall(json.dumps(response).encode()+b'\n')
            thread=threading.Thread(target=answer);thread.start()
            try:
                with patch.dict(os.environ,PGMVLM_STORAGE_SOCKET=self.endpoint,SLURM_JOB_ID='42'):
                    return require_headroom(self.root)
            finally:
                thread.join(timeout=2);Path(self.endpoint).unlink(missing_ok=True)

    def test_live_response_and_all_limits(self):
        self.assertEqual(self.request(self.response())['free_files'],20000)
        for changes in ({'free_bytes':0},{'free_files':1},{'metadata_free_files':1}):
            response=self.response();response.update(changes)
            with self.subTest(changes=changes),self.assertRaises(StorageHeadroomError):
                self.request(response)

    def test_rejects_stale_wrong_identity_and_errors(self):
        for changes in ({'checked_unix':time.time()-120},{'uid':-1},{'job_id':'43'},
                        {'path':'/elsewhere'},{'error':'lfs failed'}):
            response=self.response();response.update(changes)
            with self.subTest(changes=changes),self.assertRaises(StorageHeadroomError):
                self.request(response)
        with patch.dict(os.environ,PGMVLM_STORAGE_SOCKET=self.endpoint,SLURM_JOB_ID='42'):
            with self.assertRaises(StorageHeadroomError):storage_report(self.root)

    def test_host_checks_target_quota_and_restricts_paths(self):
        quota='ice1-MDT0003 0 - 0 - 950 - 1000 -\nTotal allocated inode limit: 19900, total allocated block limit: 0\n'
        native=dict(path=str(self.root),mount='/storage/ice1',uid=os.getuid(),limit_files=20000)
        with patch('tools.storage_quota_bridge._native_storage_report',return_value=native), \
             patch('tools.storage_quota_bridge.subprocess.check_output',side_effect=[quota,'3\n']):
            report=collect(self.root,self.root,'42')
        self.assertEqual(report['metadata_free_files'],150)
        with self.assertRaises(StorageHeadroomError):collect(self.root.parent,self.root,'42')

    def test_native_quota_parser_unchanged(self):
        report=parse_lustre_quota('/storage/ice1 100 1000 1000 - 10 100 100 -','/storage/ice1')
        self.assertEqual(report['free_bytes'],900*1024)
        self.assertEqual(report['free_files'],90)


if __name__=='__main__':unittest.main()
