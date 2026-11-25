"""Same-user, node-local quota service for containers lacking host Lustre tools.

Only read-only requests under the chosen run directory are accepted. Every
request invokes the host's actual lfs tools; no cached or synthetic quota data.
"""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pgmvlm.storage import _native_storage_report, StorageHeadroomError


def collect(path,root,job_id):
    path=Path(path).resolve();root=Path(root).resolve()
    if not path.is_relative_to(root) or not path.exists():
        raise StorageHeadroomError('Quota request is outside the existing run directory')
    report=_native_storage_report(path)
    mount=report['mount']
    quota=subprocess.check_output(['lfs','quota','-v','-u',str(os.getuid()),mount],text=True,timeout=20)
    mdt=int(subprocess.check_output(['lfs','getdirstripe','-m',str(path)],text=True,timeout=20).strip())
    target=next(line.split() for line in quota.splitlines()
                if line.split() and line.split()[0].endswith(f'MDT{mdt:04x}'))
    allocated=int(re.search(r'Total allocated inode limit:\s*(\d+)',quota).group(1))
    available=int(target[7])-int(target[5].rstrip('*'))+max(0,report['limit_files']-allocated)
    report.update(metadata_target=mdt,metadata_free_files=max(0,available),
        job_id=job_id,checked_unix=time.time(),provider='live_host_lfs')
    return report


def serve(endpoint,root,ready):
    endpoint,root,ready=Path(endpoint),Path(root).resolve(),Path(ready)
    if endpoint.exists() or not root.is_dir():
        raise ValueError('Need a new socket and an existing run directory')
    job=os.environ.get('SLURM_JOB_ID','')
    # Fail before publishing readiness if the host itself cannot query quota.
    collect(root,root,job)
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
        server.bind(str(endpoint));os.chmod(endpoint,0o600);server.listen(4)
        ready.write_text(json.dumps({'pid':os.getpid(),'uid':os.getuid(),'root':str(root),'job_id':job})+'\n')
        while True:
            client,_=server.accept()
            with client:
                client.settimeout(10)
                try:
                    _,uid,_=struct.unpack('3i',client.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
                    if uid!=os.getuid():raise ValueError('Wrong quota client UID')
                    with client.makefile('rb') as stream: raw=stream.readline(8193)
                    if len(raw)>8192 or not raw.endswith(b'\n'):raise ValueError('Malformed quota request')
                    request=json.loads(raw)
                    if request.get('job_id')!=job:raise ValueError('Wrong quota client job')
                    response=collect(request['path'],root,job)
                except Exception as exc:
                    response={'error':f'{type(exc).__name__}: {exc}'}
                try:client.sendall(json.dumps(response).encode()+b'\n')
                except OSError:pass


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--socket',required=True)
    p.add_argument('--root',required=True)
    p.add_argument('--ready',required=True)
    a=p.parse_args()
    serve(a.socket,a.root,a.ready)
