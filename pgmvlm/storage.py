"""Optional personal Lustre quota checks; never deletes checkpoints."""
import os
import json
import re
import socket
import subprocess
import time
from pathlib import Path
GIB = 1024**3

class StorageHeadroomError(RuntimeError):
    pass

def parse_lustre_quota(output, mount):
    """Parse non-human-readable lfs quota rows, including wrapped mount names."""
    lines=output.splitlines()
    for index,line in enumerate(lines):
        parts=line.split()
        if not parts or parts[0]!=mount:
            continue
        values=parts[1:]
        if len(values)<8 and index+1<len(lines):
            values+=lines[index+1].split()
        if len(values)!=8:
            raise StorageHeadroomError('Unrecognized personal quota row')
        def number(value):
            if not re.fullmatch(r'\d+\*?',value):
                raise StorageHeadroomError('Quota must be reported in raw KiB/counts')
            return int(value.rstrip('*'))
        used,soft,hard=map(number,values[:3])
        files,isoft,ihard=map(number,values[4:7])
        # Conservatively honor soft limits too, independent of any grace period.
        limits=[x for x in (soft,hard) if x>0]
        ilimits=[x for x in (isoft,ihard) if x>0]
        if not limits or not ilimits:
            raise StorageHeadroomError('Cannot establish finite personal storage and file limits')
        return {'used_bytes':used*1024,'limit_bytes':min(limits)*1024,
                'free_bytes':max(0,min(limits)-used)*1024,'used_files':files,
                'limit_files':min(ilimits),'free_files':max(0,min(ilimits)-files)}
    raise StorageHeadroomError('No quota row for the checkpoint filesystem')

def _native_storage_report(path):
    path=Path(path).resolve()
    while not path.exists():
        path=path.parent
    fs=subprocess.run(['df','-PT',str(path)],check=True,capture_output=True,text=True,timeout=20)
    fields=fs.stdout.splitlines()[-1].split()
    if len(fields)<7 or fields[1]!='lustre':
        raise StorageHeadroomError('Expected verified Lustre checkpoint filesystem')
    mount=fields[-1]
    quota=subprocess.run(['lfs','quota','-u',str(os.getuid()),mount],check=True,
                         capture_output=True,text=True,timeout=20)
    report=parse_lustre_quota(quota.stdout,mount)
    stats=os.statvfs(path)
    report.update(path=str(path),mount=mount,uid=os.getuid(),
                  filesystem_free_bytes=stats.f_bavail*stats.f_frsize,
                  filesystem_free_files=stats.f_favail,checked_unix=time.time())
    return report

def storage_report(path):
    """Use native tools, or the same user's live host service inside a container."""
    endpoint=os.environ.get('PGMVLM_STORAGE_SOCKET')
    if not endpoint:
        return _native_storage_report(path)
    path=Path(path).resolve()
    while not path.exists():
        path=path.parent
    job=os.environ.get('SLURM_JOB_ID','')
    request=json.dumps({'path':str(path),'job_id':job}).encode()+b'\n'
    try:
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
            client.settimeout(65)
            client.connect(endpoint);client.sendall(request)
            with client.makefile('rb') as stream:
                raw=stream.readline(65537)
        if len(raw)>65536 or not raw.endswith(b'\n'):
            raise ValueError('Invalid quota response')
        data=json.loads(raw)
        if 'error' in data:
            raise ValueError(data['error'])
        if data['uid']!=os.getuid() or data['job_id']!=job or data['path']!=str(path):
            raise ValueError('Quota response identity/path mismatch')
        age=time.time()-data['checked_unix']
        if not -5<=age<=60:
            raise ValueError('Stale quota response')
        for key in ('free_bytes','free_files','filesystem_free_bytes','filesystem_free_files','metadata_free_files'):
            if not isinstance(data[key],int) or isinstance(data[key],bool) or data[key]<0:
                raise ValueError('Invalid quota counts')
        return data
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise StorageHeadroomError(f'Host quota check failed: {exc}') from exc

def require_headroom(path, reserve_bytes=8*GIB, reserve_files=2000):
    report=storage_report(path)
    if report['free_bytes']<reserve_bytes or report['filesystem_free_bytes']<reserve_bytes:
        raise StorageHeadroomError(f"Checkpoint storage headroom too low: personal {report['free_bytes']/GIB:.3f} GiB; require {reserve_bytes/GIB:.3f} GiB")
    if report['free_files']<reserve_files or (report['filesystem_free_files']>0 and report['filesystem_free_files']<reserve_files):
        raise StorageHeadroomError('Checkpoint file-count quota headroom too low')
    if report.get('metadata_free_files',reserve_files)<reserve_files:
        raise StorageHeadroomError('Checkpoint metadata-target quota headroom too low')
    return report
