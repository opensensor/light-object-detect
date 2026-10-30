#!/usr/bin/env python3
"""Consume the latest successful main image build and resume paired deployments."""
import argparse
import fcntl
import io
import json
import os
from pathlib import Path
import re
import subprocess
import time
import zipfile

from hosting_runtime import deploy, save_progress

REPO = 'opensensor/light-object-detect'
WORKFLOW = '.github/workflows/hosting-runtime.yml'
CONTEXT = 'hosting/detector-release'
STATE = Path.home() / '.config/opensensor-hosting/detector-release'
IMAGE = re.compile(r'ghcr\.io/opensensor/light-object-detect@sha256:[a-f0-9]{64}')


def gh(path, **fields):
    args = ['gh', 'api', '-H', 'Cache-Control: no-cache', path]
    for key, value in fields.items():args += ['-f', key + '=' + str(value)]
    return json.loads(subprocess.check_output(args, timeout=90))


def validate(target, workflow):
    if (workflow['event'] != 'push' or workflow['head_branch'] != 'main'
            or workflow['path'] != WORKFLOW or workflow['repository']['full_name'] != REPO
            or set(target) != {'format', 'repository', 'commit', 'run_id', 'run_attempt', 'image'}
            or target['format'] != 'detector-release-v1' or target['repository'] != REPO
            or target['commit'] != workflow['head_sha'] or target['run_id'] != workflow['id']
            or target['run_attempt'] != workflow['run_attempt']
            or not re.fullmatch('[a-f0-9]{40}', target['commit']) or not IMAGE.fullmatch(target['image'])):
        raise ValueError('Release must identify this main build and its immutable image')
    return target


def latest():
    runs = gh(f'repos/{REPO}/actions/workflows/hosting-runtime.yml/runs?branch=main&event=push&per_page=20')['workflow_runs']
    for workflow in sorted(runs, key=lambda r:(r['run_number'], r['run_attempt']), reverse=True):
        jobs = gh(f'repos/{REPO}/actions/runs/{workflow["id"]}/attempts/{workflow["run_attempt"]}/jobs?per_page=100')['jobs']
        if not any(j['name']=='runtime' and j['conclusion']=='success' for j in jobs):continue
        artifacts = gh(f'repos/{REPO}/actions/runs/{workflow["id"]}/artifacts?per_page=100')['artifacts']
        name = f'hosting-release-{workflow["head_sha"]}-{workflow["id"]}-{workflow["run_attempt"]}'
        matches = [a for a in artifacts if a['name']==name and not a['expired']]
        if len(matches)!=1:raise RuntimeError('Successful build is missing its release artifact')
        raw = subprocess.check_output(['gh','api',f'repos/{REPO}/actions/artifacts/{matches[0]["id"]}/zip'],timeout=90)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            if archive.namelist()!=['hosting-release.json'] or archive.infolist()[0].file_size>8192:
                raise ValueError('Unexpected release artifact')
            return validate(json.loads(archive.read('hosting-release.json')),workflow)
    return None


def report(target, state, description):
    gh(f'repos/{REPO}/statuses/{target["commit"]}',state=state,
       context=f'{CONTEXT}/{target["run_id"]}/{target["run_attempt"]}',description=description[:140],
       target_url=f'https://github.com/{REPO}/actions/runs/{target["run_id"]}')


def choose(pending, candidate, previous):
    # Finish a handoff already in progress before starting another release.
    if pending:return pending
    if not candidate:return None
    if previous:
        before=(previous['run_id'],previous['run_attempt'])
        after=(candidate['run_id'],candidate['run_attempt'])
        if after<before:raise RuntimeError('Refuse an older release')
        if after==before and previous.get('state')=='success':return None
    return candidate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    os.umask(0o077);STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/'agent.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        read=lambda name:json.loads((STATE/name).read_text()) if (STATE/name).exists() else {}
        pending, previous=read('pending.json'),read('latest.json')
        target=choose(pending,None if pending else latest(),previous)
        if not target:return
        if not args.apply:print(json.dumps(target));return
        save_progress(STATE/'pending.json',target)
        directory=STATE/f'{target["run_id"]}-{target["run_attempt"]}'
        directory.mkdir(exist_ok=True)
        report(target,'pending','Automatically deploying this build to both detector origins')
        try:
            result=deploy(target,directory)
            receipt={**target,**result,'state':'success','updated_at':time.time()}
            report(target,'success','Exact image serving locally; cloud recovery image and persistent cache verified')
            save_progress(STATE/'latest.json',receipt)
            (STATE/'pending.json').unlink()
        except Exception as error:
            save_progress(STATE/'latest.json',{**target,'state':'failure','error_type':type(error).__name__,'updated_at':time.time()})
            report(target,'failure','Automatic deployment will retry: '+type(error).__name__)
            raise


if __name__=='__main__':main()
