#!/usr/bin/env python3
"""Keep CI pending until this build is verified on both production origins."""
import json
import os
import subprocess
import time


def main():
    repo, commit = os.environ['GITHUB_REPOSITORY'], os.environ['GITHUB_SHA']
    context = f'hosting/detector-release/{os.environ["GITHUB_RUN_ID"]}/{os.environ["GITHUB_RUN_ATTEMPT"]}'
    deadline = time.monotonic() + 2100
    while time.monotonic() < deadline:
        raw = subprocess.check_output(['gh','api','--paginate','--slurp','-H','Cache-Control: no-cache',
            f'repos/{repo}/commits/{commit}/statuses?per_page=100'],timeout=60)
        statuses = [s for page in json.loads(raw) for s in page if s['context']==context]
        if statuses:
            status = max(statuses,key=lambda s:s['id'])
            print(status['state']+': '+status['description'],flush=True)
            if status['state']=='success':return
            # The host resumes failed handoffs automatically; keep this run open for its retry.
        else:print('Waiting for the production release agent',flush=True)
        time.sleep(20)
    raise SystemExit('This build has not been verified in production; inspect hosting/detector-release')


if __name__=='__main__':main()
