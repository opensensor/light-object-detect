#!/usr/bin/env python3
"""Bind a release to this build's amd64 manifest, excluding provenance attestations."""
import json
import os
import subprocess


def platform_digest(document, digest):
    if 'manifests' not in document:return digest
    matches=[m['digest'] for m in document['manifests'] if m.get('platform')=={'architecture':'amd64','os':'linux'}]
    if len(matches)!=1:raise ValueError('Expected exactly one linux/amd64 image')
    return matches[0]


def main():
    digest=os.environ['BUILD_DIGEST']
    repository='ghcr.io/opensensor/light-object-detect'
    document=json.loads(subprocess.check_output(['docker','buildx','imagetools','inspect',repository+'@'+digest,'--raw']))
    target={'format':'detector-release-v1','repository':os.environ['GITHUB_REPOSITORY'],
        'commit':os.environ['GITHUB_SHA'],'run_id':int(os.environ['GITHUB_RUN_ID']),
        'run_attempt':int(os.environ['GITHUB_RUN_ATTEMPT']),
        'image':repository+'@'+platform_digest(document,digest)}
    with open('hosting-release.json','w') as out:json.dump(target,out)


if __name__=='__main__':main()
