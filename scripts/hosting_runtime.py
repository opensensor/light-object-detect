"""Paired release with a warmed cloud origin and separate temporary model cache."""
import copy
import json
from pathlib import Path
import subprocess
import time
import urllib.request

import shlex

NS = 'light-object-detect'
CLOUD = 'do-nyc1-k8s-1-33-1-do-1-nyc1-1752275240114'

def command(site, args):
    if site == 'cloud':
        return ['kubectl','--context',CLOUD,'--request-timeout=20s',*args]
    remote = ['/srv/ai/opensensor-hosting/home/bin/kubectl','--kubeconfig',
              '/srv/ai/opensensor-hosting/config/kubeconfig','--context','kind-opensensor-local',
              '--request-timeout=20s',*args]
    return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',
            'opensensor-hosting@192.168.53.187',shlex.join(remote)]

def get(site, *args):
    return json.loads(run(site,[*args,'-o','json']))


def run(site, args, body=None):
    result = subprocess.run(command(site, args), input=body, text=True, capture_output=True, timeout=620)
    if result.returncode:raise RuntimeError('Cluster operation failed: ' + result.stderr[:400])
    return result.stdout


def patch(site, kind, name, operations):
    obj = get(site, 'get', kind, name, '-n', NS)
    return run(site, ['patch', kind, name, '-n', NS, '--type=json', '--patch-file=/dev/stdin'],
               json.dumps([{'op': 'test', 'path': '/metadata/resourceVersion', 'value': obj['metadata']['resourceVersion']}] + operations))


def placement():return get('cloud', 'get', 'cm', 'object-detection-placement', '-n', NS)


def owned(revision):
    data = placement()['data']
    if data.get('revision') != revision or data.get('release_hold') != revision or data.get('desired') != 'cloud':
        raise RuntimeError('Detector release hold changed')


def wait(label, function, seconds=600):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if function():print(label, flush=True);return
        time.sleep(3)
    raise RuntimeError('Timed out: ' + label)


def ready(site, image):
    d = get(site, 'get', 'deploy', NS, '-n', NS)
    status = d['status']
    if d['spec']['replicas'] != 1 or d['spec']['template']['spec']['containers'][0]['image'] != image:return False
    if status.get('observedGeneration', 0) < d['metadata']['generation']:return False
    if any(status.get(k, 0) != 1 for k in ('replicas', 'updatedReplicas', 'readyReplicas')):return False
    labels = ','.join(k + '=' + v for k, v in d['spec']['selector']['matchLabels'].items())
    pods = get(site, 'get', 'pods', '-n', NS, '-l', labels)['items']
    if len(pods) != 1 or pods[0]['metadata'].get('deletionTimestamp'):return False
    statuses = pods[0].get('status', {}).get('containerStatuses', [])
    return (len(statuses) == 1 and statuses[0].get('ready') is True
            and statuses[0].get('imageID', '').endswith(image.split('@')[1]))


def public():
    with urllib.request.urlopen('https://detect.lightnvr.com/__hosting/health', timeout=10) as response:
        body = json.load(response)
        assert response.status == 200 and body['status'] == 'ok'
        return response.headers.get('X-OpenSensor-Location')


def update(site, image, revision, temporary_cache=False):
    owned(revision)
    d = get(site, 'get', 'deploy', NS, '-n', NS)
    spec = copy.deepcopy(d['spec']['template']['spec'])
    c = spec['containers'][0]
    if c['image'] != image:
        c['image'] = image
        c['env'] = [e for e in c.get('env', []) if e['name'] != 'SENTRY_RELEASE']
        c['env'].append({'name':'SENTRY_RELEASE','value':'light-object-detect@'+revision.split(':')[-1]})
        changes = []
        if temporary_cache:
            spec['volumes'] = [{'name':v['name'],'emptyDir':{'sizeLimit':'5Gi'}}
                              if v['name']=='hf-cache' else v for v in spec['volumes']]
            changes.append({'op':'replace','path':'/spec/strategy','value':{
                'type':'RollingUpdate','rollingUpdate':{'maxUnavailable':0,'maxSurge':1}}})
        changes.append({'op':'replace','path':'/spec/template/spec','value':spec})
        patch('cloud' if site=='cloud' else 'local','deploy',NS,changes)
    wait(site+' exact image and model Ready',lambda:ready(site,image))


def deploy(target, directory):
    image, commit = target['image'], target['commit']
    revision = 'ci:'+str(target['run_id'])+':'+commit
    path = directory/'progress.json'
    if path.exists():state=json.loads(path.read_text())
    else:
        data=placement()['data']
        if data.get('desired')!='local' or any(data.get(k) for k in ('hold','release_hold','maintenance_hold')):
            raise RuntimeError('Existing maintenance or release intent must finish first')
        baseline=get('cloud','get','deploy',NS,'-n',NS)
        if baseline['spec']['strategy']!={'type':'Recreate'}:
            raise RuntimeError('Unexpected baseline cloud cache strategy')
        state={'target':target,'phase':0,'baseline':baseline,'placement_revision':data.get('revision','')}
        save_progress(path,state)
    if state['target']!=target:raise RuntimeError('Release state belongs to another build')
    baseline=state['baseline']
    def stage(number, action):
        if state['phase'] < number:
            action()
            state['phase']=number;save_progress(path,state)
    def intent(values):
        current=placement()['data']
        if (current.get('release_hold') not in ('',None,revision)
                or current.get('revision','') not in (state['placement_revision'],revision)
                or any(current.get(k) for k in ('hold','maintenance_hold'))):
            raise RuntimeError('Another operator owns detector placement')
        changes=[{'op':'test','path':'/data','value':current}]
        changes += [{'op':'add','path':'/data/'+key,'value':str(value)}
                    for key,value in {**values,'revision':revision}.items()]
        run('cloud',['patch','cm','object-detection-placement','-n',NS,
                     '--type=json','--patch-file=/dev/stdin'],json.dumps(changes))
    def warm():
        intent({'recovery_test_until':time.time()+1800})
        current=get('cloud','get','deploy',NS,'-n',NS)['spec']['template']['spec']['containers'][0]['image']
        wait('Cloud model warmed',lambda:ready('cloud',current))
    def hold():
        intent({'revision':revision,'release_hold':revision,'desired':'cloud'})
        wait('Cloud selected',lambda:public()=='cloud',120)
    def gateway():
        owned(revision)
        d=get('cloud','get','deploy','object-detection-gateway','-n',NS)
        i=next(i for i,c in enumerate(d['spec']['template']['spec']['containers']) if c['name']=='controller')
        env=d['spec']['template']['spec']['containers'][i].get('env',[])
        if next((e.get('value') for e in env if e['name']=='EXPECTED_IMAGE'),None)!=image:
            env=[e for e in env if e['name']!='EXPECTED_IMAGE']+[{'name':'EXPECTED_IMAGE','value':image}]
            patch('cloud','deploy','object-detection-gateway',[{'op':'add','path':f'/spec/template/spec/containers/{i}/env','value':env}])
        run('cloud',['rollout','status','deploy/object-detection-gateway','-n',NS,'--timeout=600s'])
    def local():
        intent({'release_hold':'','desired':'local','recovery_test_until':'0'})
        wait('Local selected and cloud drained',lambda:public()=='local' and get('cloud','get','deploy',NS,'-n',NS)['spec']['replicas']==0)
    def cache():
        intent({})
        d=get('cloud','get','deploy',NS,'-n',NS)
        volumes=d['spec']['template']['spec']['volumes']
        desired=baseline['spec']['template']['spec']['volumes']
        if volumes!=desired or d['spec']['strategy']!=baseline['spec']['strategy']:
            # Status heartbeats must not conflict with this spec-only CAS.
            operations=[{'op':'test','path':'/spec/replicas','value':0},
                {'op':'test','path':'/spec/template/spec/containers/0/image','value':image},
                {'op':'test','path':'/spec/template/spec/volumes','value':volumes},
                {'op':'replace','path':'/spec/strategy','value':baseline['spec']['strategy']},
                {'op':'replace','path':'/spec/template/spec/volumes','value':desired}]
            run('cloud',['patch','deploy',NS,'-n',NS,'--type=json','--patch-file=/dev/stdin'],json.dumps(operations))
        intent({'recovery_test_until':time.time()+600})
        wait('Persistent cloud model Ready',lambda:ready('cloud',image))
        intent({'recovery_test_until':'0'})
        wait('Cloud recovery idle',lambda:get('cloud','get','deploy',NS,'-n',NS)['spec']['replicas']==0)
    stage(1,warm);stage(2,hold)
    stage(3,lambda:update('cloud',image,revision,True))
    stage(4,lambda:update('local',image,revision))
    stage(5,gateway);stage(6,local);stage(7,cache)
    if not ready('local',image) or public()!='local':raise RuntimeError('Latest local detector not serving')
    d=get('cloud','get','deploy',NS,'-n',NS)
    if d['spec']['template']['spec']['containers'][0]['image']!=image or d['spec']['replicas']!=0:
        raise RuntimeError('Cloud recovery template differs from this build')
    return {'local_ready':True,'cloud_replicas':0,'persistent_cache_verified':True}


def save_progress(path,state):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(state));temp.replace(path)
