import copy
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import hosting_release as agent
import hosting_runtime as runtime
from hosting_release_manifest import platform_digest

IMAGE = 'ghcr.io/opensensor/light-object-detect@sha256:'+'a'*64
TARGET = dict(format='detector-release-v1',repository=agent.REPO,commit='b'*40,
              run_id=100,run_attempt=1,image=IMAGE)
WORKFLOW = dict(event='push',head_branch='main',path=agent.WORKFLOW,
                repository={'full_name':agent.REPO},head_sha='b'*40,id=100,run_attempt=1)


class ReleaseTests(unittest.TestCase):
    def test_health_probe_retries_transient_502_but_not_configuration_errors(self):
        with patch.object(runtime.time,'sleep'), patch.object(runtime.time,'monotonic',side_effect=[0,1,2]):
            probe=unittest.mock.Mock(side_effect=[urllib.error.HTTPError('https://example.invalid',502,'gateway',{},None),True])
            runtime.wait('Recovered health probe',probe,10)
            self.assertEqual(probe.call_count,2)
        with self.assertRaises(ValueError):
            runtime.wait('Invalid deployment',unittest.mock.Mock(side_effect=ValueError('configuration changed')),10)

    def test_artifact_must_belong_to_exact_main_run(self):
        self.assertEqual(agent.validate(TARGET,WORKFLOW),TARGET)
        for key,value in [('run_id',99),('run_attempt',2),('commit','c'*40),
                          ('image','ghcr.io/opensensor/light-object-detect:latest')]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                agent.validate({**TARGET,key:value},WORKFLOW)
        for key,value in [('event','pull_request'),('head_branch','feature')]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                agent.validate(TARGET,{**WORKFLOW,key:value})

    def test_interrupted_handoff_finishes_before_next_release(self):
        newer={**TARGET,'run_id':101}
        self.assertEqual(agent.choose(TARGET,newer,{}),TARGET)
        self.assertEqual(agent.choose({},newer,{**TARGET,'state':'success'}),newer)
        self.assertIsNone(agent.choose({},TARGET,{**TARGET,'state':'success'}))
        with self.assertRaises(RuntimeError):agent.choose({},TARGET,{**newer,'state':'success'})

    def test_image_selection_excludes_attestation(self):
        manifest={'manifests':[
            {'digest':'sha256:attestation','platform':{'architecture':'unknown','os':'unknown'}},
            {'digest':'sha256:runtime','platform':{'architecture':'amd64','os':'linux'}}]}
        self.assertEqual(platform_digest(manifest,'sha256:index'),'sha256:runtime')
        manifest['manifests'].append(manifest['manifests'][1])
        with self.assertRaises(ValueError):platform_digest(manifest,'sha256:index')

    def test_ready_rejects_old_binary_and_pending_or_terminating_pods(self):
        deployment={'metadata':{'generation':3},'spec':{'replicas':1,
            'template':{'spec':{'containers':[{'image':IMAGE}]}},
            'selector':{'matchLabels':{'app':'detector'}}},
            'status':{'observedGeneration':3,'replicas':1,'updatedReplicas':1,'readyReplicas':1}}
        pod={'metadata':{},'status':{'containerStatuses':[{'ready':True,'imageID':'docker-pullable://'+IMAGE}]}}
        def ready(pods):
            with patch.object(runtime,'get',side_effect=[deployment,{'items':pods}]):
                return runtime.ready('local',IMAGE)
        self.assertTrue(ready([pod]))
        old=copy.deepcopy(pod);old['status']['containerStatuses'][0]['imageID']='old'
        self.assertFalse(ready([old]))
        self.assertFalse(ready([{'metadata':{},'status':{}}]))
        terminating=copy.deepcopy(pod);terminating['metadata']['deletionTimestamp']='now'
        self.assertFalse(ready([pod,terminating]))

    def test_resumed_release_does_not_overwrite_operator_hold(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            runtime.save_progress(path/'progress.json',{'target':TARGET,'phase':5,
                'baseline':{},'placement_revision':'previous'})
            with patch.object(runtime,'placement',return_value={'data':{
                    'revision':'operator','release_hold':'operator','desired':'cloud'}}), \
                    patch.object(runtime,'run') as mutate:
                with self.assertRaisesRegex(RuntimeError,'Another operator'):
                    runtime.deploy(TARGET,path)
                mutate.assert_not_called()

    def test_completed_resume_checks_runtime_without_restarting_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            runtime.save_progress(path/'progress.json',{'target':TARGET,'phase':7,
                'baseline':{},'placement_revision':'previous'})
            with patch.object(runtime,'ready',return_value=True), \
                    patch.object(runtime,'public',return_value='local'), \
                    patch.object(runtime,'get',return_value={'spec':{'replicas':0,
                      'template':{'spec':{'containers':[{'image':IMAGE}]}}}}), \
                    patch.object(runtime,'run') as mutate:
                self.assertTrue(runtime.deploy(TARGET,path)['persistent_cache_verified'])
                mutate.assert_not_called()


if __name__=='__main__':unittest.main()
