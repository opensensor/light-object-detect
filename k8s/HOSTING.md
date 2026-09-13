# Local inference with cloud recovery

The normal deployment serves inference from the OpenSensor Threadripper. Camera
recordings, LightNVR instances and provisioning remain in DigitalOcean.

`light-object-detect.light-object-detect.svc:8000` and `detect.lightnvr.com` keep
their existing addresses. The Service selects the cloud hosting gateway, which
uses the local origin through a dedicated authenticated tunnel. The original
application is retained behind `object-detection-cloud-origin`, normally with
zero replicas. Its existing 5 GiB HuggingFace cache PVC is retained.

The controller starts one cloud application replica when the local origin fails
two consecutive checks. After local health is stable for 90 seconds it returns
cloud capacity to zero. Image pulls, volume attachment and autoscaler node
startup add to recovery time. In-memory tracking IDs reset when an application
origin changes, as they do on an ordinary application restart; recordings are
independent of this process.

The application image is pinned by digest. The hosting image uses the previously
running application/dependency image and changes only ONNX session threading:
two inference threads, one inter-operation thread, and no idle thread spinning.
`.github/workflows/hosting-runtime.yml` builds it and runs a real model inference
in GitHub CI. No local image build or package installation is required.

The dedicated gateway, tunnel, model-readiness launcher and recovery controller
are maintained in `storefront/k8s/local-hosting/object-detection`. Apply those
resources and their private tunnel credentials before applying these manifests.
The launcher ConfigMap is included here so reapplying the recovery Deployment
does not discard its model health check. Do not repoint the Service directly to
the cloud application while its replica count is zero.
