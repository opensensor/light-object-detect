# Automatic paired detector deployment

Every main push builds the hosting image and publishes a receipt containing its
source revision, workflow run and immutable amd64 image digest. The operator host
polls for successful builds every minute using its existing GitHub and cluster
access. CI waits for a status specific to that build and fails if deployment is
not verified. Pull requests test coordination without deploying.

The agent warms cloud, moves traffic there, updates cloud and local, updates the
recovery controller's expected image, returns traffic locally, and verifies the
cloud persistent model cache before scaling cloud back to zero. Checkpoints make
interrupted handoffs resumable. A new build waits for an active handoff; an older
build cannot replace a newer completed release. Operator holds are preserved.

Install once on the existing operator host (no new credentials):

```sh
install -Dm755 scripts/deploy/detector-release-launch.sh ~/.local/lib/opensensor-hosting/detector-release-launch.sh
install -Dm644 scripts/deploy/opensensor-detector-release.service ~/.config/systemd/user/opensensor-detector-release.service
install -Dm644 scripts/deploy/opensensor-detector-release.timer ~/.config/systemd/user/opensensor-detector-release.timer
systemctl --user daemon-reload
systemctl --user enable --now opensensor-detector-release.timer
```

The launcher follows main. Receipts and resumable state are private files under
`~/.config/opensensor-hosting/detector-release/`. The progress snapshot includes
existing deployment configuration; keep that directory private. Review failures
with `journalctl --user -u opensensor-detector-release.service`. The timer retries
a failed handoff; do not delete pending state during a deployment.
