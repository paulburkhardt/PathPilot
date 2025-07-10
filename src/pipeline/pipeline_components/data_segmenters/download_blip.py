from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Salesforce/blip-base",
    local_dir="/usr/prakt/s0120/PathPilot/blip-base",
    local_dir_use_symlinks=False
)