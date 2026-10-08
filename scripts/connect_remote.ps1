$ErrorActionPreference = 'Stop'

# Requires the local VPN and the existing Text2Face entry in ~/.ssh/config.
$ssh = (Get-Command ssh.exe -ErrorAction Stop).Source
& $ssh -tt Text2Face 'cd /data/MCAW && exec tmux attach-session -t mcaw'

if ($LASTEXITCODE -ne 0) {
    throw "Remote connection ended with exit code $LASTEXITCODE"
}
