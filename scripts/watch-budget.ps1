param([Parameter(Mandatory=$true)][string]$PodId,
      [Parameter(Mandatory=$true)][string]$DeadlineUtc,
      [Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
$cli = 'C:\Users\msjgr\AppData\Local\Programs\runpodctl\runpodctl.exe'
$deadline = [DateTime]::Parse($DeadlineUtc).ToUniversalTime()
while ([DateTime]::UtcNow -lt $deadline) {
    Start-Sleep -Seconds 30
    if (Test-Path -LiteralPath (Join-Path $OutputDirectory 'terminated.json')) { exit 0 }
}
while ($true) {
    & $cli pod delete $PodId 2>&1 | Out-File -Append (Join-Path $OutputDirectory 'watchdog.log')
    $pods = & $cli pod list | ConvertFrom-Json
    if ($LASTEXITCODE -eq 0 -and -not ($pods | Where-Object id -eq $PodId)) {
        @{pod=$PodId; terminatedAt=[DateTime]::UtcNow.ToString('o'); reason='budget deadline'} |
            ConvertTo-Json | Set-Content (Join-Path $OutputDirectory 'terminated.json')
        exit 0
    }
    Start-Sleep -Seconds 30
}
