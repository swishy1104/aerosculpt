# Stops the public site: the launcher (public_site.py or public_link.py), the app and worker it
# started, and the Cloudflare tunnel. Other Python processes (a dev server, notebooks) are left alone.
$all = Get-CimInstance Win32_Process
$alive = @{}
foreach ($p in $all) { $alive[[int]$p.ProcessId] = $p }
$ids = New-Object System.Collections.Generic.List[int]

function Add-Tree($root) {
    if ($ids.Contains([int]$root)) { return }
    $ids.Add([int]$root)
    foreach ($c in ($all | Where-Object { $_.ParentProcessId -eq $root })) { Add-Tree $c.ProcessId }
}

foreach ($r in ($all | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'public_site\.py|public_link\.py' })) { Add-Tree $r.ProcessId }
# an app whose launcher already died (its parent is gone) belongs to the site too
foreach ($o in ($all | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'run\.py serve' -and -not $alive.ContainsKey([int]$_.ParentProcessId) })) { Add-Tree $o.ProcessId }
foreach ($t in ($all | Where-Object { $_.Name -eq 'cloudflared.exe' })) { Add-Tree $t.ProcessId }

foreach ($id in $ids) {
    $p = $alive[$id]
    Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
    Write-Host ("stopped " + $p.Name + " " + $id)
}
if ($ids.Count -eq 0) { Write-Host "The public site was not running." } else { Write-Host "The public site is stopped." }
