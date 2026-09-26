# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：ZCode；版本/型号：GLM-5.3（会话标识 AI-06）；开发机构：智谱（Z.ai）；版本发布日期：待补
# 用途：T023 台账一致性检查（D033）——五对机械查对，输出差异清单交主窗口处理：
#   [A] 02 <-> 04/05：RES 引用是否已登记、状态词是否合规、编号是否重复
#   [B] 05 <-> figure_registry.csv：图表/表格 ID 双轨（D029）
#   [C] 03 <-> 正文 tex：章节状态声明 vs \pending/TODO 待补标记实况
#   [D] 09 <-> D030 联动规则：A 条目状态与人工核验记录矛盾
#   [E] 06/D020 禁用术语 <-> 正文 tex（清单见 -BannedTerms，默认仅"处理核"）
# 用法：powershell -NoProfile -ExecutionPolicy Bypass -File scripts\check_ledger_consistency.ps1
#   仅报告不阻断（发现项标【差】须处理、【提示】人工判断）；T020 数字三方核对待数值入文后并入本框架。
[CmdletBinding()]
param(
    # 禁用术语清单来源：D020 及后续决策；06 术语表如增设机器可读禁用行，可扩展此处
    [string[]]$BannedTerms = @('处理核')
)
$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

$area   = Split-Path -Parent $PSScriptRoot
$notes  = Join-Path $area 'notes'
$figs   = Join-Path $area 'figs'
$script:diffs  = New-Object System.Collections.Generic.List[string]
$script:infos  = New-Object System.Collections.Generic.List[string]
function Add-Diff($s) { $script:diffs.Add($s) }
function Add-Info($s) { $script:infos.Add($s) }
function Read-Text($p) {
    if (-not (Test-Path -LiteralPath $p)) { throw "找不到 $p" }
    return (Get-Content -LiteralPath $p -Raw -Encoding UTF8)
}
function Split-Lines($t) { return @($t -split '\r?\n') }

# ---------- [A] 02 <-> 04/05 ----------
$t02 = Read-Text (Join-Path $notes '02_confirmed_results.md')
$resStatus = @{}; $resDup = @(); $resShared = @{}
foreach ($ln in (Split-Lines $t02)) {
    if ($ln -match '^\|\s*(RES-\d{3})\s*\|') {
        $id = $Matches[1]
        # 含"共用记录见"的行是跨问共用记录（如 RES-012）在多问索引表中的合法重复引用，不计差
        if ($resStatus.ContainsKey($id) -and ($ln -notmatch '共用记录见')) { $resDup += $id }
        $cells = @(($ln -split '\|') | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' })
        $resStatus[$id] = [string]$cells[-1]
        if ($ln -match '共用记录见') { $resShared[$id] = $true }
    }
}
# §1.8 等节以 bullet（- **RES-xxx**）登记的共用/口径记录：计入"已登记"，状态不在此解析
foreach ($ln in (Split-Lines $t02)) {
    if ($ln -match '^\s*-\s*\*\*(RES-\d{3})[^*]*\*\*') {
        $id = $Matches[1]
        if (-not $resStatus.ContainsKey($id)) { $resStatus[$id] = '(bullet登记,状态未解析)'; $resShared[$id] = $true }
    }
}
$validStatus = @('CONFIRMED', 'PRELIMINARY', 'DEPRECATED', 'UNKNOWN')
foreach ($k in @($resStatus.Keys)) {
    if ($validStatus -notcontains $resStatus[$k] -and -not $resShared.ContainsKey($k)) {
        Add-Diff ("[A][差] 02 的 {0} 状态列「{1}」不在四种标准状态内" -f $k, $resStatus[$k])
    }
}
if ($resDup.Count) { Add-Diff ('[A][差] 02 存在重复结果编号：{0}' -f (($resDup | Sort-Object -Unique) -join '、')) }

$t04 = Read-Text (Join-Path $notes '04_claim_evidence.md')
$refs04 = @([regex]::Matches($t04, 'RES-\d{3}') | ForEach-Object { $_.Value } | Sort-Object -Unique)
foreach ($r in $refs04) {
    if (-not $resStatus.ContainsKey($r)) { Add-Diff ("[A][差] 04 引用了 02 未登记的 {0}（论断证据必须引已登记结果）" -f $r) }
}
$claimIds = @(); $claimDup = @()
foreach ($ln in (Split-Lines $t04)) {
    if ($ln -match '^\|\s*(C\d{2})\s*\|') {
        if ($claimIds -contains $Matches[1]) { $claimDup += $Matches[1] } else { $claimIds += $Matches[1] }
    }
}
if ($claimDup.Count) { Add-Diff ('[A][差] 04 存在重复论断编号：{0}' -f (($claimDup | Sort-Object -Unique) -join '、')) }

$t05 = Read-Text (Join-Path $notes '05_figure_plan.md')
$refs05 = @([regex]::Matches($t05, 'RES-\d{3}') | ForEach-Object { $_.Value } | Sort-Object -Unique)
foreach ($r in $refs05) {
    if (-not $resStatus.ContainsKey($r)) { Add-Info ("[A][提示] 05 出现 02 未登记的 {0}（若为计划占位，登记后消除）" -f $r) }
}
$confirmedUnused = @($resStatus.Keys | Where-Object { $resStatus[$_] -eq 'CONFIRMED' -and $refs04 -notcontains $_ })
foreach ($c in $confirmedUnused) { Add-Info ("[A][提示] {0} 已 CONFIRMED 但未被 04 任何论断引用" -f $c) }

# ---------- [B] 05 <-> figure_registry ----------
$ids05 = @([regex]::Matches($t05, '(?m)^\|\s*((?:F|R)\d{2}|TBL-\d{3})\s*\|') | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique)
$reg = @(Import-Csv -LiteralPath (Join-Path $figs 'figure_registry.csv') -Encoding UTF8)
$regIds = @($reg | ForEach-Object { $_.figure_id })
$regDup = @($regIds | Group-Object | Where-Object { $_.Count -gt 1 } | ForEach-Object { $_.Name })
if ($regDup.Count) { Add-Diff ('[B][差] figure_registry 存在重复 ID：{0}' -f ($regDup -join '、')) }
foreach ($i in $ids05) { if ($regIds -notcontains $i) { Add-Diff ("[B][差] 05 已分配 {0} 但未登记 figure_registry（违反 D029 即分配即登记）" -f $i) } }
foreach ($i in $regIds) { if ($ids05 -notcontains $i) { Add-Info ("[B][提示] {0} 已登记 registry 但 05 无对应计划行（孤儿登记，确认是否废弃）" -f $i) } }

# ---------- [C] 03 <-> 正文待补标记 ----------
$texFiles = @(Get-ChildItem -LiteralPath (Join-Path $area 'paper') -Recurse -Filter '*.tex' |
    Where-Object { $_.FullName -notmatch '\\build\\' })
function Count-Markers($p) {
    $t = Get-Content -LiteralPath $p -Raw -Encoding UTF8
    $n = [regex]::Matches($t, '\\pending').Count + [regex]::Matches($t, 'TODO').Count
    return $n
}
$lines03 = Split-Lines (Read-Text (Join-Path $notes '03_outline.md'))
foreach ($f in ($texFiles | Where-Object { $_.Directory.Name -eq 'sections' -and $_.BaseName -match '^0[1-7]_' })) {
    $ch = $f.BaseName.Substring(1, 1)
    $row = $null
    foreach ($ln in $lines03) { if ($ln -match ('^\|\s*' + $ch + '\s')) { $row = $ln; break } }
    if (-not $row) { Add-Info ("[C][提示] 03 缺少第 {0} 章状态行（文件 {1}）" -f $ch, $f.Name); continue }
    $parts = @((($row -split '\|') | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }))
    $status = [string]$parts[-1]
    $markers = Count-Markers $f.FullName
    $claimedDone = ($status -notmatch 'TODO|待写|骨架|待补|待人工')
    if ($claimedDone -and $markers -gt 0) {
        Add-Diff ("[C][差] 03 称第 {0} 章已完成（「{1}」），但 {2} 仍有 {3} 处待补标记" -f $ch, $status, $f.Name, $markers)
    }
    elseif (-not $claimedDone -and $markers -eq 0) {
        Add-Info ("[C][提示] 03 称第 {0} 章待写（「{1}」），但 {2} 已无待补标记——若正文确已写完，请更新 03" -f $ch, $status, $f.Name)
    }
}

# ---------- [D] 09 联动与编号 ----------
$lines09 = Split-Lines (Read-Text (Join-Path $notes '09_ai_usage_log.md'))
$aSeen = @(); $aDup = @()
$lines08 = Split-Lines (Read-Text (Join-Path $notes '08_decision_log.md'))
$dSeen = @(); $dDup = @()
foreach ($ln in $lines08) {
    if ($ln -match '^\|\s*(D\d{3})\s*\|') {
        if ($dSeen -contains $Matches[1]) { $dDup += $Matches[1] } else { $dSeen += $Matches[1] }
    }
}
if ($dDup.Count) { Add-Diff ('[D][差] 08 存在重复决策编号：{0}' -f (($dDup | Sort-Object -Unique) -join '、')) }
$validA = @('待核验', '已核验', '已入文', '已否决')
foreach ($ln in $lines09) {
    if ($ln -match '^\|\s*(A\d{3})\s*\|') {
        $id = $Matches[1]
        if ($aSeen -contains $id) { $aDup += $id; continue }
        $aSeen += $id
        $cells = @($ln -split '\|')
        if ($cells.Count -lt 11) { Add-Info ("[D][提示] {0} 行列数异常，跳过联动检查" -f $id); continue }
        $verify = [string]$cells[7]
        $status = (([string]$cells[10]).Trim() -split '（')[0].Trim()
        if ($validA -notcontains $status) { Add-Diff ("[D][差] {0} 状态「{1}」不在四种标准状态内" -f $id, $status) }
        if ($status -eq '待核验' -and $verify -match '逐句审定') {
            Add-Diff ("[D][差] {0} 人工核验栏已记用户逐句审定，但状态仍为待核验（D030 联动缺失）" -f $id)
        }
        if ($status -eq '已入文' -and $verify -notmatch '审定|核验|确认|否决') {
            Add-Diff ("[D][差] {0} 状态为已入文，但人工核验栏无任何人工核验记录" -f $id)
        }
    }
}
if ($aDup.Count) { Add-Diff ('[D][差] 09 存在重复 AI 记录编号：{0}' -f (($aDup | Sort-Object -Unique) -join '、')) }

# ---------- [E] 禁用术语 ----------
foreach ($term in $BannedTerms) {
    if ([string]::IsNullOrWhiteSpace($term)) { continue }
    $hits = @($texFiles | Select-String -Pattern ([regex]::Escape($term)) -SimpleMatch:$false)
    foreach ($h in $hits) { Add-Diff ("[E][差] 正文 {0}:{1} 出现禁用术语「{2}」（D020/06 术语表）" -f $h.Filename, $h.LineNumber, $term) }
}

# ---------- 汇总 ----------
Write-Output '==== T023 台账一致性检查（D033，详见 scripts/check_ledger_consistency.ps1 头注）===='
Write-Output ("检查范围：02<->04/05、05<->registry、03<->正文、09 联动、禁用术语；正文 tex {0} 个" -f $texFiles.Count)
Write-Output ''
if ($diffs.Count) { Write-Output ('---- 【差】{0} 项（须处理）----' -f $diffs.Count); $diffs | ForEach-Object { Write-Output $_ } }
else { Write-Output '---- 【差】0 项 ----' }
if ($infos.Count) { Write-Output ''; Write-Output ('---- 【提示】{0} 项（人工判断）----' -f $infos.Count); $infos | ForEach-Object { Write-Output $_ } }
if (-not $diffs.Count -and -not $infos.Count) { Write-Output '全部通过，无差异。' }
