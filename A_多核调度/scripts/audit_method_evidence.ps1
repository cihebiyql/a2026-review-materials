# 本程序及代码是在人工智能工具辅助下完成的。
# 工具：Codex；版本/型号：GPT-6（当前会话系统标识）；开发机构：OpenAI；版本发布日期：待补。
# PowerShell 7. Static provenance and arithmetic audit, not a solver/evaluator run.
param([string]$SourceCommit = 'e3f5854edfb5aa75e3073a3b7acaf28d1fc4bc1f')
$ErrorActionPreference = 'Stop'
$paperRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = (Resolve-Path (Join-Path $paperRoot '../../..')).Path
function Get-SourceRecord([string]$RelativePath) {
    $fullPath = Join-Path $repoRoot $RelativePath
    $expected = git -C $repoRoot rev-parse "${SourceCommit}:$RelativePath"
    if ($LASTEXITCODE -ne 0) { throw "Missing source: $RelativePath" }
    $actual = git -C $repoRoot hash-object -- $fullPath
    if ($LASTEXITCODE -ne 0 -or $actual -ne $expected) { throw "Source differs from pinned commit: $RelativePath" }
    $changedAt = git -C $repoRoot log -1 $SourceCommit --format=%H -- $RelativePath
    [ordered]@{path=$RelativePath; blob=$actual; sha256=(Get-FileHash -LiteralPath $fullPath).Hash.ToLowerInvariant(); last_repository_change=$changedAt}
}
$codePaths = @(
    'competition/2026/A题_代码包/a_lab/lab_solvers/arm_merge.py',
    'competition/2026/A题_代码包/a_lab/solvers/exp_merge/p1_v3.py',
    'competition/2026/A题_代码包/a_lab/lab_solvers/arm_p2strand.py',
    'competition/2026/A题_代码包/a_lab/solvers/exp_p2strand/p2_v3.py',
    'competition/2026/A题_代码包/a_lab/lab_solvers/arm_p3strand.py',
    'competition/2026/A题_代码包/a_lab/lab_solvers/arm_v3_wave.py',
    'competition/2026/A题_外部AI审阅包_0924晚/src/mechA/p1_refine.py',
    'competition/2026/A题_外部AI审阅包_0924晚/src/mechA/mechA_deep_seq.py',
    'competition/2026/A题_外部AI审阅包_0924晚/src/mechA/mechA_v4.py',
    'competition/2026/A题_代码包/a_lab/registry/solvers.json',
    'competition/2026/A题_代码包/a_lab/contracts/budget.yaml',
    'competition/2026/A题_代码包/a_lab/contracts/version.py'
)
$expIds = @('A-P1-MERGE-002','A-P2-STRAND-002','A-P3-STRAND-002')
$results = for ($q=1; $q -le 3; $q++) {
    $expId = $expIds[$q-1]
    $prefix = 'competition/2026/a_lab实验系统_自动同步'
    $sources = @(
        Get-SourceRecord "$prefix/experiments/$expId.yaml"
        Get-SourceRecord "$prefix/verdicts/$expId.comparison.json"
        Get-SourceRecord "$prefix/verdicts/$expId.verdict.json"
    )
    $comparison = Get-Content -LiteralPath (Join-Path $repoRoot $sources[1].path) -Raw | ConvertFrom-Json -AsHashtable
    $verdict = Get-Content -LiteralPath (Join-Path $repoRoot $sources[2].path) -Raw | ConvertFrom-Json -AsHashtable
    $keys = @(1..100 | ForEach-Object { 'case_{0:D3}|q{1}|N4|s11' -f $_,$q })
    $rows = $comparison.per_case
    if ($rows.Count -ne 100 -or @($keys | Where-Object { -not $rows.ContainsKey($_) }).Count) { throw "Unexpected task keys: $expId" }
    $armStats = foreach ($arm in @('A','B')) {
        $values = @($keys | ForEach-Object { [double]$rows[$_]["${arm}_speedup"] })
        if (@($values | Where-Object { $_ -le 0 -or [double]::IsNaN($_) -or [double]::IsInfinity($_) }).Count) { throw "Bad speedups: $expId" }
        $mean = ($values | Measure-Object -Average).Average
        $reported = [double]$comparison["S_mean_$arm"]
        if ([math]::Abs($mean-$reported) -gt 0.00000051) { throw "Mean mismatch: $expId $arm" }
        [ordered]@{arm=$arm; name=$comparison["name_"+$arm.ToLowerInvariant()]; mean_from_per_case=$mean; reported_mean=$reported; status_counts=$comparison["status_$arm"]; reported_cost=$comparison["cost_$arm"]}
    }
    [ordered]@{
        result_id = 'RES-00'+($q+6)
        experiment_id = $expId
        question = $q
        cores = 4
        seed = 11
        count = 100
        case_keys = $keys
        budget_policy_from_yaml = 'dev_full'
        source_verdict = $verdict.verdict
        sources = $sources
        arms = @($armStats)
        wins_B = $comparison.wins_B
        losses_B = $comparison.losses_B
        ties = $comparison.ties
        status = 'PRELIMINARY'
        limitation = 'Repository provenance and stored arithmetic only. Runtime binding, original plans, official evaluation, actual budget enforcement and raw timing were not independently verified.'
    }
}
$outPath = Join-Path $paperRoot 'results/progress_audit_20260925/method_evidence.json'
[ordered]@{
    source_commit=$SourceCommit
    generated_at=(Get-Date).ToString('o')
    code_sources=@($codePaths | ForEach-Object { Get-SourceRecord $_ })
    note='A last_repository_change is a repository snapshot identifier, not proof of the code actually used in an experiment.'
    results=@($results)
} | ConvertTo-Json -Depth 9 | Set-Content -LiteralPath $outPath -Encoding utf8
$results | ForEach-Object { '{0} {1}: A={2:F8}, B={3:F8}; recorded verdict={4}' -f $_.result_id,$_.experiment_id,$_.arms[0].mean_from_per_case,$_.arms[1].mean_from_per_case,$_.source_verdict }
Write-Output "Saved: $outPath"
