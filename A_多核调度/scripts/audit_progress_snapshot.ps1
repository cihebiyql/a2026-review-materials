# 本程序及代码是在人工智能工具辅助下完成的。
# 工具：Codex；版本/型号：GPT-6（当前会话系统标识）；开发机构：OpenAI；版本发布日期：待补。
# Scope: arithmetic/file-completeness audit only; no solver or official evaluation.
# Run from PowerShell 7: ./audit_progress_snapshot.ps1
param([string]$SourceCommit = 'c427d3ab945dad6bd150adba86e1638429433286')
$ErrorActionPreference = 'Stop'
$paperRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = (Resolve-Path (Join-Path $paperRoot '../../..')).Path
$sourceDir = 'competition/2026/A题_外部AI审阅包_0924晚/results'
$specs = @(
    @{problem = 1; file = 'verified_merge_q1.json'; kind = 'historical solution pool'},
    @{problem = 2; file = 'final_preview_q2.json'; kind = 'preview of historical solution pool'},
    @{problem = 3; file = 'final_preview_q3.json'; kind = 'preview of historical solution pool'}
)
$expectedKeys = @(1..100 | ForEach-Object { 'case_{0:D3}' -f $_ })
$rows = foreach ($spec in $specs) {
    $relativePath = "$sourceDir/$($spec.file)"
    $fullPath = Join-Path $repoRoot $relativePath
    $expectedBlob = git -C $repoRoot rev-parse "${SourceCommit}:$relativePath"
    if ($LASTEXITCODE -ne 0) { throw "Cannot locate pinned source: $relativePath" }
    $actualBlob = git -C $repoRoot hash-object -- $fullPath
    if ($LASTEXITCODE -ne 0 -or $actualBlob -ne $expectedBlob) {
        throw "Working file differs from pinned source: $relativePath"
    }
    $data = Get-Content -LiteralPath $fullPath -Raw -Encoding utf8 | ConvertFrom-Json -AsHashtable
    $missing = @($expectedKeys | Where-Object { -not $data.ContainsKey($_) })
    $extra = @($data.Keys | Where-Object { $_ -notin $expectedKeys })
    if ($missing.Count -or $extra.Count -or $data.Count -ne 100) { throw "Unexpected case coverage: $relativePath" }
    $values = foreach ($key in $expectedKeys) {
        if ($data[$key] -isnot [ValueType]) { throw "Non-numeric value: $key" }
        $value = [double]$data[$key]
        if ([double]::IsNaN($value) -or [double]::IsInfinity($value) -or $value -le 0) {
            throw "Invalid speedup: $key"
        }
        $value
    }
    $stats = $values | Measure-Object -Average -Minimum -Maximum
    [ordered]@{
        result_id = "RES-00$($spec.problem + 3)"
        problem = $spec.problem
        nominal_core_count_from_reports = 5
        source = $relativePath
        source_blob = $actualBlob
        sha256 = (Get-FileHash -LiteralPath $fullPath -Algorithm SHA256).Hash.ToLowerInvariant()
        result_kind = $spec.kind
        status = 'PRELIMINARY'
        case_keys = $expectedKeys
        count = $values.Count
        arithmetic_mean_of_stored_speedups = $stats.Average
        min = $stats.Minimum
        max = $stats.Maximum
        n_above_five = @($values | Where-Object { $_ -gt 5 }).Count
        scope = 'Stored scalar aggregation and exact case-key coverage only. Baselines, plans, evaluator outputs, actual core slots and budgets were not revalidated.'
    }
}
$outDir = Join-Path $paperRoot 'results/progress_audit_20260925'
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
$outPath = Join-Path $outDir 'summary.json'
[ordered]@{
    source_commit = $SourceCommit
    generated_at = (Get-Date).ToString('o')
    formula = 'sum(stored per-case speedup) / 100; not ratio of sums'
    note = 'Audit evidence for notes/02_confirmed_results.md, not an independent result ledger or publication-ready result.'
    results = @($rows)
} | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $outPath -Encoding utf8
$rows | ForEach-Object { '{0}: n={1}, mean={2:F10}, status={3}' -f $_.result_id, $_.count, $_.arithmetic_mean_of_stored_speedups, $_.status }
Write-Output "Saved: $outPath"
