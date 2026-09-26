[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$engine = Get-Command xelatex -ErrorAction SilentlyContinue
if (-not $engine) {
    # PATH 未命中时回退探测常见安装位（用户级 MiKTeX / TeX Live）
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\MiKTeX\miktex\bin\x64\xelatex.exe'),
        'C:\Program Files\MiKTeX\miktex\bin\x64\xelatex.exe',
        'C:\texlive\2026\bin\windows\xelatex.exe',
        'C:\texlive\2025\bin\windows\xelatex.exe'
    ) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if ($candidates) { $engine = [pscustomobject]@{ Source = $candidates } }
}
if (-not $engine) {
    throw '未找到 XeLaTeX。请安装含中文支持的 TeX Live 或 MiKTeX（用户级 MiKTeX 会自动回退探测），并确保 xelatex 可直接调用。'
}
Push-Location -LiteralPath $PSScriptRoot
try {
    New-Item -ItemType Directory -Path 'build' -Force | Out-Null
    for ($pass = 1; $pass -le 2; $pass++) {
        & $engine.Source '-interaction=nonstopmode' '-halt-on-error' '-file-line-error' '-output-directory=build' '论文.tex'
        if ($LASTEXITCODE -ne 0) {
            throw "第 $pass 遍编译失败，请查看 build/论文.log。"
        }
    }
    if (-not (Test-Path -LiteralPath 'build/论文.pdf')) {
        throw '未生成 build/论文.pdf，请查看编译日志。'
    }

    # ---- lint（D031）：只报告不阻断；提交前 Overfull 须清零 ----
    $logPath = Join-Path 'build' '论文.log'
    $logLines = if (Test-Path -LiteralPath $logPath) { Get-Content -LiteralPath $logPath } else { @() }
    $logChecks = @(
        @{ re = 'LaTeX Warning: Reference .+ undefined';      label = '未定义引用(reference, 正文显示 ??)' },
        @{ re = 'LaTeX Warning: Citation .+ undefined';       label = '未定义 citation' },
        @{ re = 'multiply defined';                           label = '重复标签(multiply defined)' },
        @{ re = 'Overfull \\hbox';                            label = 'Overfull \\hbox' },
        @{ re = 'Label\(s\) may have changed';                label = '交叉引用未稳定(需第三遍编译)' }
    )
    $findings = @()
    foreach ($c in $logChecks) {
        $m = @($logLines | Select-String -Pattern $c.re)
        if ($m.Count -gt 0) { $findings += [pscustomobject]@{ 项 = $c.label; 数量 = $m.Count; 示例 = ($m[0].Line.Trim()) } }
    }
    # 源文件待补标记（骨架期有意存在，交稿前须清零）
    $srcHits = @(Get-ChildItem -LiteralPath $PSScriptRoot -Recurse -Filter '*.tex' |
        Where-Object { $_.FullName -notmatch '\\build\\' } |
        Select-String -Pattern '\\pending|TODO')
    if ($srcHits.Count -gt 0) {
        $findings += [pscustomobject]@{ 项 = '待补标记(\pending/TODO, 源文件)'; 数量 = $srcHits.Count; 示例 = ('{0}:{1}' -f $srcHits[0].Path, $srcHits[0].LineNumber) }
    }
    if ($findings.Count -gt 0) {
        Write-Output ''
        Write-Output '==== lint 发现（提交前处理：Overfull/未定义引用/重复标签须清零；待补标记按计划收敛）===='
        $findings | Format-Table -AutoSize | Out-String -Width 200 | Write-Output
    }
    else { Write-Output 'lint 通过：无待补标记、无未定义引用、无 Overfull、无重复标签。' }

    Write-Output (Join-Path $PSScriptRoot 'build\论文.pdf')
}
finally {
    Pop-Location
}
