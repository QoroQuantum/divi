# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0
#
# PowerShell equivalent of run_tutorials.sh for Windows CI.
# Uses emojis in output to reproduce UnicodeEncodeError with cp1252 (run with chcp 1252).

$ErrorActionPreference = "Stop"

# Determine repo root: script is in .github/workflows/
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path "$RepoRoot\tutorials")) {
    $RepoRoot = Get-Location
}
Set-Location $RepoRoot

# Create temporary directory and copy tutorials
$TempTutorialsDir = Join-Path $env:TEMP ([System.Guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $TempTutorialsDir | Out-Null
Copy-Item -Path "$RepoRoot\tutorials\*" -Destination $TempTutorialsDir -Recurse

$env:PYTHONPATH = "$RepoRoot;$env:PYTHONPATH"

$ExpectedFailures = @("tutorials/qasm_thru_service.py", "tutorials/circuit_cutting.py")

# Apply sed-equivalent replacements
function Set-FileContentReplacements {
    param([string]$Path, [hashtable]$Replacements)
    $content = Get-Content -Path $Path -Raw -Encoding UTF8
    foreach ($pattern in $Replacements.Keys) {
        $content = $content -replace [regex]::Escape($pattern), $Replacements[$pattern]
    }
    Set-Content -Path $Path -Value $content -Encoding UTF8
}

$vqeHyperparameterSweep = @{
    "shots=2000" = "shots=100"
    "[-0.4, -0.25, 0, 0.25, 0.4]" = "[-0.25, 0, 0.25]"
    "[HartreeFockAnsatz(), UCCSDAnsatz()]" = "[HartreeFockAnsatz()]"
    "max_iterations=3" = "max_iterations=2"
}
Set-FileContentReplacements -Path "$TempTutorialsDir\vqe_hyperparameter_sweep.py" -Replacements $vqeHyperparameterSweep

$zneLocal = @{ "n_processes=4" = "n_processes=4,shots=500" }
Set-FileContentReplacements -Path "$TempTutorialsDir\zne_local.py" -Replacements $zneLocal

$qaoaQuboPartitioning = @{
    "max_iterations=10" = "max_iterations=3"
    "n_layers=2" = "n_layers=1"
}
Set-FileContentReplacements -Path "$TempTutorialsDir\qaoa_qubo_partitioning.py" -Replacements $qaoaQuboPartitioning

$qaoaGraphPartitioning = @{
    "N_NODES = 30" = "N_NODES = 10"
    "N_EDGES = 40" = "N_EDGES = 15"
    "max_n_nodes_per_cluster=10" = "max_n_nodes_per_cluster=5"
    "max_iterations=20" = "max_iterations=5"
    "ParallelSimulator()" = "ParallelSimulator(shots=500)"
    'partitioning_algorithm="metis"' = 'partitioning_algorithm="spectral"'
}
Set-FileContentReplacements -Path "$TempTutorialsDir\qaoa_graph_partitioning.py" -Replacements $qaoaGraphPartitioning

$qaoaQubo = @{
    "n_layers=2" = "n_layers=1"
    "max_iterations=10" = "max_iterations=3"
    "ParallelSimulator(shots=10000)" = "ParallelSimulator(shots=500)"
}
Set-FileContentReplacements -Path "$TempTutorialsDir\qaoa_qubo.py" -Replacements $qaoaQubo

$vqeH2MoleculeLocal = @{ "ParallelSimulator()" = "ParallelSimulator(shots=500)" }
Set-FileContentReplacements -Path "$TempTutorialsDir\vqe_h2_molecule_local.py" -Replacements $vqeH2MoleculeLocal

$qaoaMaxCliqueLocal = @{
    "n_layers=2" = "n_layers=1"
    "max_iterations=10" = "max_iterations=3"
    "ParallelSimulator()" = "ParallelSimulator(shots=500)"
}
Set-FileContentReplacements -Path "$TempTutorialsDir\qaoa_max_clique_local.py" -Replacements $qaoaMaxCliqueLocal

$qaoaQdriftLocal = @{
    "N_NODES, N_EDGES = 12, 25" = "N_NODES, N_EDGES = 8, 12"
    "max_iterations=5" = "max_iterations=3"
    "shots=1000" = "shots=500"
}
Set-FileContentReplacements -Path "$TempTutorialsDir\qaoa_qdrift_local.py" -Replacements $qaoaQdriftLocal

$failures = [System.Collections.ArrayList]::new()

function Run-Tutorial {
    param([string]$FilePath)
    $fileName = Split-Path -Leaf $FilePath
    $originalPath = "tutorials/$fileName"
    Write-Host "🔹 Running $originalPath"

    $expectedFailure = $ExpectedFailures -contains $originalPath
    if ($expectedFailure) {
        Write-Host "⚠️ Expecting failure for $originalPath"
    }

    & poetry run python $FilePath
    $succeeded = ($LASTEXITCODE -eq 0)

    if ($expectedFailure) {
        if ($succeeded) {
            Write-Host "❌ $originalPath was expected to fail but passed"
            [void]$failures.Add("$originalPath (unexpected success)")
        } else {
            Write-Host "✅ $originalPath failed as expected"
        }
    } else {
        if (-not $succeeded) {
            Write-Host "❌ $originalPath failed unexpectedly"
            [void]$failures.Add("$originalPath (unexpected failure)")
        } else {
            Write-Host "✅ $originalPath passed"
        }
    }
}

# Run each tutorial script sequentially
$pyFiles = Get-ChildItem -Path $TempTutorialsDir -Filter "*.py" -File
foreach ($file in $pyFiles) {
    Run-Tutorial -FilePath $file.FullName
}

Write-Host ""
if ($failures.Count -gt 0) {
    Write-Host "❌ Some scripts failed:"
    foreach ($f in $failures) {
        Write-Host "   - $f"
    }
    Remove-Item -Path $TempTutorialsDir -Recurse -Force -ErrorAction SilentlyContinue
    exit 1
} else {
    Write-Host "✅ All tutorials scripts behaved as expected."
    Remove-Item -Path $TempTutorialsDir -Recurse -Force -ErrorAction SilentlyContinue
    exit 0
}
