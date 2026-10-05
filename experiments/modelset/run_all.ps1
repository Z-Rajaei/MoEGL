# Runs every method of classify.py as an independent process (one result file per method).
param(
    [string]$Prepared = "E:/Project/ModelSet/prepared",
    [string]$Variant = "nodup",
    [string]$Out = "$PSScriptRoot/results",
    [int]$Threads = 2,
    [string[]]$Groups = @("structure", "homo_names", "hetero_names", "hetero_attrs", "hetero_noedges", "tfidf_ffnn tfidf_svm w4mde_svm")
)
$py = Join-Path $PSScriptRoot "..\..\..\..\Revision\.venv\Scripts\python.exe"
foreach ($g in $Groups) {
    $tag = ($g -split " ")[0]
    $dir = "$Out/$Variant/$tag"
    New-Item -ItemType Directory -Force $dir | Out-Null
    $argList = @("-u", "`"$PSScriptRoot/classify.py`"", "--prepared", "`"$Prepared`"", "--variant", $Variant,
                 "--threads", $Threads, "--out", "`"$dir`"", "--methods") + ($g -split " ")
    Start-Process -FilePath $py -ArgumentList $argList -NoNewWindow `
        -RedirectStandardOutput "$dir/log.txt" -RedirectStandardError "$dir/err.txt"
}
