$ErrorActionPreference = 'Stop'
$deck = 'D:\StockPredictor\presentation_output\Stock_Market_Intelligence_Engine.pptx'
$outDir = 'D:\StockPredictor\presentation_output\render'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Get-ChildItem $outDir -Filter 'slide-*.png' | Remove-Item -Force

$ppt = New-Object -ComObject PowerPoint.Application
try {
    $pres = $ppt.Presentations.Open($deck, $true, $false, $false)
    Write-Output ("SLIDES=" + $pres.Slides.Count)
    Write-Output ("SIZE=" + $pres.PageSetup.SlideWidth + "x" + $pres.PageSetup.SlideHeight)
    foreach ($s in $pres.Slides) {
        $target = Join-Path $outDir ("slide-{0:00}.png" -f $s.SlideIndex)
        $s.Export($target, 'PNG', 1600, 900)
        Write-Output ("EXPORTED " + $target)
    }
    $pres.Close()
} finally {
    $ppt.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
