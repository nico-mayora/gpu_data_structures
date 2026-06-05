@echo off
setlocal EnableDelayedExpansion

set SCENE=%1
set NORMAL=%2
set CAUSTIC=%3
set "VISIBLE="
if /i "%4"=="visible" set "VISIBLE=visible"

if "%SCENE%"=="" goto usage
if "%NORMAL%"=="" goto usage
if "%CAUSTIC%"=="" goto usage
goto start

:usage
echo Usage: run_benchmark.bat ^<scene^> ^<normal_photons^> ^<caustic_photons^> [visible]
echo Example: run_benchmark.bat cornell-box 1000000 100000
echo          run_benchmark.bat cornell-box 1000000 100000 visible
echo.
exit /b 1

:start
set RESULTS=results_%SCENE%_n%NORMAL%_c%CAUSTIC%.txt
echo Benchmark: scene=%SCENE% normal=%NORMAL% caustic=%CAUSTIC% > %RESULTS%
echo. >> %RESULTS%
echo Executable                Time(ms) >> %RESULTS%
echo ------------------------- -------- >> %RESULTS%

set EXES=pathTracer_kglobal_4 pathTracer_kglobal_16 pathTracer_kglobal_64 pathTracer_kglobal_128 pathTracer_kglobal_256 pathTracer_kcaustic_4 pathTracer_kcaustic_16 pathTracer_kcaustic_64 pathTracer_kcaustic_128 pathTracer_kcaustic_256

for %%E in (%EXES%) do (
    echo [RUN] %%E ...
    set "time_ms=DNF"
    if exist "%%E.exe" (
        .\%%E.exe %SCENE% %NORMAL% %CAUSTIC% benchmark %VISIBLE% > tmp_%%E.log 2>&1
        if not errorlevel 1 (
            for /f "tokens=2" %%A in ('findstr "BENCHMARK_RESULT:" tmp_%%E.log') do (
                set "time_ms=%%A"
            )
        ) else (
            echo [WARN] %%E.exe crashed or exited with non-zero code.
        )
    ) else (
        echo [WARN] %%E.exe not found, skipping.
    )
    echo %%E !time_ms! >> %RESULTS%
    echo [DONE] %%E = !time_ms! ms
)

echo. >> %RESULTS%
echo Summary Table >> %RESULTS%
echo ------------------------- -------- >> %RESULTS%

type %RESULTS%
echo.
echo Results saved to %RESULTS%

exit /b 0
