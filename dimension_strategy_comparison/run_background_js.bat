@echo off
chcp 65001 >nul
:: ============================================================
:: JavaScript Watermark Pipeline (Strategy 6)
:: ============================================================

cd /d D:\kyl410\XDF\dimension_strategy_comparison

:: Create log directories
if not exist "logs" mkdir "logs"
if not exist "logs\js" mkdir "logs\js"

:: Set environment variables
set CONFIG=configs/base_config_js.json
set PYTHON="D:\kyl410\XDF\Neurips\XDF\python.exe"
set LOGDIR=logs\js
set PYTHONIOENCODING=utf-8

:: Clear old log
echo. > %LOGDIR%\execution_log_js.txt

:: ============================================================
:: Record start time
:: ============================================================
echo ============================================================ >> %LOGDIR%\execution_log_js.txt
echo JavaScript Watermark Pipeline Started >> %LOGDIR%\execution_log_js.txt
echo Start Time: %DATE% %TIME% >> %LOGDIR%\execution_log_js.txt
echo Config: %CONFIG% >> %LOGDIR%\execution_log_js.txt
echo ============================================================ >> %LOGDIR%\execution_log_js.txt
echo. >> %LOGDIR%\execution_log_js.txt

echo.
echo ============================================================
echo JavaScript Watermark Pipeline - Strategy 6
echo ============================================================
echo Config File: %CONFIG%
echo Log Directory: %LOGDIR%
echo Start Time: %DATE% %TIME%
echo ============================================================
echo.

:: ============================================================
:: Command 1: Run complete pipeline (Steps 1c-5)
:: ============================================================
echo [%DATE% %TIME%] Starting complete pipeline (Steps 1c-5)... >> %LOGDIR%\execution_log_js.txt
echo [Pipeline] Running Steps 1c-5...
echo.

%PYTHON% test_strategy6_pipeline.py ^
    --config %CONFIG% ^
    --concurrency 10 ^
    --resume ^
    > %LOGDIR%\log6_js_pipeline.txt 2>&1

set PIPELINE_EXIT=%ERRORLEVEL%
echo [%DATE% %TIME%] Pipeline completed (Exit Code: %PIPELINE_EXIT%) >> %LOGDIR%\execution_log_js.txt

if %PIPELINE_EXIT% EQU 0 (
    echo [Pipeline] Completed successfully
) else (
    echo [Pipeline] Failed with exit code %PIPELINE_EXIT%
    echo [Pipeline] Check log: %LOGDIR%\log6_js_pipeline.txt
)
echo.

:: ============================================================
:: Command 2: Run LLM attacks extraction (optional)
:: ============================================================
echo [%DATE% %TIME%] Starting LLM attacks extraction... >> %LOGDIR%\execution_log_js.txt
echo [LLM Attacks] Running Step 4c...
echo.

%PYTHON% scripts\step4c_extract_with_llm_attacks.py ^
    --config %CONFIG% ^
    --concurrency 1 ^
    --resume ^
    > %LOGDIR%\log6_js_llm.txt 2>&1

set LLM_EXIT=%ERRORLEVEL%
echo [%DATE% %TIME%] LLM attacks completed (Exit Code: %LLM_EXIT%) >> %LOGDIR%\execution_log_js.txt

if %LLM_EXIT% EQU 0 (
    echo [LLM Attacks] Completed successfully
) else (
    echo [LLM Attacks] Failed with exit code %LLM_EXIT%
    echo [LLM Attacks] Check log: %LOGDIR%\log6_js_llm.txt
)
echo.

:: ============================================================
:: Record end time
:: ============================================================
echo. >> %LOGDIR%\execution_log_js.txt
echo ============================================================ >> %LOGDIR%\execution_log_js.txt
echo JavaScript Watermark Pipeline Finished >> %LOGDIR%\execution_log_js.txt
echo End Time: %DATE% %TIME% >> %LOGDIR%\execution_log_js.txt
echo Pipeline Exit Code: %PIPELINE_EXIT% >> %LOGDIR%\execution_log_js.txt
echo LLM Exit Code: %LLM_EXIT% >> %LOGDIR%\execution_log_js.txt
echo ============================================================ >> %LOGDIR%\execution_log_js.txt

:: ============================================================
:: Display completion info
:: ============================================================
echo ============================================================
echo JavaScript Watermark Pipeline - Completed!
echo ============================================================
echo End Time: %DATE% %TIME%
echo.
echo Pipeline Exit Code: %PIPELINE_EXIT%
echo LLM Exit Code: %LLM_EXIT%
echo.
echo Logs saved to: %LOGDIR%\
echo   - execution_log_js.txt (summary)
echo   - log6_js_pipeline.txt (detailed pipeline output)
echo   - log6_js_llm.txt (LLM attacks output)
echo.
echo Results saved to: results\strategy_6_adaptive_js\
echo.
echo To view detailed logs:
echo   type %LOGDIR%\execution_log_js.txt
echo   type %LOGDIR%\log6_js_pipeline.txt
echo ============================================================
echo.
