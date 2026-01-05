@echo off
:: Set working directory to project root
cd /d %~dp0..

:: Record start time
echo [%DATE% %TIME%] Starting background execution >> execution_log.txt

:: Command 1: Run strategy6 pipeline, output to log6_output.txt
echo [%DATE% %TIME%] Starting command 1: test_strategy6_pipeline.py >> execution_log.txt
python dimension_strategy_comparison/test_strategy6_pipeline.py --concurrency 5 --resume > dimension_strategy_comparison/log6_output.txt 2>&1
echo [%DATE% %TIME%] Command 1 completed >> execution_log.txt

:: Command 2: Run LLM attack extraction, output to log6_llm.txt
echo [%DATE% %TIME%] Starting command 2: step4c_extract_with_llm_attacks.py >> execution_log.txt
python dimension_strategy_comparison/scripts/step4c_extract_with_llm_attacks.py --concurrency 1 --resume > dimension_strategy_comparison/log6_llm.txt 2>&1
echo [%DATE% %TIME%] Command 2 completed >> execution_log.txt

:: Record completion time
echo [%DATE% %TIME%] All commands completed >> execution_log.txt
