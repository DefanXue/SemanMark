@echo off
:: 设置工作目录
cd /d D:\kyl410\XDF\dimension_strategy_comparison

:: 记录开始时间
echo [%DATE% %TIME%] Starting background execution >> execution_log.txt

:: 命令1：执行strategy6流水线，输出到log6_output.txt
echo [%DATE% %TIME%] Starting command 1: test_strategy6_pipeline.py >> execution_log.txt
"D:\kyl410\XDF\Neurips\XDF\python.exe" test_strategy6_pipeline.py --concurrency 5 --resume > log6_output.txt 2>&1
echo [%DATE% %TIME%] Command 1 completed >> execution_log.txt

:: 命令2：执行LLM攻击提取，输出到log6_llm.txt
echo [%DATE% %TIME%] Starting command 2: step4c_extract_with_llm_attacks.py >> execution_log.txt
"D:\kyl410\XDF\Neurips\XDF\python.exe" scripts\step4c_extract_with_llm_attacks.py --concurrency 1 --resume > log6_llm.txt 2>&1
echo [%DATE% %TIME%] Command 2 completed >> execution_log.txt

:: 记录完成时间
echo [%DATE% %TIME%] All commands completed >> execution_log.txt

