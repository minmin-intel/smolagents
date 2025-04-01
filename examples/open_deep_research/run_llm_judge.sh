subset=validation
agent_type=code
PROVIDER="deepseek"
MODEL="deepseek-chat"

test_type=text
folder=$WORKDIR/datasets/gaia/smolagents_tests/
filename=results_${test_type}_${PROVIDER}_singlecodeagent_${agent_type}.csv

python eval_llm_judge.py \
--data_folder $folder \
--filename $filename
