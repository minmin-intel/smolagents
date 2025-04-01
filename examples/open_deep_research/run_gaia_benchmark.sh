agent_type=code
# agent_type=toolcalling

PROVIDER="deepseek"
MODEL="deepseek-chat"

# PROVIDER="openai"
# MODEL="openai/gpt-4o"

max_steps=3

# data path
subset=validation
datapath=$WORKDIR/owl/examples/data/gaia/2023/
export DATAPATH=${datapath}/${subset}
test_type=text
output=$WORKDIR/datasets/gaia/smolagents_tests/results_${test_type}_${PROVIDER}_singlewebagent_${agent_type}.jsonl
logfile=$WORKDIR/datasets/gaia/smolagents_tests/log_${test_type}_${PROVIDER}_singlewebagent_${agent_type}.log

python test_web_agent.py \
--provider $PROVIDER \
--model-id $MODEL \
--max_steps $max_steps \
--agent_type $agent_type \
--datapath $datapath \
--answer_file $output \
--test_type $test_type \
--debug | tee $logfile