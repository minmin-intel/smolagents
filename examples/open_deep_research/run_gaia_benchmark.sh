agent_type=code
# agent_type=toolcalling

PROVIDER="deepseek"
MODEL="deepseek-reasoner"

# PROVIDER="openai"
# MODEL="openai/gpt-4o"

max_steps=20 # action steps

# data path
subset=validation
datapath=$WORKDIR/owl/examples/data/gaia/2023/
export DATAPATH=${datapath}/${subset}
test_type=text
output=$WORKDIR/datasets/gaia/smolagents_tests/results_${test_type}_${MODEL}_singlecodeagent.jsonl
logfile=$WORKDIR/datasets/gaia/smolagents_tests/log_${test_type}_${MODEL}_singlecodeagent.log

python test_web_agent.py \
--provider $PROVIDER \
--model-id $MODEL \
--max_steps $max_steps \
--agent_type $agent_type \
--datapath $datapath \
--answer_file $output \
--subset $subset \
--sampled \
--test_type $test_type \
--question 3 | tee $logfile
#--debug | tee $logfile