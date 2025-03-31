agent_type=code
# agent_type=toolcalling

PROVIDER="deepseek"
MODEL="deepseek-chat"

# PROVIDER="openai"
# MODEL="openai/gpt-4o"

max_steps=15
QUESTION=2
logfile=$WORKDIR/datasets/test_web_search/smolagents_q${QUESTION}_${PROVIDER}_${agent_type}.log

python test_web_agent.py \
--provider $PROVIDER \
--model-id $MODEL \
--question $QUESTION \
--max_steps $max_steps \
--agent_type $agent_type | tee $logfile