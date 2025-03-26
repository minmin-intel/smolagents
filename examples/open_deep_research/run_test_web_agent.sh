# PROVIDER="deepseek"
# MODEL="deepseek-chat"

PROVIDER="openai"
MODEL="openai/gpt-4o"

QUESTION=2
logfile=$WORKDIR/datasets/test_web_search/smolagents_q${QUESTION}_$PROVIDER.log

python test_web_agent.py --provider $PROVIDER --model-id $MODEL --question $QUESTION | tee $logfile