PROVIDER="deepseek"
MODEL="deepseek-chat"
QUESTION=0
logfile=$WORKDIR/datasets/test_web_search/smolagents_q${QUESTION}_$PROVIDER.log

python test_web_agent.py --provider $PROVIDER --model-id $MODEL --question $QUESTION | tee $logfile