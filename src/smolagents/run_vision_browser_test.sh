# MODELTYPE=LiteLLMModel
# MODEL=gpt-4o
# MODELNAME="GPT-4o"

MODELTYPE=OpenAIServerModel
MODEL="Qwen/Qwen2.5-VL-72B-Instruct"
MODELNAME="Qwen25-VL-72B"

datapath=$WORKDIR/owl/examples/data/gaia/2023/validation
filename=metadata_test_browseruse.csv

input="$datapath/$filename"
output=$WORKDIR/datasets/gaia/smolagents_tests/results_${MODELNAME}_visionbrowser.jsonl
logfile=$WORKDIR/datasets/webarena/test_smolagents/logs/log_${MODELNAME}_visionbrowser.log

python vision_web_browser.py \
    --model-id $MODEL \
    --model-type $MODELTYPE \
    --input $input \
    --output $output \
    --quick-test | tee $logfile
   
    #  
