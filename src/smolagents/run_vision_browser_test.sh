MODEL=gpt4o
datapath=$WORKDIR/owl/examples/data/gaia/2023/validation
filename=metadata_test_browseruse.csv

input="$datapath/$filename"
output=$WORKDIR/datasets/gaia/smolagents_tests/results_${MODEL}_visionbrowser.jsonl
logfile=$WORKDIR/datasets/gaia/smolagents_tests/log_${MODEL}_visionbrowser_run3.log

python vision_web_browser.py \
    --input $input \
    --output $output | tee $logfile
