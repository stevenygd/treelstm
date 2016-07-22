#! /bin/bash

LOG_FILE_PATH=../textmatters/stats/rel_text/median.pkl
OUT_FILE_NAME='rel_text_median.out'
FIELD_NAME='tree_lstm_sim'
while [ $# -lt 3 ]; do
    echo "Usage: ./run.sh <log_file_path> <out_file_name> <field_name>"
done

LOG_FILE_PATH=$1
OUT_FILE_NAME=$2
FIELD_NAME=$3
echo "Display arguments:"
echo "LOG_FILE_PATH = $LOG_FILE_PATH"
echo "OUT_FILE_NATM = $OUT_FILE_NAME"
echo "FIELD_NAME    = $FIELD_NAME"

# Preprocessing
echo "Start preprocessing..." && \
python scripts/preprocess-sick.py --logfile $LOG_FILE_PATH --tmpfile "tmp_${OUT_FILE_NAME}" && \

# Evaluation
echo "Start evaluating..." && \
th relatedness/eval.lua trained_models/rel-dependency.1l.150d.2.th -d tmp_$OUT_FILE_NAME/ -o $OUT_FILE_NAME  && \

# Postprocess, re-added into the previous data dictionary
echo "Start post-processing..." && \
python scripts/util.py reattach --logfile $LOG_FILE_PATH --scorefile predictions/$OUT_FILE_NAME --fieldname $FIELD_NAME
