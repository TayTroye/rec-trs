MODEL="AutoCode"
CONFIG="yaml/autocode_ml.yaml"
DATASET="ml-1m"


python run_recbole.py \
    --model=$MODEL \
    --config_files=$CONFIG \
    --dataset=$DATASET



CONFIG="yaml/autocode-tag.yaml"
DATASET="MovielensLatest_x1"

python run_recbole.py \
    --model=$MODEL \
    --config_files=$CONFIG \
    --dataset=$DATASET


CONFIG="yaml/autocode-frappe.yaml"
DATASET="Frappe"

python run_recbole.py \
    --model=$MODEL \
    --config_files=$CONFIG \
    --dataset=$DATASET