TASK=$1
CHECKPOINT_NUMBER=$2
INPUT_TYPE=image_tensor
export CUDA_VISIBLE_DEVICES=
# export environment variables
ROOT_DIR="$( cd "$(dirname "$0")"/.. ; pwd -P )"
echo 'Root directory: ' $ROOT_DIR
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR:$ROOT_DIR/slim

PIPELINE_CONFIG_PATH=experiments/${TASK}/pipeline.config
TRAINED_CKPT_PREFIX=models/${TASK}/model.ckpt-${CHECKPOINT_NUMBER}
EXPORT_DIR=inference_models/${TASK}_${CHECKPOINT_NUMBER}

python3 ../object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
