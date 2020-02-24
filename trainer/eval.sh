TASK=$1
GPU_ID=

# export environment variables
ROOT_DIR="$( cd "$(dirname "$0")"/..  ; pwd -P )"
echo 'Root directory: ' $ROOT_DIR
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR:$ROOT_DIR/slim
export CUDA_VISIBLE_DEVICES=$GPU_ID

# assign by hand, TODO: set as arguments (and give some defaults)
OBJAPI=${ROOT_DIR}/object_detection
PDRoot=${ROOT_DIR}/trainer
echo $OBJAPI

# traininf parameters
MODEL_DIR=${PDRoot}/models/${TASK}
PIPELINE_CONFIG=${PDRoot}/experiments/${TASK}/pipeline.config

echo 'Pipeline configuration: ' $PIPELINE_CONFIG
echo 'Save model to ' $MODEL_DIR

python3 $OBJAPI/legacy/eval.py \
  --alsologtostder \
  --pipeline_config_path=${PIPELINE_CONFIG} \
  --checkpoint_dir=${MODEL_DIR} \
  --eval_dir=${MODEL_DIR}/eval \
