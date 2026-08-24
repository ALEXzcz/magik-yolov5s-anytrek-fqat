#!/bin/bash

weight=""
bit=""
mode=""
source_path="data/anytrek_detect/images"
data_cfg="data/anytrek_multiclass.yaml"
float_model="./checkpoint/32W32F/weights/best-tmp.pt"

while [ $# -gt 0 ]; do
    case "$1" in
        --bit)
            bit=$2
            shift 2
            ;;
        --weight)
            weight=$2
            shift 2
            ;;
        --mode)
            mode=$2
            shift 2
            ;;
        --source)
            source_path=$2
            shift 2
            ;;
        --data)
            data_cfg=$2
            shift 2
            ;;
        --float-model)
            float_model=$2
            shift 2
            ;;
        *) # 未知参数
            echo "Unknown parameter: $1"
            shift
            ;;
    esac
done

if [ -z "$weight" ]; then
    weight="checkpoint/${bit}W${bit}F/weights/best-tmp.pt"
fi

log_file="$mode.log"
current_time=$(date "+%Y-%m-%d %H:%M:%S")
{
   echo "----------------------------------------------------------------------------------------------------"
   echo "Process: $mode, Bit: $bit, Weight: $weight, Time: $current_time"
   echo "----------------------------------------------------------------------------------------------------"
} >> "$log_file"

case "$mode" in
   detect)
       python detect.py --source "$source_path" --data "$data_cfg" --float_model "$float_model" --weights "$weight" --imgs 640 --device 0 --bit "$bit" >> "$log_file" 2>&1
       ;;
   val)
       python val.py --data "$data_cfg" --float_model "$float_model" --weights "$weight" --imgs 640 --device 0 --batch-size 50 --bit "$bit" >> "$log_file" 2>&1
       ;;
   export)
       python detect.py --source "$source_path" --data "$data_cfg" --float_model "$float_model" --weights "$weight" --imgs 640 --device 0 --bit "$bit" --onnx >> "$log_file" 2>&1
       ;;
   *)
       echo "Unknown mode: $mode" >&2
       exit 1
       ;;
esac
