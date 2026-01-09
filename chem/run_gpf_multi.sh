base_model=$1
all_run_seeds=(42 77 728 2025 1234)

for seed in ${all_run_seeds[@]}
do
    echo "Running seed: $seed"
    case $base_model in
        infomax)
            python prompt_tuning_full_shot.py --model_file pretrained_models/infomax.pth --dataset bbbp --tuning_type gpf_multi --num_layers 2 --runseed $seed --output_file result_multi.log --lr 0.002 --decay 0.001
            python prompt_tuning_full_shot.py --model_file pretrained_models/infomax.pth --dataset tox21 --tuning_type gpf_multi --num_layers 2 --runseed $seed --output_file result_multi.log --lr 0.0001
            python prompt_tuning_full_shot.py --model_file pretrained_models/infomax.pth --dataset toxcast --tuning_type gpf_multi --num_layers 3 --runseed $seed --output_file result_multi.log --lr 0.0002
            python prompt_tuning_full_shot.py --model_file pretrained_models/infomax.pth --dataset sider --tuning_type gpf_multi --num_layers 3 --runseed $seed --output_file result_multi.log
            python prompt_tuning_full_shot.py --model_file pretrained_models/infomax.pth --dataset clintox --tuning_type gpf_multi --num_layers 2 --runseed $seed --output_file result_multi.log
            # python prompt_tuning_full_shot.py --model_file pretrained_models/infomax.pth --dataset muv --tuning_type gpf_multi --num_layers 1 --runseed $seed --output_file result_multi.log
            # python prompt_tuning_full_shot.py --model_file pretrained_models/infomax.pth --dataset hiv --tuning_type gpf_multi --num_layers 2 --runseed $seed --output_file result_multi.log
            python prompt_tuning_full_shot.py --model_file pretrained_models/infomax.pth --dataset bace --tuning_type gpf_multi --num_layers 2 --runseed $seed --output_file result_multi.log --lr 0.0002 --use_cosine
            ;;
        masking)
            python prompt_tuning_full_shot.py --model_file pretrained_models/masking.pth --dataset bbbp --tuning_type gpf_multi --num_layers 3 --output_file result_multi.log --runseed $seed --lr 0.001 --decay 0.001 --use_cosine
            python prompt_tuning_full_shot.py --model_file pretrained_models/masking.pth --dataset tox21 --tuning_type gpf_multi --num_layers 2 --output_file result_multi.log --runseed $seed --lr 0.001 --decay 0.0001 --use_cosine
            python prompt_tuning_full_shot.py --model_file pretrained_models/masking.pth --dataset toxcast --tuning_type gpf_multi --num_layers 2 --output_file result_multi.log --runseed $seed --use_cosine
            python prompt_tuning_full_shot.py --model_file pretrained_models/masking.pth --dataset sider --tuning_type gpf_multi --num_layers 3 --output_file result_multi.log --runseed $seed --use_cosine
            python prompt_tuning_full_shot.py --model_file pretrained_models/masking.pth --dataset clintox --tuning_type gpf_multi --num_layers 3 --output_file result_reproduce.log --runseed $seed --lr 0.0005 --use_cosine
            ;;
        *)
            echo "Unknown base model: $base_model"
            ;;
    esac
done
