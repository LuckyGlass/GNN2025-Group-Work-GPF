base_model=$1
all_run_seeds=(42 77 728 2025 1234)

for seed in ${all_run_seeds[@]}
do
    echo "Running seed: $seed"
    case $base_model in
        infomax)
            python prompt_tuning_full_shot.py --model_file pretrained_models/infomax.pth --dataset bbbp --tuning_type gpf --num_layers 2 --output_file result_reproduce.log --runseed $seed
            python prompt_tuning_full_shot.py --model_file pretrained_models/infomax.pth --dataset tox21 --tuning_type gpf --num_layers 2 --output_file result_reproduce.log --runseed $seed
            python prompt_tuning_full_shot.py --model_file pretrained_models/infomax.pth --dataset toxcast --tuning_type gpf --num_layers 3 --output_file result_reproduce.log --runseed $seed
            python prompt_tuning_full_shot.py --model_file pretrained_models/infomax.pth --dataset sider --tuning_type gpf --num_layers 3 --output_file result_reproduce.log --runseed $seed
            python prompt_tuning_full_shot.py --model_file pretrained_models/infomax.pth --dataset clintox --tuning_type gpf --num_layers 2 --output_file result_reproduce.log --runseed $seed
            python prompt_tuning_full_shot.py --model_file pretrained_models/infomax.pth --dataset bace --tuning_type gpf --num_layers 2 --output_file result_reproduce.log --runseed $seed
            ;;
        masking)
            python prompt_tuning_full_shot.py --model_file pretrained_models/masking.pth --dataset bbbp --tuning_type gpf --num_layers 3 --output_file result_reproduce.log --runseed $seed
            python prompt_tuning_full_shot.py --model_file pretrained_models/masking.pth --dataset tox21 --tuning_type gpf --num_layers 2 --output_file result_reproduce.log --runseed $seed
            python prompt_tuning_full_shot.py --model_file pretrained_models/masking.pth --dataset toxcast --tuning_type gpf --num_layers 2 --output_file result_reproduce.log --runseed $seed
            python prompt_tuning_full_shot.py --model_file pretrained_models/masking.pth --dataset sider --tuning_type gpf --num_layers 3 --output_file result_reproduce.log --runseed $seed
            python prompt_tuning_full_shot.py --model_file pretrained_models/masking.pth --dataset clintox --tuning_type gpf --num_layers 3 --output_file result_reproduce.log --runseed $seed
            python prompt_tuning_full_shot.py --model_file pretrained_models/masking.pth --dataset bace --tuning_type gpf --num_layers 2 --output_file result_reproduce.log --runseed $seed
            ;;
        contextpred)
            python prompt_tuning_full_shot.py --model_file pretrained_models/contextpred.pth --dataset bbbp --tuning_type gpf --num_layers 2 --output_file result_reproduce.log --runseed $seed
            python prompt_tuning_full_shot.py --model_file pretrained_models/contextpred.pth --dataset tox21 --tuning_type gpf --num_layers 2 --output_file result_reproduce.log --runseed $seed
            python prompt_tuning_full_shot.py --model_file pretrained_models/contextpred.pth --dataset toxcast --tuning_type gpf --num_layers 3 --output_file result_reproduce.log --runseed $seed
            python prompt_tuning_full_shot.py --model_file pretrained_models/contextpred.pth --dataset sider --tuning_type gpf --num_layers 2 --output_file result_reproduce.log --runseed $seed
            python prompt_tuning_full_shot.py --model_file pretrained_models/contextpred.pth --dataset clintox --tuning_type gpf --num_layers 4 --output_file result_reproduce.log --runseed $seed
            python prompt_tuning_full_shot.py --model_file pretrained_models/contextpred.pth --dataset bace --tuning_type gpf --num_layers 2 --output_file result_reproduce.log --runseed $seed
            ;;
        *)
            echo "Unknown base model: $base_model"
            ;;
    esac
done
