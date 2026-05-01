for model in small medium large xl; do
    python train.py --model $model --device cuda --learning_rate 3e-3 --run_name ${model}_01 --mup
done
