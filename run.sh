warmup_epi=10
replay_buffer_size=2000000
total_steps=2000000
max_episode_length=100
train_repeats=5
batch_size=512
eval_freq=10000
gradient_estimator="gst"
env="grid_world"
gst_gap=0.7
user_name="aithalskanda66"
project_name="InterIIT"

python main.py --env ${env} --warmup_episodes ${warmup_epi} --replay_buffer_size ${replay_buffer_size} --total_steps ${total_steps} --eval_freq ${eval_freq} --max_episode_length ${max_episode_length} --train_repeats ${train_repeats} --batch_size ${batch_size} --gradient_estimator ${gradient_estimator} --gst_gap ${gst_gap} --user_name ${user_name} --wandb_project_name ${project_name} --log_grad_variance 

