from ding.entry import serial_pipeline
from easydict import EasyDict

agent_num = 3
collector_env_num = 8
evaluator_env_num = 32

main_config = dict(
    env=dict(
        env_name='academy_3_vs_1_with_keeper',
        agent_num=agent_num,
        obs_dim=21,
        save_replay=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        stop_value=2,
        n_evaluator_episode=32,
    ),
    policy=dict(
        cuda=True,
        priority=False,
        model=dict(
            agent_num=agent_num,
            state_len=21,
            relation_len=4,
            hidden_len=128,
            local_pred_len=7,
            global_pred_len=14,
        ),
        learn=dict(
            multi_gpu=False,
            update_per_collect=50,
            batch_size=320,
            learning_rate=0.002,
            clip_value=50,
            double_q=False,
            target_update_theta=0.08,
            nstep=9,
            discount_factor=0.99,
            aux_loss_weight=dict(
                begin=100,
                end=100,
                T_max=5000,
            ),
            aux_label_norm=True,
            aux_class_balance=True,
            weight_decay=1e-5,
            optimizer_type='rmsprop',
            env='grf',
            learning_rate_type='cosine',
            learning_rate_tmax=5000,
            learning_rate_eta_min=2e-5,
),
        collect=dict(
            n_episode=32,
            unroll_len=1,
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num, evaluator=dict(eval_freq=500, )),
        other=dict(
            eps=dict(
                type='linear',
                start=1,
                end=0.05,
                decay=50000,
            ),
            replay_buffer=dict(
                replay_buffer_size=300000,
                # (int) The maximum reuse times of each data
                max_reuse=1e+9,
                max_staleness=1e+9,
            ),
        ),
    ),
)
main_config = EasyDict(main_config)
create_config = dict(
    env=dict(
        type='gfootball-academy-ace',
        import_names=['dizoo.gfootball.envs.gfootball_academy_env_ace'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='smac_ace_dqn'),
    collector=dict(type='episode', get_train_sample=True),
)
create_config = EasyDict(create_config)


def train(args):
    main_config.exp_name='seed'+f'{args.seed}'
    config = [main_config, create_config]
    serial_pipeline(config, seed=args.seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()

    train(args)
