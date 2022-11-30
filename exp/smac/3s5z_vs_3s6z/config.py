from copy import deepcopy
from ding.entry import serial_pipeline
from easydict import EasyDict

agent_num = 8
collector_env_num = 8
evaluator_env_num = 8

main_config = dict(
    env=dict(
        map_name='3s5z_vs_3s6z',
        difficulty=7,
        reward_type='original',
        agent_num=agent_num,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        manager=dict(shared_memory=True, ),
        stop_value=1.999,
        n_evaluator_episode=32,
    ),
    policy=dict(
        cuda=True,
        priority=False,
        model=dict(
            agent_num=agent_num,
            embed_num=6,
            state_len=34, # 32+ num of unit type
            relation_len=6,
            hidden_len=256,
            local_pred_len=6,
            global_pred_len=12,
        ),
        learn=dict(
            multi_gpu=False,
            update_per_collect=50,
            batch_size=320,
            learning_rate=0.0003,
            clip_value=50,
            double_q=False,
            target_update_theta=0.008,
            nstep=3,
            discount_factor=0.99,
            aux_loss_weight=dict(
                begin=10,
                end=10,
                T_max=400000,
            ),
            aux_label_norm=True,
            shuffle=True,
            learning_rate_type='cosine',
            learning_rate_tmax=100000,
            learning_rate_eta_min=3e-6,
            learner=dict(
                hook=dict(
                    log_show_after_iter=2000,
                    save_ckpt_after_iter=10000000000,
                    save_ckpt_after_run=True,
                ),
            ),
        ),
        collect=dict(
            n_episode=32,
            unroll_len=1,
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num, evaluator=dict(eval_freq=1000, )),
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
        type='smac_ace',
        import_names=['dizoo.smac.envs.smac_env_ace'],
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
