import gfootball.env as football_env
from gfootball.env import observation_preprocessing
import gym
import numpy as np
from ding.utils import ENV_REGISTRY
from typing import Any, List, Union, Optional
import copy
import torch
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray, to_list
import os
from matplotlib import animation
import matplotlib.pyplot as plt


@ENV_REGISTRY.register('gfootball-academy-ace')
class GfootballAcademyEnv(BaseEnv):

    def __init__(
            self,
            cfg: dict,
            dense_reward=False,
            write_full_episode_dumps=False,
            write_goal_dumps=False,
            dump_freq=1000,
            render=False,
            time_limit=150,
            time_step=0,
            stacked=False,
            representation="simple115",
            rewards='scoring',
            logdir='football_dumps',
            write_video=True,
            number_of_right_players_agent_controls=0,
            reward_score=100,
    ):
        """
        'academy_3_vs_1_with_keeper'
        n_agents=4,
        'academy_counterattack_hard'
        n_agents=5,
        """
        self._cfg = cfg
        self.dense_reward = dense_reward
        self.write_full_episode_dumps = write_full_episode_dumps
        self.write_goal_dumps = write_goal_dumps
        self.dump_freq = dump_freq
        self.render = render
        self.env_name = self._cfg.env_name  # TODO
        self.n_agents = self._cfg.agent_num + 1  # add our keeper
        self.obs_dim = self._cfg.obs_dim

        self.episode_limit = self._cfg.get('time_limit', time_limit)
        self.time_step = time_step
        self.stacked = stacked
        self.representation = representation
        self.rewards = rewards
        self.logdir = logdir
        self.write_video = write_video
        self.number_of_right_players_agent_controls = number_of_right_players_agent_controls
        self._save_replay = self._cfg.save_replay
        self.reward_score = reward_score

        self._env = football_env.create_environment(
            write_full_episode_dumps=self.write_full_episode_dumps,
            write_goal_dumps=self.write_goal_dumps,
            env_name=self.env_name,
            stacked=self.stacked,
            representation=self.representation,
            rewards=self.rewards,
            logdir=self.logdir,
            render=self.render,
            write_video=self.write_video,
            dump_frequency=self.dump_freq,
            number_of_left_players_agent_controls=self.n_agents,
            number_of_right_players_agent_controls=self.number_of_right_players_agent_controls,
            channel_dimensions=(observation_preprocessing.SMM_WIDTH, observation_preprocessing.SMM_HEIGHT),
            other_config_options={'action_set': 'v2', }
        )
        obs = self._env.unwrapped.observation()[1]
        self.left_player_num = obs['left_team'].shape[0] - 1  # minus our keeper
        self.right_player_num = obs['right_team'].shape[0]
        self.n_entities = self.left_player_num + self.right_player_num + 1
        obs_space_low = self._env.observation_space.low[0][:self.obs_dim]
        obs_space_high = self._env.observation_space.high[0][:self.obs_dim]

        self._action_space = gym.spaces.Dict(
            {agent_i: gym.spaces.Discrete(self._env.action_space.nvec[1]) for agent_i in range(self.n_entities)})
        self._observation_space = gym.spaces.Dict({agent_i:
                                                       gym.spaces.Box(low=obs_space_low, high=obs_space_high,
                                                                      dtype=self._env.observation_space.dtype)
                                                   for agent_i in range(self.n_agents-1)
                                                   })
        self._reward_space = gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)  # TODO(pu)

        self.n_actions = self.action_space[0].n - 1


    def check_if_done(self):
        cur_obs = self._env.unwrapped.observation()[1]
        ball_loc = cur_obs['ball']
        ours_loc = cur_obs['left_team'][-self.n_agents:]

        if ball_loc[0] < 0 or any(ours_loc[1:, 0] < 0): # our players is out of bounds excpet for our keeper
            """
            This is based on the CDS paper:
            'We make a small and reasonable change to the half-court offensive scenarios: our players will lose if
            they or the ball returns to our half-court.'
            """
            return True

        return False

    def reset(self):
        """Returns initial observations and states."""
        if self._save_replay:
            self._frames = []
        self.time_step = 0
        self._env.reset()
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        self._final_eval_reward = 0
        self._init_states()
        return self.get_obs()
        return obs

    def _init_states(self):
        self.state_len = 4 + 1 + 3 + 3 + 10 # left player or right keeper or right player or ball, whether own ball, pos.x,y,z relative to the goal, direction.x,y,z relative to the goal
        self.relation_len = 2 + 2  # distance, cos(theta) and sin(theta)
        self.action_len = self.n_actions
        # init satetes
        self.states = np.zeros((self.n_entities, self.state_len), dtype=np.float32)
        self.states[:self.left_player_num, 0] = 1
        self.states[self.left_player_num, 1] = 1
        self.states[self.left_player_num + 1: self.left_player_num + self.right_player_num, 2] = 1
        self.states[self.left_player_num + self.right_player_num, 3] = 1
        # the z axis of pos and direction of all players is zero
        self.states[:self.left_player_num + self.right_player_num, 7] = 0
        self.states[:self.left_player_num + self.right_player_num, 10] = 0
        # init states of the goal, all are zero except the type identification bit
        # init relations
        self.relations = np.zeros((self.n_entities, self.n_entities, self.relation_len), dtype=np.float32)
        # init actions
        self.action_mask = np.ones((self.left_player_num, self.action_len), dtype=np.bool)  # all actions are valid
        self.alive_mask = np.ones(self.n_entities, dtype=np.bool)
        self.last_actions = np.ones(self.left_player_num, dtype=np.int64)
        # update states with current units
        self._update_states()

    def _update_states(self):
        # update unit states
        full_obs = self._env.unwrapped.observation()[1]
        # update ball owner
        self.states[:, 4] = 0
        if int(full_obs['ball_owned_player']) >= 1:
            self.states[full_obs['ball_owned_player']-1, 4] = 1
        # pos
        goal_pos = np.array([1,0,0])
        self.states[:self.left_player_num, 5:7] = full_obs['left_team'][1:] - goal_pos[:2]  # delete our keeper
        self.states[self.left_player_num:self.left_player_num + self.right_player_num, 5:7] = full_obs['right_team'] - goal_pos[:2]
        self.states[self.left_player_num + self.right_player_num, 5:8] = full_obs['ball'] - goal_pos
        # direction
        self.states[:self.left_player_num, 8:10] = full_obs['left_team_direction'][1:] / 0.02  # delete our keeper
        self.states[self.left_player_num:self.left_player_num + self.right_player_num, 8:10] = full_obs['right_team_direction'] / 0.02
        self.states[self.left_player_num + self.right_player_num, 8:11] = full_obs['ball_direction'] / 0.02
        # state
        self.states[:self.left_player_num, 11:] = np.array([i['sticky_actions'] for i in self._env.unwrapped.observation()])[1:].astype(np.float32)

        # update relations
        pos = self.states[:, 5:]
        disp = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt(np.sum(disp * disp, axis=2))
        self.relations[:, :, 0] = disp[:, :, 0]   # x distance
        self.relations[:, :, 1] = disp[:, :, 1]   # y distance
        self.relations[:, :, 2] = disp[:, :, 0] / (dist + 1e-8)  # cos(theta)
        self.relations[:, :, 3] = disp[:, :, 1] / (dist + 1e-8)  # sin(theta)

        # update action mask
        own_ball = self.states[:self.left_player_num, 4].astype(np.bool)
        sprint = self.states[:self.left_player_num, -2].astype(np.bool)
        dribble = self.states[:self.left_player_num, -1].astype(np.bool)
        self.action_mask[:, 9:13] = own_ball[:, None]
        self.action_mask[:, 13] = ~sprint
        self.action_mask[:, 15] = sprint
        self.action_mask[:, 16] = False
        self.action_mask[:, 17] = own_ball & ~dribble
        self.action_mask[:, 18] = dribble

    def step(self, actions):
        """Returns reward, terminated, info."""
        assert isinstance(actions, np.ndarray) or isinstance(actions, list), type(actions)
        self.time_step += 1
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
        actions = [19] + actions  # add 19(action_builtin_ai) for our keeper.
        if self._save_replay:
            self._frames.append(self._env.render(mode='rgb_array'))

        _, reward, done, info = self._env.step(actions)
        reward = sum(reward)

        self._update_states()
        obs = self.get_obs()

        info_ = {
            'battle_won': False,
            'battle_lost': False,
            'draw': False,
            'final_eval_reward': info['score_reward'],
            'final_eval_fake_reward': 0,
        }
        if done:
            if reward > 0:
                reward = self.reward_score
                info_['battle_won'] = True
            else:
                reward = 0
                info_['battle_lost'] = True
        elif self.check_if_done():
            reward = -1
            done = True
            info_['battle_lost'] = True
        else:
            reward = 0
            if self.time_step >= self.episode_limit:
                done = True
        info_['final_eval_fake_reward'] = reward
        info.update(info_)
        info['episode_info'] = {
                'final_eval_fake_reward': info['final_eval_fake_reward'],
            }
        if done:
            if self._save_replay:
                path = os.path.join(
                    self._replay_path, '{}_episode_{}.gif'.format(self.env_name, self._save_replay_count)
                )
                self.display_frames_as_gif(self._frames, path)
                self._save_replay_count += 1

        return BaseEnvTimestep(obs, np.array([reward]).astype(np.float32), done, info)

    def get_obs(self):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        obs = {
            'states': self.states.copy(),
            'relations': self.relations.copy(),
            'action_mask': self.action_mask.copy(),
            'alive_mask': self.alive_mask.copy(),
        }
        return obs

    def render(self):
        pass

    def close(self):
        self._env.close()

    def save_replay(self):
        """Save a replay."""
        pass

    def enable_save_replay(self, replay_path: str) -> None:
        """
        Overview:
            Save replay file in the given path, need to be self-implemented.
        Arguments:
            - replay_path(:obj:`str`): Storage path.
        """
        raise NotImplementedError

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        random_action = to_ndarray([random_action], dtype=np.int64)
        return random_action

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def __repr__(self) -> str:
        return f'GfootballEnv Academy Env {self.env_name}'

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._save_replay = True
        self._replay_path = replay_path
        self._save_replay_count = 0

    @staticmethod
    def display_frames_as_gif(frames: list, path: str) -> None:
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
        anim.save(path, writer='imagemagick', fps=20)

    def info(self) -> 'BaseEnvInfo':
        pass
