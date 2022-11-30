from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple
import torch
import copy
import math

from ding.torch_utils import Adam, RMSprop, to_device
from ding.rl_utils import q_nstep_td_data, q_nstep_td_error, get_nstep_return_data, get_train_sample, l2_balance
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import timestep_collate, default_collate, default_decollate
from .base_policy import Policy

@POLICY_REGISTRY.register('smac_ace_dqn')
class SMACACEDQNPolicy(Policy):
    """
    Overview:
        Policy class of ACE algorithm. ACE is a multi agent reinforcement learning algorithm, \
            you can view the paper in the following link https://arxiv.org/abs/2211.16068
    Interface:
        _init_learn, _data_preprocess_learn, _forward_learn, _reset_learn, _state_dict_learn, _load_state_dict_learn \
            _init_collect, _forward_collect, _reset_collect, _process_transition, _init_eval, _forward_eval \
            _reset_eval, _get_train_sample, default_model
    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      qmix           | RL policy register name, refer to      | this arg is optional,
                                                        | registry ``POLICY_REGISTRY``           | a placeholder
        2  ``cuda``             bool     True           | Whether to use cuda for network        | this arg can be diff-
                                                                                                 | erent from modes
        3  ``on_policy``        bool     False          | Whether the RL algorithm is on-policy
                                                        | or off-policy
        4. ``priority``         bool     False          | Whether use priority(PER)              | priority sample,
                                                                                                 | update priority
        5  | ``priority_``      bool     False          | Whether use Importance Sampling        | IS weight
           | ``IS_weight``                              | Weight to correct biased update.
        6  | ``learn.update_``  int      20             | How many updates(iterations) to train  | this args can be vary
           | ``per_collect``                            | after collector's one collection. Only | from envs. Bigger val
                                                        | valid in serial training               | means more off-policy
        7  | ``learn.target_``   float    0.001         | Target network update momentum         | between[0,1]
           | ``update_theta``                           | parameter.
        8  | ``learn.discount`` float    0.99           | Reward's future discount factor, aka.  | may be 1 when sparse
           | ``_factor``                                | gamma                                  | reward env
        == ==================== ======== ============== ======================================== =======================
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='smac_ace_dqn',
        # (bool) Whether to use cuda for network.
        cuda=True,
        # (bool) Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            update_per_collect=20,
            batch_size=32,
            learning_rate=0.0005,
            clip_value=100,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) Target network update momentum parameter.
            # in [0, 1].
            target_update_theta=0.008,
            # (float) The discount factor for future rewards,
            # in [0, 1].
            discount_factor=0.99,
            nstep=1,
            shuffle=False,
            aux_loss_weight=0.0,
            learning_rate_type='constant',
            weight_decay=1e-5,
            optimizer_type='rmsprop',
        ),
        collect=dict(
            # (int) Only one of [n_sample, n_episode] shoule be set
            # n_episode=32,
            # (int) Cut trajectories into pieces with length "unroll_len", the length of timesteps
            # in each forward when training. In qmix, it is greater than 1 because there is RNN.
            unroll_len=1,
        ),
        eval=dict(),
        other=dict(
            eps=dict(
                # (str) Type of epsilon decay
                type='exp',
                # (float) Start value for epsilon decay, in [0, 1].
                # 0 means not use epsilon decay.
                start=1,
                # (float) Start value for epsilon decay, in [0, 1].
                end=0.05,
                # (int) Decay length(env step)
                decay=50000,
            ),
            replay_buffer=dict(
                replay_buffer_size=10000,
                # (int) The maximum reuse times of each data
                max_reuse=1e+9,
                max_staleness=1e+9,
            ),
        ),
    )

    def _init_learn(self) -> None:
        """
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the learner model
        Arguments:
            .. note::

                The _init_learn method takes the argument from the self._cfg.learn in the config file

            - learning_rate (:obj:`float`): The learning rate fo the optimizer
            - gamma (:obj:`float`): The discount factor
            - agent_num (:obj:`int`): Since this is a multi-agent algorithm, we need to input the agent num.
            - batch_size (:obj:`int`): Need batch size info to init hidden_state plugins
        """
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        if self._cfg.learn.optimizer_type == 'adam':
            self._optimizer = Adam(
                self._model.parameters(),
                lr=self._cfg.learn.learning_rate,
                weight_decay=self._cfg.learn.weight_decay,
            )
        else:
            self._optimizer = RMSprop(
                params=self._model.parameters(),
                lr=self._cfg.learn.learning_rate,
                alpha=0.99,
                eps=0.00001,
                weight_decay=self._cfg.learn.weight_decay,
            )
        if self._cfg.learn.learning_rate_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self._scheduler = CosineAnnealingLR(self._optimizer, T_max=self._cfg.learn.learning_rate_tmax,
                                                eta_min=self._cfg.learn.learning_rate_eta_min)

        self._gamma = self._cfg.learn.discount_factor
        self._nstep = self._cfg.learn.nstep
        self._aux_loss_weight = self._cfg.learn.get('aux_loss_weight', None)

        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.learn.target_update_theta}
        )
        self._target_model.reset()
        self._learn_model = self._model

        self._forward_learn_cnt = 0  # count iterations

    def _shuffle_data(self, data):
        obs, next_obs = data['obs'], data['next_obs']
        batch_size, ally_num, entity_num = data['action'].shape[0], data['action'].shape[1], obs['states'].shape[1]
        action_no_attack_num = obs['action_mask'].shape[2] - entity_num
        ally_ind = torch.randperm(ally_num, device=data['action'].device)
        enemy_ind = torch.arange(ally_num, entity_num, device=data['action'].device)
        entity_ind = torch.cat([ally_ind, enemy_ind], dim=0)
        action_ind = torch.cat([torch.arange(action_no_attack_num, device=data['action'].device), action_no_attack_num + entity_ind], dim=0)
        action_ind_reverse = action_ind.sort().indices.unsqueeze(0).expand(batch_size, -1)
        obs['states'] = obs['states'].index_select(1, entity_ind)
        obs['relations'] = obs['relations'].index_select(1, entity_ind).index_select(2, entity_ind)
        obs['alive_mask'] = obs['alive_mask'].index_select(1, entity_ind)
        obs['action_mask'] = obs['action_mask'].index_select(1, ally_ind).index_select(2, action_ind)
        next_obs['states'] = next_obs['states'].index_select(1, entity_ind)
        next_obs['relations'] = next_obs['relations'].index_select(1, entity_ind).index_select(2, entity_ind)
        next_obs['alive_mask'] = next_obs['alive_mask'].index_select(1, entity_ind)
        next_obs['action_mask'] = next_obs['action_mask'].index_select(1, ally_ind).index_select(2, action_ind)
        data['action'] = action_ind_reverse.gather(1, data['action'].index_select(1, ally_ind))

    def _collect_shuffle_data(self, obs):
        batch_size, ally_num, entity_num = obs['action_mask'].shape[0], obs['action_mask'].shape[1], obs['states'].shape[1]
        action_no_attack_num = obs['action_mask'].shape[2] - entity_num
        ally_ind = torch.randperm(ally_num, device=obs['states'].device)
        ally_ind_reverse = ally_ind.sort().indices
        enemy_ind = torch.arange(ally_num, entity_num, device=obs['states'].device)
        entity_ind = torch.cat([ally_ind, enemy_ind], dim=0)
        action_ind = torch.cat([torch.arange(action_no_attack_num, device=obs['states'].device), action_no_attack_num + entity_ind], dim=0)
        action_ind_reverse = action_ind.unsqueeze(0).expand(batch_size, -1)
        obs['states'] = obs['states'].index_select(1, entity_ind)
        obs['relations'] = obs['relations'].index_select(1, entity_ind).index_select(2, entity_ind)
        obs['alive_mask'] = obs['alive_mask'].index_select(1, entity_ind)
        obs['action_mask'] = obs['action_mask'].index_select(1, ally_ind).index_select(2, action_ind)
        return action_ind_reverse, ally_ind_reverse

    def _preprocess_learn(
            self,
            data: List[Any],
            use_priority_IS_weight: bool = False,
            use_priority: bool = False,
            use_nstep: bool = False,
    ) -> dict:
        # data preprocess
        data = default_collate(data)
        data['done'] = data['done'].float()
        if use_priority_IS_weight:
            assert use_priority, "Use IS Weight correction, but Priority is not used."
        if use_priority and use_priority_IS_weight:
            data['weight'] = data['IS']
        else:
            data['weight'] = data.get('weight', torch.ones_like(data['done']).float())
        if use_nstep:
            # Reward reshaping for n-step
            reward = data['reward']
            if len(reward.shape) == 1:
                data['reward'] = reward.unsqueeze(1)
        if self._cfg.learn.shuffle:
            self._shuffle_data(data)
        return data


    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, a batch of data for training, values are torch.Tensor or \
                np.ndarray or dict/list combinations.
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Dict type data, a info dict indicated training result, which will be \
                recorded in text log and tensorboard, values are python scalar or a list of scalars.
        ArgumentsKeys:
            - necessary: ``obs``, ``next_obs``, ``action``, ``reward``, ``weight``, ``prev_state``, ``done``
        ReturnsKeys:
            - necessary: ``cur_lr``, ``total_loss``
                - cur_lr (:obj:`float`): Current learning rate
                - total_loss (:obj:`float`): The calculated loss
        """
        # preprocess
        data = self._preprocess_learn(
            data,
            use_priority=self._priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            use_nstep=True
        )
        if self._cuda:
            data = to_device(data, self._device)
        obs, action, reward, next_obs, done, weight = data['obs'], data['action'], data['reward'], data['next_obs'], data['done'], data['weight']
        batch_size, agent_num = action.shape
        self._learn_model.train()
        self._target_model.train()

        # forward
        q, _, aux_pred = self._learn_model.forward(obs, action, with_aux=True)  # [batch, agent, action]
        action_max = (q - 1e9 * (~obs['action_mask'].bool())).max(-1).indices  # [batch, agent]
        q = q.gather(2, action.unsqueeze(2)).squeeze(2)  # [batch, agent]

        with torch.no_grad():
            target_q, _ = self._target_model.forward(obs, action)  # [batch, agent, action]
            _, next_action_max = self._learn_model.forward(next_obs, {'type': 'arg_max'})  # [batch, agent]
            next_target_q, _ = self._target_model.forward(next_obs, next_action_max)  # [batch, agent, action]

            target_q = target_q.gather(2, action_max.unsqueeze(2)).squeeze(2)  # [batch, agent]
            next_target_q = next_target_q.gather(2, next_action_max.unsqueeze(2)).squeeze(2)  # [batch, agent]
            target_q = target_q.index_select(1,
                                             torch.arange(1, agent_num).to(target_q.device))  # [batch, agent - 1]
            next_target_q = (1-done.unsqueeze(1))*(next_target_q.index_select(1, torch.arange(agent_num - 1, agent_num).to(
                target_q.device)))  # [batch, 1]

        target = target_q
        next_target = torch.mm(torch.cat([reward, next_target_q], dim=1),
                               self._gamma ** torch.arange(0, self._nstep + 1).unsqueeze(1).to(reward))
        target = torch.cat([target, next_target], dim=1)
        td_error_per_sample = (next_target - q).pow(2)
        td_error_per_sample = td_error_per_sample.mean(-1)
        rl_loss = (td_error_per_sample * weight).mean()
        # aux loss
        if self.cfg.learn.get('env', 'smac') == 'smac':
            state, next_state = obs['states'][:,:,:-25], next_obs['states'][:,:,:-25] # [batch, agent, state_len - 25]
            local_pred, global_pred = aux_pred['local_pred'], aux_pred['global_pred']
            ally = state[:,:,:1] # [batch, agent, 1]
            local_label = (next_state - state)[:,:,-6:] # [batch, agent, state_len - 25 - 1 - unit_type_bits]]
            global_ally_label = (ally * local_label).sum(1) # [batch, state_len - 25 - 1 - unit_type_bits]
            global_enemy_label = ((1 - ally) * local_label).sum(1) # [batch, state_len - 25 - 1 - unit_type_bits]
            global_label = torch.cat([global_ally_label, global_enemy_label], dim=1)
            if self._cfg.learn.get('aux_label_norm', False):
                local_label = local_label/(local_label.abs().max(dim=1).values.unsqueeze(1)+1e-9)
                global_label = global_label/(global_label.abs().max(dim=1).values.unsqueeze(1)+1e-9)
            if self._cfg.learn.get('aux_class_balance', False):
                local_loss = l2_balance(local_label, (local_label - local_pred).pow(2))
            else:
                local_loss = (local_label - local_pred).pow(2).mean()
            global_loss = (global_label - global_pred).pow(2).mean()
            aux_loss = local_loss + global_loss
        elif self.cfg.learn.env == 'grf':
            state, next_state = obs['states'][:, :, 4:11], next_obs['states'][:, :, 4:11]  # [batch, agent, state_len - 25]
            local_pred, global_pred = aux_pred['local_pred'], aux_pred['global_pred']
            ally = state[:, :, :1]  # [batch, agent, 1]
            local_label = (next_state - state)  # [batch, agent, state_len - 25 - 1 - unit_type_bits]]
            if self._cfg.learn.get('aux_label_norm', False):
                local_label = local_label / (local_label.abs().max(dim=1).values.unsqueeze(1) + 1e-9)
            if self._cfg.learn.get('aux_class_balance', False):
                local_loss = l2_balance(local_label, (local_label - local_pred).pow(2))
            else:
                local_loss = (local_label - local_pred).pow(2).mean()
            aux_loss = local_loss
        # update
        # cosine aux_loss_weight
        tem = self._aux_loss_weight
        cur = min(self._forward_learn_cnt/tem.T_max, 1)
        aux_loss_weight = tem.end + 0.5*(tem.begin - tem.end) * (1 + math.cos(math.pi*cur))
        loss = rl_loss + aux_loss_weight * aux_loss
        self._optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._cfg.learn.clip_value)
        self._optimizer.step()
        if self._cfg.learn.learning_rate_type == 'cosine' and self._scheduler.last_epoch < self._scheduler.T_max:
            self._scheduler.step()
        # =============
        # after update
        # =============
        self._forward_learn_cnt += 1
        self._target_model.update(self._learn_model.state_dict())
        ret = {
            'lr': self._optimizer.param_groups[0]['lr'],
            'loss': loss.item(),
            'q': target_q.mean().item(),
            'grad_norm': grad_norm,
            'rl_loss': rl_loss.item(),
            'aux_loss': aux_loss.item(),
            'priority': td_error_per_sample.abs().tolist(),
            'aux_loss_weight': aux_loss_weight,
        }
        return ret

    def _state_dict_learn(self) -> Dict[str, Any]:
        r"""
        Overview:
            Return the state_dict of learn mode, usually including model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): the dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): the dict of policy learn state saved before.

        .. tip::
            If you want to only load some parts of model, you can simply set the ``strict`` argument in \
            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \
            complicated operation.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        # self._optimizer.load_state_dict(state_dict['optimizer'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, collect model.
            Enable the eps_greedy_sample and the hidden_state plugin.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        # self._collect_model = model_wrap(self._model, wrapper_name='eps_greedy_sample')
        self._collect_model = self._model
        # self._collect_model.reset()

    def _forward_collect(self, data: dict, eps: float) -> dict:
        r"""
        Overview:
            Forward function for collect mode with eps_greedy
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
            - eps (:obj:`float`): epsilon value for exploration, which is decayed by collected env step.
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, including at least inferred action according to input obs.
        ReturnsKeys
            - necessary: ``action``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        if self._cfg.learn.shuffle:
            action_ind_reverse, ally_ind_reverse = self._collect_shuffle_data(data)
        self._collect_model.eval()
        with torch.no_grad():
            _, action = self._collect_model.forward(data, {'type': 'eps_greedy', 'eps': eps})
            if self._cfg.learn.shuffle:
                action = action_ind_reverse.gather(1, action.index_select(1, ally_ind_reverse))
            output = {'action': action}
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Overview:
            For a given trajectory(transitions, a list of transition) data, process it into a list of sample that \
            can be used for training directly. A train sample can be a processed transition(DQN with nstep TD) \
            or some continuous transitions(DRQN).
        Arguments:
            - data (:obj:`List[Dict[str, Any]`): The trajectory data(a list of transition), each element is the same \
                format as the return value of ``self._process_transition`` method.
        Returns:
            - samples (:obj:`dict`): The list of training samples.

        .. note::
            We will vectorize ``process_transition`` and ``get_train_sample`` method in the following release version. \
            And the user can customize the this data processing procecure by overriding this two methods and collector \
            itself.
        """
        data = get_nstep_return_data(data, self._nstep, gamma=self._gamma)
        return get_train_sample(data, self._unroll_len)

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - model_output (:obj:`dict`): Output of collect model, including at least ['action', 'prev_state']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done']\
                (here 'obs' indicates obs after env step).
        Returns:
            - transition (:obj:`dict`): Dict type transition data, including 'obs', 'next_obs', 'prev_state',\
                'action', 'reward', 'done'
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': model_output['action'],
            'reward': timestep.reward,
        }
        if self.cfg.learn.get('env', 'smac') == 'smac':
            transition['done'] = timestep.info['battle_won'] | timestep.info['battle_lost'] | timestep.info['draw']
        elif self.cfg.learn.env == 'grf':
            transition['done'] = timestep.done,
        return transition

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval model with argmax strategy and the hidden_state plugin.
        """
        self._eval_model = self._model

    def _forward_eval(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.
        ReturnsKeys
            - necessary: ``action``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            _, action = self._eval_model.forward(data, action={'type': 'arg_max'})
            output = {'action': action}
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default model setting for demonstration.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): model name and mode import_names

        .. note::
            The user can define and use customized network model but must obey the same inferface definition indicated \
            by import_names path. For QMIX, ``ding.model.qmix.qmix``
        """
        if self.cfg.learn.get('env', 'smac') == 'smac':
            return 'smac_ace', ['ding.model.template.smac_ace']
        elif self.cfg.learn.env == 'grf':
            return 'grf_ace', ['ding.model.template.grf_ace']

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        ret = ['lr', 'loss', 'rl_loss', 'aux_loss', 'q', 'grad_norm', 'aux_loss_weight']
        return ret
