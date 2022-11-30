from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.utils import MODEL_REGISTRY
from ding.torch_utils.network.nn_module import MLP


class RelationAggregator(nn.Module):
    def __init__(
            self,
            state_len: int,
            relation_len: int,
    ) -> None:
        super(RelationAggregator, self).__init__()
        self._state_encoder = nn.Sequential(
            nn.Linear(state_len + relation_len, state_len),
            nn.ReLU(inplace=True),
        )

    def forward(self, state, relation, alive_mask):
        relation_avr, relation_max = relation.chunk(2, dim=-1)
        relation_avr = (relation_avr * alive_mask.unsqueeze(1).unsqueeze(-1)).mean(-2)
        relation_max = (relation_max * alive_mask.unsqueeze(1).unsqueeze(-1)).max(-2).values
        state = self._state_encoder(torch.cat([state, relation_avr, relation_max], dim=-1))
        return state


def ActionSampler(logit, action_mask, cfg):
    if cfg['type'] == 'arg_max':
        return (logit - 1e9 * (~action_mask.bool())).max(-1).indices
    elif cfg['type'] == 'eps_greedy':
        action_max = (logit - 1e9 * (~action_mask.bool())).max(-1).indices
        action_rnd = torch.multinomial(action_mask.float(), 1).squeeze(-1)
        rand_mask = (torch.rand(logit.shape[:-1]) < cfg['eps']).to(logit.device)
        return (rand_mask.float() * action_rnd + (1 - rand_mask.float()) * action_max).long()
    elif cfg['type'] == 'boltzman':
        action_max = (logit - 1e9 * (~action_mask.bool())).max(-1).indices
        action_rnd = torch.multinomial(action_mask.float(), 1).squeeze(-1)
        action_bzm = torch.multinomial(logit.softmax(-1) * action_mask.float(), 1).squeeze(-1)
        rand_mask = (torch.rand(logit.shape[:-1]) < cfg['eps']).to(logit.device)
        btzm_mask = (torch.rand(logit.shape[:-1]) < cfg['bzm']).to(logit.device)
        return (rand_mask.float() * (btzm_mask.float() * action_bzm + (1 - btzm_mask.float()) * action_rnd) + (
                1 - rand_mask.float()) * action_max).long()


class DecisionEncoder(nn.Module):
    def __init__(
            self,
            hidden_len: int,
    ) -> None:
        super(DecisionEncoder, self).__init__()
        self._decision_encoder = MLP(hidden_len, hidden_len, 2 * hidden_len, 2, activation=nn.ReLU(inplace=True))
        self._logit_encoder = nn.Sequential(
            nn.Linear(2 * hidden_len, hidden_len),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_len, 1),
        )

    def forward(self, i, state, active_embed, passive_embed, alive_mask, action_mask, action):
        batch_size, agent_num, hidden_len, device = state.shape[0], state.shape[1], \
                                                               state.shape[2], state.device
        agent_id = torch.LongTensor([i,]).to(device)
        ball_id = torch.LongTensor([agent_num-1,]).to(device)
        # get decision embed
        decision = self._decision_encoder(state).unsqueeze(2).repeat(1, 1, 19, 1) # [batch, agent, action_number, 2 * hidden]
        active_state = state.index_select(1, agent_id).unsqueeze(2) + active_embed # [batch, 1, action_number, hidden]
        active_decision = self._decision_encoder(active_state) # [batch, 1, action_number, 2 * hidden]
        decision.scatter_(1, agent_id.view(1, 1, 1, 1).expand(batch_size, 1, 19, 2 * hidden_len), active_decision)# [batch, agent, action_number, 2 * hidden]
        passive_state = state.index_select(1, ball_id).unsqueeze(2) + passive_embed # [batch, 1, action_number, hidden]
        passive_decision = self._decision_encoder(passive_state) # [batch, 1, action_number, 2 * hidden]
        decision.scatter_(1, ball_id.view(1, 1, 1, 1).expand(batch_size, 1, 19, 2 * hidden_len), passive_decision)# [batch, agent, action_number, 2 * hidden]
        # get logit
        decision_avr, decision_max = decision.chunk(2, dim=-1)  # [batch, agent, action_num, hidden]
        decision_avr = (decision_avr * alive_mask.unsqueeze(-1).unsqueeze(-1)).mean(1)  # [batch, action_num, hidden]
        decision_max = (decision_max * alive_mask.unsqueeze(-1).unsqueeze(-1)).max(1).values  # [batch, action_num, hidden]
        decision = torch.cat([decision_avr, decision_max], dim=-1)
        logit = self._logit_encoder(decision).squeeze(-1)  # [batch, action_num]
        # get action
        if isinstance(action, dict):
            action = ActionSampler(logit, action_mask, action)  # [batch]
        active_embed = active_embed.gather(2, action.view(-1, 1, 1, 1).expand(-1, -1, -1, hidden_len)).squeeze(2) # [batch, 1, hidden]
        state = state.scatter_add(1, agent_id.view(1, 1, 1).expand(batch_size, 1, hidden_len), active_embed)
        passive_embed = passive_embed.gather(2, action.view(-1, 1, 1, 1).expand(-1, -1, -1, hidden_len)).squeeze(2) # [batch, 1, hidden]
        state = state.scatter_add(1, ball_id.view(1, 1, 1).expand(batch_size, 1, hidden_len), passive_embed)
        decision = decision.gather(1, action.view(-1, 1, 1).expand(-1, -1, hidden_len * 2))
        return state, decision, logit, action


@MODEL_REGISTRY.register('grf_ace')
class GRFACE(nn.Module):
    def __init__(
            self,
            agent_num: int,
            state_len: int,
            relation_len: int,
            hidden_len: int,
            local_pred_len: int, # 7
            global_pred_len: int, # 14
    ) -> None:
        super(GRFACE, self).__init__()
        self.agent_num = agent_num
        self._action_encoder = MLP(hidden_len, hidden_len, 19 * hidden_len, 2,
                                   activation=nn.ReLU(inplace=True))
        self._action_embed = nn.Parameter(
            torch.zeros(1, 1, 19, hidden_len))  # [batch, agent, 19, hidden]
        nn.init.kaiming_normal_(self._action_embed, mode='fan_out')
        self._state_encoder = MLP(state_len, hidden_len, hidden_len, 2, activation=nn.ReLU(inplace=True))
        self._relation_encoder = MLP(hidden_len + relation_len, hidden_len, 2 * hidden_len, 2,
                                     activation=nn.ReLU(inplace=True))
        self._relation_aggregator = RelationAggregator(hidden_len, 2 * hidden_len)
        self._decision_encoder = DecisionEncoder(hidden_len)
        self._local_predictor = nn.Sequential(
            nn.Linear(hidden_len, hidden_len // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_len // 2, local_pred_len),
        )
        self._global_predictor = nn.Sequential(
            nn.Linear(2 * hidden_len, hidden_len),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_len, global_pred_len),
        )

    def encode_state(self, obs) -> dict:
        state = obs['states']  # [batch, entity_num, state_len]
        relation = obs['relations']  # [batch, entity_num, entity_num, relation_len]
        alive_mask = obs['alive_mask']  # [batch, entity_num]
        state = self._state_encoder(state)
        relation = self._relation_encoder(
            torch.cat([relation, state.unsqueeze(1).expand(-1, relation.shape[1], -1, -1)], dim=-1))
        state = self._relation_aggregator(state, relation, alive_mask)
        return state

    def forward(self, obs, action, with_aux=False) -> dict:
        own_mask = obs['states'][:,:,4]
        alive_mask = obs['alive_mask']  # [batch, entity_num]
        action_mask = obs['action_mask']  # [batch, agent_num, action_len]
        state = self.encode_state(obs)
        logit = []
        action_ = []
        for i in range(self.agent_num):
            active_embed = self._action_embed.expand(state.shape[0], -1, -1, -1)
            passive_embed = self._action_encoder(state.select(1, i)).view(state.shape[0], 1, 19, -1) * own_mask[:, i].view(-1, 1, 1, 1)
            state, decision, lgt, act = self._decision_encoder(i, state, active_embed, passive_embed, alive_mask,
                                                     action_mask.select(1, i),
                                                     action.select(1, i) if not isinstance(action, dict) else action)
            logit.append(lgt.unsqueeze(1))
            action_.append(act.unsqueeze(1))
        if with_aux:
            local_pred = self._local_predictor(state)
            global_pred = self._global_predictor(decision).squeeze(1)
            aux_pred = {'local_pred': local_pred, 'global_pred': global_pred}
            return torch.cat(logit, dim=1), torch.cat(action_, dim=1), aux_pred
        else:
            return torch.cat(logit, dim=1), torch.cat(action_, dim=1)
