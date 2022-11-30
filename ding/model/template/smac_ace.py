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
            embed_num: int,  # 6 [dead, stop, move_n, move_s, move_e, move_w]
            hidden_len: int,
            update_state: bool = True,
    ) -> None:
        super(DecisionEncoder, self).__init__()
        self.embed_num = embed_num
        self.update_state = update_state
        self._action_embed = nn.Parameter(
            torch.zeros(1, 1, embed_num, hidden_len))  # [batch, agent, no_attack_action_num, hidden]
        nn.init.kaiming_normal_(self._action_embed, mode='fan_out')
        self._decision_encoder = MLP(hidden_len, hidden_len, 2 * hidden_len, 2, activation=nn.ReLU(inplace=True))
        self._logit_encoder = nn.Sequential(
            nn.Linear(2 * hidden_len, hidden_len),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_len, 1),
        )

    def forward(self, i, state, action_embed, alive_mask, action_mask, action):
        batch_size, agent_num, hidden_len, embed_num, device = state.shape[0], state.shape[1], \
                                                               state.shape[2], self._action_embed.shape[
                                                                   2], state.device
        agent_id = torch.LongTensor([i]).to(device)
        active_self_embed = self._action_embed.expand(batch_size, -1, -1, -1)
        passive_none_embed = torch.zeros(batch_size, agent_num, 1, hidden_len).to(device)
        passive_map = torch.cat(
            [torch.zeros(agent_num, embed_num).long().to(device),
             torch.diag(torch.ones(agent_num)).long().to(device)],
            dim=1).view(1, -1, embed_num + agent_num, 1).expand(batch_size, -1, -1,
                                                                hidden_len*2)  # [batch, agent, action_num, hidden]
        # get action embed
        active_embed, passive_embed = action_embed.chunk(2, dim=-1)  # [batch, agent, hidden], [batch, agent, hidden]
        active_embed_ = torch.cat([active_self_embed, active_embed.unsqueeze(1)],
                                  dim=2)  # [batch, 1, active_action, hidden] embed for active agent
        active_embed = active_embed_.scatter_add(2,
                                                 embed_num + agent_id.view(1, 1, 1, 1).expand(batch_size, 1, 1, hidden_len),
                                                 passive_embed.unsqueeze(1).index_select(2, agent_id))
        passive_embed = torch.cat([passive_none_embed, passive_embed.unsqueeze(2)],
                                  dim=2)  # [batch, agent, passive_action(2), hidden] embed for passive agent
        # get decision embed
        active_state = state.index_select(1, agent_id).unsqueeze(2) + active_embed
        passive_state = state.unsqueeze(2) + passive_embed
        active_decision = self._decision_encoder(active_state)    # [batch, 1, active_action, hidden]
        passive_decision = self._decision_encoder(passive_state)  # [batch, agent, passive_action(2), hidden]
        decision = passive_decision.gather(2, passive_map)        # [batch, agent, action_number, hidden]
        decision.scatter_(1, agent_id.view(1, 1, 1, 1).expand(batch_size, 1, embed_num + agent_num, 2 * hidden_len),
                          active_decision)
        # get logit
        decision_avr, decision_max = decision.chunk(2, dim=-1)  # [batch, agent, action_num, hidden]
        decision_avr = (decision_avr * alive_mask.unsqueeze(-1).unsqueeze(-1)).mean(1)  # [batch, action_num, hidden]
        decision_max = (decision_max * alive_mask.unsqueeze(-1).unsqueeze(-1)).max(1).values  # [batch, action_num, hidden]
        decision = torch.cat([decision_avr, decision_max], dim=-1)
        logit = self._logit_encoder(decision).squeeze(-1)  # [batch, action_num]
        # get action
        if isinstance(action, dict):
            action = ActionSampler(logit, action_mask, action)  # [batch]
        # get updated state
        if self.update_state:
            active_embed = active_embed_.gather(2, action.view(-1, 1, 1, 1).expand(-1, -1, -1, hidden_len))
            state = state.scatter_add(1, agent_id.view(1, 1, 1).expand(batch_size, 1, hidden_len), active_embed.squeeze(2))
            passive_map = passive_map.gather(2, action.view(-1, 1, 1, 1).expand(-1, agent_num, 1,
                                                                                hidden_len))  # [batch, agent, 1, hidden]
            passive_embed = passive_embed.gather(2, passive_map).squeeze(2)
            state = state + passive_embed
        decision = decision.gather(1, action.view(-1, 1, 1).expand(-1, -1, hidden_len*2))
        return state, decision, logit, action


@MODEL_REGISTRY.register('smac_ace')
class SMACACE(nn.Module):
    def __init__(
            self,
            agent_num: int,
            embed_num: int,
            state_len: int,
            relation_len: int,
            hidden_len: int,
            local_pred_len: int, # 6
            global_pred_len: int, # 12
            update_state: bool = True,
    ) -> None:
        super(SMACACE, self).__init__()
        self.agent_num = agent_num
        self._action_encoder = MLP(2 * hidden_len + hidden_len + hidden_len, hidden_len, 2 * hidden_len, 2,
                                   activation=nn.ReLU(inplace=True))
        self._state_encoder = MLP(state_len, hidden_len, hidden_len, 2, activation=nn.ReLU(inplace=True))
        self._relation_encoder = MLP(hidden_len + relation_len, hidden_len, 2 * hidden_len, 2,
                                     activation=nn.ReLU(inplace=True))
        self._relation_aggregator = RelationAggregator(hidden_len, 2 * hidden_len)
        self._decision_encoder = DecisionEncoder(embed_num, hidden_len, update_state)
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
        relation = obs['relations']  # [batch, entity_num, relation_len]
        alive_mask = obs['alive_mask']  # [batch, entity_num]
        state = self._state_encoder(state)
        relation = self._relation_encoder(
            torch.cat([relation, state.unsqueeze(1).expand(-1, relation.shape[1], -1, -1)], dim=-1))
        state = self._relation_aggregator(state, relation, alive_mask)
        action_embed = self._action_encoder(torch.cat(
            [relation, state.unsqueeze(1).expand(-1, relation.shape[1], -1, -1),
             state.unsqueeze(2).expand(-1, -1, relation.shape[1], -1)], dim=-1))
        return state, action_embed

    def forward(self, obs, action, with_aux=False) -> dict:
        alive_mask = obs['alive_mask']  # [batch, entity_num]
        action_mask = obs['action_mask']  # [batch, agent_num, action_len]
        state, action_embed = self.encode_state(obs)
        logit = []
        action_ = []
        for i in range(self.agent_num):
            state, decision, lgt, act = self._decision_encoder(i, state, action_embed.select(1, i), alive_mask,
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
