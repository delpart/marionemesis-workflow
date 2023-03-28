# import cv2
import numpy as np
import torch
# from torch import nn
# from torchrl.objectives import ClipPPOLoss
# from torchrl.objectives.value import GAE
# from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
# from torchrl.data.replay_buffers import ReplayBuffer
# from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
# from torchrl.data.replay_buffers.storages import LazyTensorStorage
# from tensordict import TensorDict
# from tensordict.nn import TensorDictModule
# from tensordict.nn.distributions import NormalParamExtractor
# from tqdm import trange

# from envs import MarioNemesis

num_cells = 64
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4
lr = 3e-4
max_grad_norm = 1.0

# class NemesisBackend(nn.Module):
#     def __init__(self):
#         super(NemesisBackend, self).__init__()
#         self.net1 = nn.Sequential(
#             nn.LazyLinear(num_cells),
#             nn.Tanh(),
#             nn.LazyLinear(num_cells),
#             nn.Tanh()
#         )
#         self.net2 = nn.Sequential(
#             nn.LazyLinear(num_cells),
#             nn.Tanh(),
#             nn.LazyLinear(num_cells),
#             nn.Tanh()
#         )
#         self.net3 = nn.Sequential(
#             nn.LazyLinear(num_cells),
#             nn.Tanh(),
#             nn.LazyLinear(num_cells),
#             nn.Tanh()
#         )
#
#     def forward(self, obs):
#         if len(obs.shape) > 1:
#             level = obs[:, :33 * 16]
#             pos = obs[:, 33 * 16:33 * 16 + 2]
#             elapsed = obs[:, 33 * 16 + 2:33 * 16 + 2 + 1]
#         else:
#             level = obs[:33 * 16]
#             pos = obs[33 * 16:33 * 16 + 2]
#             elapsed = obs[33 * 16 + 2:33 * 16 + 2 + 1]
#         level = nn.functional.one_hot(level.long(), 7).float()
#         level = self.net1(level.reshape(-1, 528*7))
#         pos = self.net2(pos).reshape(level.shape)
#         elapsed = self.net3(elapsed).reshape(level.shape)
#         return torch.concat([level, pos, elapsed], dim=1)# if len(obs.shape) > 1 else torch.concat([level, pos, elapsed], dim=0)
#
#
# class ActorNemesisNet(nn.Module):
#     def __init__(self):
#         super(ActorNemesisNet, self).__init__()
#         self.backend = NemesisBackend()
#         self.out = nn.Sequential(
#             nn.LazyLinear(num_cells),
#             nn.Tanh(),
#             nn.LazyLinear(2*16*7),
#             NormalParamExtractor()
#         )
#         self.scale = 1
#
#     def forward(self, obs):
#         return self.out(self.backend(obs))
#
#
# class ValueNemesisNet(nn.Module):
#     def __init__(self):
#         super(ValueNemesisNet, self).__init__()
#         self.backend = NemesisBackend()
#         self.out = nn.Sequential(
#             nn.LazyLinear(num_cells),
#             nn.Tanh(),
#             nn.LazyLinear(1)
#         )
#
#     def forward(self, obs):
#         return self.out(self.backend(obs))
#
#
# class ActorMarioNet(nn.Module):
#     def __init__(self):
#         super(ActorMarioNet, self).__init__()
#         self.net = nn.Sequential(
#             nn.LazyLinear(num_cells),
#             nn.Tanh(),
#             nn.LazyLinear(num_cells),
#             nn.Tanh(),
#             nn.LazyLinear(num_cells),
#             nn.Tanh(),
#             nn.LazyLinear(2 * 3),
#             NormalParamExtractor(),
#         )
#
#     def forward(self, obs):
#         return self.net(obs)
#
#
# def train_ppo():
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     n_steps = 1000
#     epochs = 100000
#
#     env = MarioNemesis()
#
#     actor_net_mario = ActorMarioNet()
#     policy_module_mario = TensorDictModule(actor_net_mario, in_keys=['observation'], out_keys=['loc', 'scale'])
#     policy_module_mario = ProbabilisticActor(
#         module=policy_module_mario,
#         in_keys=['loc', 'scale'],
#         distribution_class=TanhNormal,
#         distribution_kwargs={
#             'min': -1,
#             'max': 1,
#         },
#         return_log_prob=True,
#     )
#     value_net_mario = nn.Sequential(
#         nn.LazyLinear(num_cells, device=device),
#         nn.Tanh(),
#         nn.LazyLinear(num_cells, device=device),
#         nn.Tanh(),
#         nn.LazyLinear(num_cells, device=device),
#         nn.Tanh(),
#         nn.LazyLinear(1, device=device),
#     )
#     value_module_mario = ValueOperator(
#         module=value_net_mario,
#         in_keys=['observation'],
#     )
#
#     actor_net_nemesis = ActorNemesisNet()
#     policy_module_nemesis = TensorDictModule(actor_net_nemesis, in_keys=['observation'], out_keys=['loc', 'scale'])
#     policy_module_nemesis = ProbabilisticActor(
#         module=policy_module_nemesis,
#         in_keys=['loc', 'scale'],
#         distribution_class=TanhNormal,
#         return_log_prob=True,
#     )
#     value_net_nemesis = ValueNemesisNet()
#     value_module_nemesis = ValueOperator(
#         module=value_net_nemesis,
#         in_keys=['observation'],
#     )
#
#     # initialize parameters (using LazyLinear)
#     obs, _ = env.reset()
#     obs_mario = torch.tensor(obs['mario'].astype(np.float32).reshape(1, -1) / 255.).to(device)
#     td_mario_tmp = TensorDict({'observation': obs_mario}, batch_size=[1])
#     policy_module_mario(td_mario_tmp)
#     value_module_mario(td_mario_tmp)
#     obs_nemesis = torch.concat(
#         [torch.tensor(obs['nemesis']['level'].reshape(-1)), torch.tensor(obs['nemesis']['elapsed']),
#          torch.tensor(obs['nemesis']['pos_mario'])]).unsqueeze(0)
#     obs_nemesis = obs_nemesis.float()
#     td_nemesis_tmp = TensorDict({'observation': obs_nemesis}, batch_size=[1])
#     policy_module_nemesis(td_nemesis_tmp)
#     value_module_nemesis(td_nemesis_tmp)
#
#
#     advantage_module_mario = GAE(
#         gamma=gamma, lmbda=lmbda, value_network=value_module_mario, average_gae=True
#     )
#     advantage_module_nemesis = GAE(
#         gamma=gamma, lmbda=lmbda, value_network=value_module_nemesis, average_gae=True
#     )
#
#     loss_module_mario = ClipPPOLoss(
#         actor=policy_module_mario,
#         critic=value_module_mario,
#         advantage_key="advantage",
#         clip_epsilon=clip_epsilon,
#         entropy_bonus=bool(entropy_eps),
#         entropy_coef=entropy_eps,
#         # these keys match by default but we set this for completeness
#         value_target_key=advantage_module_mario.value_target_key,
#         critic_coef=1.0,
#         gamma=0.99,
#         loss_critic_type="smooth_l1",
#     )
#     loss_module_nemesis = ClipPPOLoss(
#         actor=policy_module_nemesis,
#         critic=value_module_nemesis,
#         advantage_key="advantage",
#         clip_epsilon=clip_epsilon,
#         entropy_bonus=bool(entropy_eps),
#         entropy_coef=entropy_eps,
#         # these keys match by default but we set this for completeness
#         value_target_key=advantage_module_nemesis.value_target_key,
#         critic_coef=1.0,
#         gamma=0.99,
#         loss_critic_type="smooth_l1",
#     )
#
#     optim_mario = torch.optim.Adam(loss_module_mario.parameters(), lr)
#     optim_nemesis = torch.optim.Adam(loss_module_nemesis.parameters(), lr)
#
#     replay_buffer_mario = ReplayBuffer(
#         storage=LazyTensorStorage(int(8e4)),
#         sampler=SamplerWithoutReplacement()
#     )
#     replay_buffer_nemesis = ReplayBuffer(
#         storage=LazyTensorStorage(int(8e4)),
#         sampler=SamplerWithoutReplacement()
#     )
#
#     epochs = trange(epochs)
#     next_obs, info = env.reset()
#     for epoch in epochs:
#         done = torch.tensor([0])
#         skip_nemesis = False
#         with torch.no_grad():
#             td_nemesis = []
#             td_mario = []
#             for k in trange(n_steps, leave=False):
#                 obs = next_obs
#                 obs_mario = torch.tensor(obs['mario'].astype(np.float32).reshape(1, -1)).to(device)
#                 td_mario_tmp = TensorDict({'observation': obs_mario, 'done': done, 'reward': 0.,  'next': TensorDict({'observation': torch.zeros_like(obs_mario)}, batch_size=[1])}, batch_size=[1])
#                 td_mario_tmp = policy_module_mario(td_mario_tmp)
#                 obs_nemesis = None
#                 cv2.imshow('mario', obs['mario'].T)
#                 cv2.waitKey(1)
#                 if info['step_nemesis']:
#                     obs_nemesis = torch.concat([torch.tensor(obs['nemesis']['level'].reshape(-1)), torch.tensor(obs['nemesis']['elapsed']), torch.tensor(obs['nemesis']['pos_mario'])]).unsqueeze(0)
#                     obs_nemesis = obs_nemesis.float()
#                     td_nemesis_tmp = TensorDict({'observation': obs_nemesis, 'done': done, 'reward': 0, 'next': TensorDict({'observation': torch.zeros_like(obs_nemesis)}, batch_size=[1])}, batch_size=[1])
#                     td_nemesis_tmp = policy_module_nemesis(td_nemesis_tmp)
#                 next_obs, reward, done, truncated, info = env.step(td_mario_tmp['action'].cpu().numpy().reshape(-1), None if obs_nemesis is None else td_nemesis_tmp['action'].cpu().numpy().reshape(-1))
#
#                 td_mario_tmp['reward'] = torch.tensor(reward['mario']).unsqueeze(0)
#                 td_mario_tmp['done'] = torch.tensor([done]).unsqueeze(0)
#                 td_mario_tmp['next']['observation'] = torch.tensor(next_obs['mario'].astype(np.float32).reshape(1, -1)/255.).to(device)
#                 td_mario.append(td_mario_tmp)
#
#                 if obs_nemesis is not None:
#                     obs_nemesis = torch.concat(
#                         [torch.tensor(next_obs['nemesis']['level'].reshape(-1)), torch.tensor(next_obs['nemesis']['elapsed']),
#                          torch.tensor(next_obs['nemesis']['pos_mario'])]).unsqueeze(0)
#                     obs_nemesis = obs_nemesis.float()
#                     td_nemesis_tmp['reward'] = torch.tensor(reward['nemesis']).unsqueeze(0)
#                     td_nemesis_tmp['done'] = torch.tensor([done]).unsqueeze(0)
#                     td_nemesis_tmp['next']['observation'] = obs_nemesis
#                     td_nemesis.append(td_nemesis_tmp)
#                 if done:
#                     next_obs, info = env.reset()
#         td_mario = torch.cat(td_mario, 0)
#         advantage_module_mario(td_mario)
#         replay_buffer_mario.extend(td_mario.reshape(-1).cpu())
#
#         if len(td_nemesis) <= 0:
#             skip_nemesis = True
#
#         if not skip_nemesis:
#             td_nemesis = torch.cat(td_nemesis, 0)
#             advantage_module_nemesis(td_nemesis)
#             replay_buffer_nemesis.extend(td_nemesis.reshape(-1).cpu())
#         for _ in range(10):
#             batch_mario, *_ = replay_buffer_mario.sample(32)
#             losses_mario = loss_module_mario(batch_mario.to(device))
#
#             loss_mario = losses_mario['loss_objective'] + losses_mario['loss_critic'] + losses_mario['loss_entropy']
#
#             optim_mario.zero_grad()
#             loss_mario.backward()
#             torch.nn.utils.clip_grad_norm_(loss_module_mario.parameters(), max_grad_norm)
#             optim_mario.step()
#
#             if not skip_nemesis:
#                 batch_nemesis, *_ = replay_buffer_nemesis.sample(32)
#                 losses_nemesis = loss_module_nemesis(batch_nemesis.to(device))
#                 loss_nemesis = losses_nemesis['loss_objective'] + losses_nemesis['loss_critic'] + losses_nemesis[
#                     'loss_entropy']
#                 optim_nemesis.zero_grad()
#                 loss_nemesis.backward()
#                 torch.nn.utils.clip_grad_norm_(loss_module_nemesis.parameters(), max_grad_norm)
#                 optim_nemesis.step()
#
#         epochs.set_description(f'Mean rewards: {td_mario["reward"].float().mean().item()} - {td_nemesis["reward"].float().mean().item() if not skip_nemesis else 0}')
#     torch.save(actor_net_nemesis, 'actor_nemesis.pt')
#     torch.save(actor_net_mario, 'actor_mario.pt')
#     torch.save(value_net_mario, 'value_mario.pt')
#     torch.save(value_net_nemesis, 'value_nemesis.pt')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--play', action='store_true')
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()

    if args.play:
        import pygame as pg
        import sys
        from envs import Level, MarioUNet, MarioDiffusion, char_to_int
        from pathlib import Path

        pg.init()

        display = pg.display.set_mode((512, 256))
        clock = pg.time.Clock()

        model = MarioUNet()
        checkpoints = Path('assets/checkpoints/').glob('*chkpt.pt')

        latest_chkpt = max(checkpoints, key=lambda f: int(f.stem.split('_')[0]))
        print(f'loading {latest_chkpt}')
        weights = torch.load(latest_chkpt)
        model.load_state_dict(weights)

        diffusion = MarioDiffusion(noise_steps=10, device='cuda' if torch.cuda.is_available() else 'cpu')

        for _ in range(3):
            level = Level(display)
            level.player.sprite.human_control = True
            while True:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        pg.quit()
                        sys.exit()

                dead, add_row = level.step()

                if dead:
                    break

                if add_row:
                    level_array = np.zeros((1, 33, 16), dtype=int)

                    for i, col in enumerate(level.map_rows):
                        for j, c in enumerate(col):
                            level_array[0, i, j] = char_to_int[c]
                    level_tensor = torch.from_numpy(level_array[:,1:,:]).type(torch.float)
                    level.append_row(diffusion.sample_continuation(model, level_tensor))

                pg.display.update()
                clock.tick(30)
    else:
        from envs import train
        train(args.output_dir)



    # train_ppo()
    #
    # mario_nemesis = MarioNemesis()
    # # mario_nemesis.run()
    # obs, info = mario_nemesis.reset()
    # while True:
    #     cv2.imshow('MarioNemesis', obs['mario'].transpose(1, 0, 2))
    #
    #     mario_actions = np.zeros(3)
    #
    #     key = cv2.waitKey(30)
    #     if key == ord('a'):
    #         mario_actions[0] = 1
    #     if key == ord('d'):
    #         mario_actions[1] = 1
    #     if key == ord('w'):
    #         mario_actions[2] = 1
    #
    #     nemesis_mario_actions = None
    #     if info['step_nemesis']:
    #         nemesis_mario_actions = '-'*14 + 'XX'
    #
    #     obs, reward, done, truncated, info = mario_nemesis.step(mario_actions, nemesis_mario_actions)
    #
    #     if done:
    #         obs, info = mario_nemesis.reset()


