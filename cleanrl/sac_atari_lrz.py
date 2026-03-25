import gymnasium as gym
import numpy as np
import torch
from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import dataclass
import tyro
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from cleanrl_utils.buffers import ReplayBuffer
import random



@dataclass
class Args:
    exp_name : str = Path(__file__).stem
    """实验名称"""
    num_envs : int = 4
    """并行环境数量"""
    track : bool = False
    """是否开启wandb追踪"""
    env_id : str = "BeamRiderNoFrameskip-v4"
    """模拟环境名称"""
    track : bool = True
    
    wandb_project_name : str = Path(__file__).stem

    wandb_entity : str = None
    """WandB 团队协作空间名称"""

    seed : int = 1

    cudnn_deterministic : bool = True
    """是否开启 CUDA 确定性算法"""

    cuda : bool = False

    capture_video : bool = True

    q_lr : float = 1e-3

    policy_lr : float = 1e-3

    autotune : bool = True

    target_entropy_scale : float = 0.95

    alpha : float = 0.2

    buffer_size : int = 400000

    total_time_step : int = 100000

    learning_start : int = 3000

    batch_size : int = 64

    update_frequency : int = 100

    gamma : float = 0.98

    target_update_frequency : int = 8000

    tau : float = 0.95





def env_make(env_id,seed,idx,capture_video,run_name):
    def thunk():   #使用延迟触发
        if capture_video and idx == 0:
            env = gym.make(env_id,render_mode = "rgb_array")  # 使用 rgb 图像存储模式
            env  = gym.wrappers.RecordVideo(env,f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env =  gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env,noop_max = 30)
        env = MaxAndSkipEnv(env,skip = 4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env,(84,84))   #重新缩放输入图像尺寸
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env,4)

        env.action_space.seed(seed)

        return env
    return thunk

def layer_init(layer,bias_const = 0.0):
    nn.init.kaiming_normal_(layer.weight) # 对于卷积使用 kaiming 初始化 ； Deep RL 中对于全连接一般采用正交初始化
    nn.init.constant_(layer.bias,bias_const)
    return layer

def create_layer(layers_list,activation = nn.ReLU()):
    layers = []
    for i in range(len(layers_list)):
        layer = nn.Conv2d(*(layers_list[i]))
        layer = layer_init(layer)
        layers.append(layer)
        if i<len(layers_list)-1:
            layers.append(activation)
    layers.append(nn.Flatten())
    return nn.Sequential(*layers)


    
class SoftQNetwork(nn.Module):
    def __init__(self,envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.conv = create_layer([[obs_shape[0],32,8,4],
                                [32,64,4,2],
                                [64,64,3,1]])
        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1,*obs_shape)).shape[1]
        self.fc1 = layer_init(nn.Linear(output_dim,512))
        self.fc_q = layer_init(nn.Linear(512,envs.single_action_space.n))
    def forward(self,x):
        x = nn.ReLU()(self.conv(x/255.0))
        x = nn.ReLU()(self.fc1(x))
        q_vals = self.fc_q(x)
        return q_vals

class Actor(nn.Module):
    def __init__(self,envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.conv = create_layer([[obs_shape[0],32,8,4],
                                [32,64,4,2],
                                [64,64,3,1]])
        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 512))
        self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

    def forward(self,x):
        x = nn.ReLU()(self.conv(x))
        x = nn.ReLU()(self.fc1(x))
        logits = self.fc_logits(x)
        return logits
    def get_action(self,x):
        logits = self(x/255.0)
        policy_dist = Categorical(logits = logits)
        action = policy_dist.sample()
        action_p = policy_dist.probs
        logp = nn.LogSoftmax(dim = 1)(logits)
        return action ,action_p, logp
if __name__ == "__main__":
        args = tyro.cli(Args)
        run_name = f"{args.env_id}_{args.exp_name}_{args.seed}__{int(time.time())}"
        if args.track:
            import wandb
            wandb.init(
                entity = args.wandb_entity,
                project = args.wandb_project_name,
                sync_tensorboard = True,
                config = vars(args),
                name = run_name,
                monitor_gym = True,
                save_code = True

            )
            writer = SummaryWriter(f"runs/{run_name}")
            writer.add_text(
                "hyperparameters",
                "|param|value|\n|--|--|--\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
            )

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.cudnn_deterministic
        device=torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        envs=gym.vector.SyncVectorEnv([env_make(args.env_id,args.seed,idx,args.capture_video,run_name=run_name) for idx in range(args.num_envs)])
        assert isinstance(envs.single_action_space,gym.spaces.Discrete), "only Discrete Space is supported"

        actor=Actor(envs).to(device)
        qf1=SoftQNetwork(envs).to(device)
        qf2=SoftQNetwork(envs).to(device)
        qf1_target=SoftQNetwork(envs).to(device)
        qf2_target=SoftQNetwork(envs).to(device)
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())

        qf1_optimizer=torch.optim.Adam(qf1.parameters(),args.q_lr)
        qf2_optimizer=torch.optim.Adam(qf2.parameters(),lr=args.q_lr)
        actor_optimizer=torch.optim.Adam(actor.parameters(),lr=args.policy_lr)

        if args.autotune:
            target_entropy=args.target_entropy_scale * torch.log(torch.as_tensor(envs.single_action_space.n,device=device))
            log_alpha=torch.zeros(1,device=device,requires_grad=True)
            alpha=torch.exp(log_alpha)
            alpha_optimizer=torch.optim.Adam([log_alpha],lr=args.q_lr)
        else:
            alpha=args.alpha
        rb=ReplayBuffer(
            int(args.buffer_size),
            envs.single_observation_space,
            envs.single_action_space,
            device,
            n_envs=args.num_envs,
            handle_timeout_termination=False

        )

        start_time=time.time()
        obs,_=envs.reset(seed=args.seed)

        for i in range(args.total_time_step):
            if i < args.learning_start:
                actions=np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
                # 冷启动策略
            else:
                actions,_,_=actor.get_action(torch.as_tensor(obs,device=device))
                actions = actions.detach().cpu().numpy()
            
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if (info is None) or ("episode" not in info):
                        continue
                    print(f"final_steps={i},episode_return={info['episode']['r']}")
                    writer.add_scalar("charts/episode_return",info['episode']['r'],global_step=i)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], i)
                    break
            for idx,trun in enumerate(truncations):
                if trun:
                    next_obs[idx]=infos["final_observation"][idx]
            rb.add(obs,next_obs,actions,rewards,terminations,infos)

            obs=next_obs

            if i > args.learning_start:
                if i % args.update_frequency==0:
                    data=rb.sample(args.batch_size)
                    with torch.no_grad():
                        _,action_p,logp=actor.get_action(data.next_observations)
                        qf1_t=qf1_target(data.next_observations)
                        qf2_t=qf2_target(data.next_observations)
                        qf_t_min=torch.minimum(qf1_t,qf2_t)   # 取最小值
                        value_expectation=(qf_t_min-alpha*logp)*action_p
                        y = data.rewards+args.gamma*(1-data.dones)*value_expectation   # 离散动作空间不需要蒙特卡洛采样 直接算所有动作的期望就行
                    qf1_q=qf1(data.observations)
                    qf2_q=qf2(data.observations)

                    loss_1=nn.MSELoss()(qf1_q,y)
                    loss_2=nn.MSELoss()(qf2_q,y)
                    qf1_optimizer.zero_grad()
                    qf2_optimizer.zero_grad()

                    loss_1.backward()
                    loss_2.backward()
                    qf1_optimizer.step()
                    qf2_optimizer.step()

                    _,action_p_obs,logp_obs=actor.get_action(data.observations)
                    with torch.no_grad():
                        qf1_q=qf1(data.observations)
                        qf2_q=qf2(data.observations)
                        qf_q_min=torch.min(qf1_q,qf2_q)

                    loss_p=-(torch.mean((qf_q_min-alpha.detach()*logp_obs)*action_p_obs))
                    actor_optimizer.zero_grad()
                    loss_p.backward()
                    actor_optimizer.step()

                    if args.autotune:

                        alpha_loss=(action_p_obs.detach()*(-log_alpha.exp()*(logp_obs+target_entropy).detach())).mean()

                        # 此处必须使用 detach() 进行分离
                        alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        alpha_optimizer.step()
                    
                    if i % args.target_update_frequency==0:
                        with torch.no_grad():

                            for param,target_param in zip(qf1.parameters(),qf1_target.parameters()):
                                target_param.copy_(args.tau*param.data+(1-args.tau)*target_param.data)
                            for param,target_param in zip(qf2.parameters(),qf2_target.parameters()):
                                target_param.copy_(args.tau*param.data+(1-args.tau)*target_param.data)
                        

                    if i % 100 ==0:
                        writer.add_scalar("losses/qf1_values",qf1_q.mean().item(),i)
                        writer.add_scalar("losses/qf2_values",qf2_q.mean().item(),i)
                        writer.add_scalar("lossed/qf1_loss",loss_1.item(),i)
                        writer.add_scalar("lossed/qf2_loss",loss_2.item(),i)
                        writer.add_scalar("losses/actor_loss",loss_p.item(),i)
                        writer.add_scalar("losses/alpha", alpha, i)
                        print("SPS:", int(i / (time.time() - start_time)))
                        writer.add_scalar("charts/SPS", int(i / (time.time() - start_time)), i)
                        if args.autotune:
                            writer.add_scalar("losses/alpha_loss", alpha_loss.item(), i)
        envs.close()
        writer.close()


            

            



        




    





