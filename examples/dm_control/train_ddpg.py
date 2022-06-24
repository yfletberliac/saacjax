import argparse
import os
from datetime import datetime

from rljax.algorithm import DDPG
from rljax.env.mujoco.dmc import make_dmc_env
from rljax.trainer import Trainer


def run(args):
    env = make_dmc_env(args.domain_name, args.task_name, args.combined_challenge, args.seed)
    env_test = make_dmc_env(args.domain_name, args.task_name, args.combined_challenge, args.seed)

    algo = DDPG(
        num_agent_steps=args.num_agent_steps,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=args.seed,
        d2rl=False
    )

    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs/safety", f"{args.domain_name}-{args.task_name}-{args.combined_challenge}", f"{str(algo)}-update_interval_policy-{args.update_interval_policy}-seed{args.seed}-{time}")

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_agent_steps=args.num_agent_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
    )
    trainer.train()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--domain_name', type=str, default='cheetah')
    p.add_argument('--task_name', type=str, default='run')
    p.add_argument('--combined_challenge', type=str, default=None)
    p.add_argument("--num_agent_steps", type=int, default=10000000)
    p.add_argument("--eval_interval", type=int, default=5000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num_constraints", type=int, default=3)
    p.add_argument("--update_interval_policy", type=int, default=1)
    args = p.parse_args()
    run(args)
