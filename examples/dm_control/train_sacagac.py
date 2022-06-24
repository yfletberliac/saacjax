import argparse
import os
from datetime import datetime

from rljax.algorithm import SACAGAC, SACVAR, SACCVAR
from rljax.env.mujoco.dmc import make_dmc_env
from rljax.trainer import Trainer


def run(args):
    env = make_dmc_env(args.domain_name, args.task_name, args.combined_challenge, args.seed)
    env_test = make_dmc_env(args.domain_name, args.task_name, args.combined_challenge, args.seed)

    algo = SACAGAC(
        num_agent_steps=args.num_agent_steps,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=args.seed,
        nstep=args.nstep,
        update_interval=args.update_interval,
        num_constraints=args.num_constraints,
        d2rl=True
    )

    # algo = SACVAR(
    #     num_agent_steps=args.num_agent_steps,
    #     state_space=env.observation_space,
    #     action_space=env.action_space,
    #     seed=args.seed,
    #     nstep=args.nstep,
    #     update_interval=args.update_interval,
    #     num_constraints=args.num_constraints,
    #     d2rl=True
    # )

    # algo = SACCVAR(
    #         num_agent_steps=args.num_agent_steps,
    #         state_space=env.observation_space,
    #         action_space=env.action_space,
    #         seed=args.seed,
    #         nstep=args.nstep,
    #         update_interval=args.update_interval,
    #         num_constraints=args.num_constraints,
    #         d2rl=True,
    #         num_quantiles=25,
    #         num_quantiles_to_drop=22,
    #     )

    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs/states/safety2", f"{args.domain_name}-{args.task_name}-{args.combined_challenge}", f"{str(algo)}-newinstantiatedchallenger-withbetalosskl-sg(beta)log(pioverpi_adv)-lr_critic_challenger1e-4-joint_angle_velocity_constraint-cons-lrbeta5e-3-safety_coeff07-seed{args.seed}-{time}")

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_agent_steps=args.num_agent_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
        num_constraints=args.num_constraints,
    )
    trainer.train()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--domain_name', type=str, default='cheetah')
    p.add_argument('--task_name', type=str, default='run')
    p.add_argument('--combined_challenge', type=str, default=None)
    p.add_argument("--num_agent_steps", type=int, default=1000000)
    p.add_argument("--eval_interval", type=int, default=10000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--nstep", type=int, default=1)
    p.add_argument("--update_interval", type=int, default=2)
    p.add_argument("--num_constraints", type=int, default=4)
    # p.add_argument("--agac_coef", type=float, default=1e-3)
    args = p.parse_args()
    run(args)
