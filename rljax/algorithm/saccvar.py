from functools import partial
from typing import List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import optix

from rljax.algorithm.base_class import OffPolicyActorCritic
from rljax.network import ContinuousQuantileFunction, StateDependentGaussianPolicy
from rljax.util import optimize, reparameterize_gaussian_and_tanh, quantile_loss


class SACCVAR(OffPolicyActorCritic):
    name = "SACCVAR"

    def __init__(
            self,
            num_agent_steps,
            state_space,
            action_space,
            seed,
            max_grad_norm=None,
            gamma=0.99,
            nstep=1,
            num_critics=2,
            buffer_size=5 * 10 ** 5,
            use_per=False,
            batch_size=256,
            start_steps=10000,
            update_interval=2,
            tau=1e-3,
            fn_actor=None,
            fn_critic=None,
            lr_actor=1e-4,
            lr_critic=1e-4,
            lr_alpha=1e-4,
            lr_beta=5e-3,
            units_actor=(256, 256, 256, 256),
            units_critic=(256, 256, 256, 256),
            log_std_min=-20.0,
            log_std_max=2.0,
            d2rl=False,
            init_alpha=1.0,
            init_beta=1.0,
            adam_b1_alpha=0.9,
            adam_b1_beta=0.9,
            num_constraints=3,
            num_quantiles=25,
            num_quantiles_to_drop=2,
            *args,
            **kwargs,
    ):
        if not hasattr(self, "use_key_critic"):
            self.use_key_critic = True
        if not hasattr(self, "use_key_actor"):
            self.use_key_actor = True

        super(SACCVAR, self).__init__(
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
            nstep=nstep,
            num_critics=num_critics,
            buffer_size=buffer_size,
            use_per=use_per,
            batch_size=batch_size,
            start_steps=start_steps,
            update_interval=update_interval,
            tau=tau,
            *args,
            **kwargs,
        )
        if d2rl:
            self.name += "-D2RL"

        if fn_critic is None:
            def fn_critic(s, a):
                return ContinuousQuantileFunction(
                    num_critics=num_critics,
                    hidden_units=units_critic,
                    num_quantiles=num_quantiles,
                    d2rl=d2rl,
                )(s, a)

        if fn_actor is None:
            def fn_actor(s):
                return StateDependentGaussianPolicy(
                    action_space=action_space,
                    hidden_units=units_actor,
                    log_std_min=log_std_min,
                    log_std_max=log_std_max,
                    d2rl=d2rl,
                )(s)

        # Critic.
        self.critic = hk.without_apply_rng(hk.transform(fn_critic))
        self.params_critic = self.params_critic_target = self.critic.init(next(self.rng), *self.fake_args_critic)
        opt_init, self.opt_critic = optix.adam(lr_critic)
        self.opt_state_critic = opt_init(self.params_critic)
        # Actor.
        self.actor = hk.without_apply_rng(hk.transform(fn_actor))
        self.challenger = hk.without_apply_rng(hk.transform(fn_actor))
        self.params_actor = self.actor.init(next(self.rng), *self.fake_args_actor)
        self.params_challenger = self.challenger.init(next(self.rng), *self.fake_args_actor)
        opt_init, self.opt_actor = optix.adam(lr_actor)
        self.opt_state_actor = opt_init(self.params_actor)
        opt_init, self.opt_challenger = optix.adam(lr_actor)
        self.opt_state_challenger = opt_init(self.params_challenger)
        # Entropy coefficient.
        if not hasattr(self, "target_entropy"):
            self.target_entropy = -float(self.action_space.shape[0])
        # Challenger bonus coefficient.
        if not hasattr(self, "target_challenger_bonus"):
            # self.target_challenger_bonus = -float(self.action_space.shape[0])
            self.target_challenger_bonus = -float(0.0)

        self.log_alpha = jnp.array(np.log(init_alpha), dtype=jnp.float32)
        self.log_beta = jnp.array(np.log(init_beta), dtype=jnp.float32)
        self.num_constraints = num_constraints
        opt_init, self.opt_alpha = optix.adam(lr_alpha, b1=adam_b1_alpha)
        self.opt_state_alpha = opt_init(self.log_alpha)
        opt_init, self.opt_beta = optix.adam(lr_beta, b1=adam_b1_beta)
        self.opt_state_beta = opt_init(self.log_beta)

        # Quantile
        self.cum_p_prime = jnp.expand_dims((jnp.arange(0, num_quantiles, dtype=jnp.float32) + 0.5) / num_quantiles, 0)
        self.num_quantiles = num_quantiles
        # self.num_quantiles_target = num_quantiles_to_drop * num_critics
        self.num_quantiles_target = (num_quantiles - num_quantiles_to_drop) * num_critics

    @partial(jax.jit, static_argnums=0)
    def _select_action(
            self,
            params_actor: hk.Params,
            state: np.ndarray,
    ) -> jnp.ndarray:
        mean, _ = self.actor.apply(params_actor, state)
        return jnp.tanh(mean)

    @partial(jax.jit, static_argnums=0)
    def _explore(
            self,
            params_actor: hk.Params,
            state: np.ndarray,
            key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        mean, log_std = self.actor.apply(params_actor, state)
        return reparameterize_gaussian_and_tanh(mean, log_std, key, False)

    def update(self, writer=None):
        self.learning_step += 1
        weight, batch = self.buffer.sample(self.batch_size)
        state, action, reward, done, next_state = batch
        reward_c = (jnp.ones((len(state), 2)) - state[:, -self.num_constraints:(-self.num_constraints + 1)]).mean()

        # print("update critic")
        # Update critic.
        self.opt_state_critic, self.params_critic, loss_critic, abs_td = optimize(
            self._loss_critic,
            self.opt_critic,
            self.opt_state_critic,
            self.params_critic,
            self.max_grad_norm,
            params_critic_target=self.params_critic_target,
            params_actor=self.params_actor,
            log_alpha=self.log_alpha,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
            weight=weight,
            **self.kwargs_critic,
        )

        # Update priority.
        if self.use_per:
            self.buffer.update_priority(abs_td)

        # print("update actor")
        # Update actor.
        self.opt_state_actor, self.params_actor, loss_actor, mean_log = optimize(
            self._loss_actor,
            self.opt_actor,
            self.opt_state_actor,
            self.params_actor,
            self.max_grad_norm,
            params_critic=self.params_critic,
            params_challenger=self.params_challenger,
            log_alpha=self.log_alpha,
            log_beta=self.log_beta,
            state=state,
            **self.kwargs_actor,
        )
        mean_log_pi, mean_log_pi_challenger = mean_log

        # print("update alpha")
        # Update alpha.
        self.opt_state_alpha, self.log_alpha, loss_alpha, _ = optimize(
            self._loss_alpha,
            self.opt_alpha,
            self.opt_state_alpha,
            self.log_alpha,
            None,
            mean_log_pi=mean_log_pi,
        )

        # print("update beta")
        # Update beta.
        self.opt_state_beta, self.log_beta, loss_beta, _ = optimize(
            self._loss_beta,
            self.opt_beta,
            self.opt_state_beta,
            self.log_beta,
            None,
            mean_log_pi=mean_log_pi,
            mean_log_pi_challenger=mean_log_pi_challenger,
        )

        # print("update challenger")
        # Update challenger.
        # self.params_challenger = self._update_challenger(self.params_challenger, self.params_actor)
        # weight_challenger, batch_challenger = self.buffer_challenger.sample(self.batch_size)
        # state_challenger, _, _, _, _ = batch_challenger
        # self.buffer_challenger.update_priority(abs_td)
        self.opt_state_challenger, self.params_challenger, loss_challenger, _ = optimize(
            self._loss_challenger,
            self.opt_challenger,
            self.opt_state_challenger,
            self.params_challenger,
            self.max_grad_norm,
            params_critic=self.params_critic,
            state=state,
            # action=action,
            log_alpha=self.log_alpha,
            **self.kwargs_actor,
        )

        # print("update target")
        # Update target network.
        self.params_critic_target = self._update_target(self.params_critic_target, self.params_critic)

        if writer and self.learning_step % 10000 == 0:
            writer.add_scalar("loss/critic", loss_critic, self.learning_step)
            writer.add_scalar("loss/actor", loss_actor, self.learning_step)
            writer.add_scalar("loss/challenger", loss_challenger, self.learning_step)
            writer.add_scalar("loss/alpha", loss_alpha, self.learning_step)
            writer.add_scalar("loss/beta", loss_beta, self.learning_step)
            writer.add_scalar("stat/alpha", jnp.exp(self.log_alpha), self.learning_step)
            writer.add_scalar("stat/beta", jnp.exp(self.log_beta), self.learning_step)
            writer.add_scalar("stat/entropy", -mean_log_pi, self.learning_step)
            writer.add_scalar("stat/challenger_bonus", mean_log_pi - mean_log_pi_challenger, self.learning_step)
            writer.add_scalar("stat/reward_c", reward_c, self.learning_step)

    @partial(jax.jit, static_argnums=0)
    def _sample_action(
            self,
            params_actor: hk.Params,
            state: np.ndarray,
            key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        mean, log_std = self.actor.apply(params_actor, state)
        return reparameterize_gaussian_and_tanh(mean, log_std, key, True)

    @partial(jax.jit, static_argnums=0)
    def _calculate_log_pi(
            self,
            action: np.ndarray,
            log_pi: np.ndarray,
    ) -> jnp.ndarray:
        return log_pi

    @partial(jax.jit, static_argnums=0)
    def _loss_critic(
            self,
            params_critic: hk.Params,
            params_critic_target: hk.Params,
            params_actor: hk.Params,
            log_alpha: jnp.ndarray,
            state: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            next_state: np.ndarray,
            weight: np.ndarray or List[jnp.ndarray],
            *args,
            **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        next_action, next_log_pi = self._sample_action(params_actor, next_state, *args, **kwargs)
        target = self._calculate_target(params_critic_target, log_alpha, reward, done, next_state,
                                        next_action, next_log_pi)
        q_list = self._calculate_value_list(params_critic, state, action)
        return self._calculate_loss_critic_and_abs_td(q_list, target, weight)

    @partial(jax.jit, static_argnums=0)
    def _loss_actor(
            self,
            params_actor: hk.Params,
            params_challenger: hk.Params,
            params_critic: hk.Params,
            log_alpha: jnp.ndarray,
            log_beta: jnp.ndarray,
            state: np.ndarray,
            *args,
            **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        action, log_pi = self._sample_action(params_actor, state, *args, **kwargs)
        _, log_pi_challenger = self._sample_action(params_challenger, state, *args, **kwargs)
        mean_q = self._calculate_value(params_critic, state, action).mean()
        mean_log_pi = self._calculate_log_pi(action, log_pi).mean()
        mean_log_pi_challenger = self._calculate_log_pi(action, log_pi_challenger).mean()

        return jax.lax.stop_gradient(jnp.exp(log_alpha)) * mean_log_pi \
               - mean_q - jax.lax.stop_gradient(jnp.exp(log_beta)) * (mean_log_pi - mean_log_pi_challenger), \
               [jax.lax.stop_gradient(mean_log_pi), jax.lax.stop_gradient(mean_log_pi_challenger)]

    @partial(jax.jit, static_argnums=0)
    def _loss_beta(
            self,
            log_beta: jnp.ndarray,
            mean_log_pi: jnp.ndarray,
            mean_log_pi_challenger: jnp.ndarray,
    ) -> jnp.ndarray:
        return -log_beta * (
                    self.target_challenger_bonus + jnp.exp(mean_log_pi) * (mean_log_pi_challenger - mean_log_pi)), None

    @partial(jax.jit, static_argnums=0)
    def _loss_alpha(
            self,
            log_alpha: jnp.ndarray,
            mean_log_pi: jnp.ndarray,
    ) -> jnp.ndarray:
        return -log_alpha * (self.target_entropy + mean_log_pi), None

    @partial(jax.jit, static_argnums=0)
    def _loss_challenger(
            self,
            params_challenger: hk.Params,
            params_critic: hk.Params,
            state: np.ndarray,
            # action: np.ndarray,
            log_alpha: jnp.ndarray,
            *args,
            **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        action_challenger, log_pi_challenger = self._sample_action(params_challenger, state, *args, **kwargs)
        mean_log_pi_challenger = self._calculate_log_pi(action_challenger, log_pi_challenger).mean()
        quantile = self._calculate_value(params_critic, state, action_challenger)
        quantile = jnp.sort(quantile)[:, : self.num_quantiles_target]
        # quantile = jnp.concatenate((jnp.sort(quantile)[:, :self.num_quantiles_target], jnp.sort(quantile)[:, (self.num_quantiles*2-self.num_quantiles_target):]), axis=1)
        return jax.lax.stop_gradient(jnp.exp(log_alpha)) * mean_log_pi_challenger \
               + quantile.mean(), None

    @partial(jax.jit, static_argnums=0)
    def _calculate_value(
            self,
            params_critic: hk.Params,
            state: np.ndarray,
            action: np.ndarray,
    ) -> jnp.ndarray:
        return jnp.concatenate(self._calculate_value_list(params_critic, state, action), axis=1)

    @partial(jax.jit, static_argnums=0)
    def _calculate_target(
            self,
            params_critic_target: hk.Params,
            log_alpha: jnp.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            next_state: np.ndarray,
            next_action: jnp.ndarray,
            next_log_pi: jnp.ndarray,
    ) -> jnp.ndarray:
        next_quantile = self._calculate_value(params_critic_target, next_state, next_action)
        next_quantile = jnp.sort(next_quantile)[:, :]
        next_quantile -= jnp.exp(log_alpha) * self._calculate_log_pi(next_action, next_log_pi)
        return jax.lax.stop_gradient(reward + (1.0 - done) * self.discount * next_quantile)

    @partial(jax.jit, static_argnums=0)
    def _calculate_loss_critic_and_abs_td(
            self,
            quantile_list: List[jnp.ndarray],
            target: jnp.ndarray,
            weight: np.ndarray,
    ) -> jnp.ndarray:
        loss_critic = 0.0
        for quantile in quantile_list:
            loss_critic += quantile_loss(target[:, None, :] - quantile[:, :, None], self.cum_p_prime, weight, "huber")
        loss_critic /= self.num_critics * self.num_quantiles
        abs_td = jnp.abs(target[:, None, :] - quantile_list[0][:, :, None]).mean(axis=1).mean(axis=1, keepdims=True)
        return loss_critic, jax.lax.stop_gradient(abs_td)
