from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .distance import matching_distance_window
from .pattern_unit import extract_benign_behavior_units


def is_subsequence(sub: Sequence[int], seq: Sequence[int]) -> bool:
    it = iter(seq)
    return all(any(x == y for y in it) for x in sub)


def _perturbation_budget(seq_len: int, ratio: float) -> int:
    return int(round(ratio * seq_len))


@dataclass
class BenignConstraint:

    benign_units: List[np.ndarray]
    attack_units: Optional[List[np.ndarray]] = None
    match_threshold: float = 0.0

    benign_vocab_: np.ndarray = field(init=False)

    def __post_init__(self):
        if len(self.benign_units) > 0:
            self.benign_vocab_ = np.unique(np.concatenate(
                [np.asarray(u).ravel() for u in self.benign_units]
            ))
        else:
            self.benign_vocab_ = np.array([], dtype=np.int64)

    def editable_mask(self, seq: Sequence[int]) -> np.ndarray:
        seq = np.asarray(seq)
        n = len(seq)
        mask = np.zeros(n, dtype=bool)
        for unit in self.benign_units:
            unit = np.asarray(unit)
            l = len(unit)
            if l == 0 or l > n:
                continue
            for start in range(n - l + 1):
                window = seq[start : start + l]
                if matching_distance_window(unit, window) <= self.match_threshold:
                    mask[start : start + l] = True
        return mask

    def editable_positions(self, seq: Sequence[int]) -> np.ndarray:
        return np.where(self.editable_mask(seq))[0]

    def is_semantics_preserved(
        self, original: Sequence[int], adversarial: Sequence[int]
    ) -> bool:
        if self.attack_units:
            for unit in self.attack_units:
                unit = list(np.asarray(unit))
                if is_subsequence(unit, original) and not is_subsequence(unit, adversarial):
                    return False
            return True

        original = np.asarray(original)
        protected = original[~self.editable_mask(original)]
        return is_subsequence(list(protected), list(adversarial))


@dataclass
class InsertionAttack:

    constraint: BenignConstraint
    ratio: float = 0.1
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def perturb(self, seq: Sequence[int]) -> List[int]:
        seq = list(seq)
        budget = _perturbation_budget(len(seq), self.ratio)
        units = self.constraint.benign_units
        if budget <= 0 or len(units) == 0:
            return seq

        adv = list(seq)
        inserted = 0
        while inserted < budget:
            unit = [int(v) for v in np.asarray(units[self.rng.integers(0, len(units))])]
            remaining = budget - inserted
            if len(unit) > remaining:
                unit = unit[:remaining]
            pos = int(self.rng.integers(0, len(adv) + 1))
            adv = adv[:pos] + unit + adv[pos:]
            inserted += len(unit)
        return adv


@dataclass
class DeletionAttack:

    constraint: BenignConstraint
    ratio: float = 0.1
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def perturb(self, seq: Sequence[int]) -> List[int]:
        seq = np.asarray(seq)
        budget = _perturbation_budget(len(seq), self.ratio)
        editable = self.constraint.editable_positions(seq)
        if budget <= 0 or editable.size == 0:
            return list(seq)

        n_del = min(budget, editable.size)
        del_idx = set(self.rng.choice(editable, size=n_del, replace=False).tolist())
        adv = [int(e) for i, e in enumerate(seq) if i not in del_idx]
        return adv


@dataclass
class ReplacementAttack:

    constraint: BenignConstraint
    ratio: float = 0.1
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def perturb(self, seq: Sequence[int]) -> List[int]:
        seq = np.asarray(seq)
        budget = _perturbation_budget(len(seq), self.ratio)
        editable = self.constraint.editable_positions(seq)
        vocab = self.constraint.benign_vocab_
        if budget <= 0 or editable.size == 0 or vocab.size == 0:
            return list(seq)

        n_rep = min(budget, editable.size)
        rep_idx = self.rng.choice(editable, size=n_rep, replace=False)
        adv = [int(e) for e in seq]
        for i in rep_idx:
            choices = vocab[vocab != adv[i]]
            if choices.size == 0:
                choices = vocab
            adv[i] = int(self.rng.choice(choices))
        return adv


def make_score_fn(
    detector,
    max_len: int,
    abnormal_class: int = 1,
) -> Callable[[Sequence[int]], float]:
    import tensorflow as tf

    def score_fn(seq: Sequence[int]) -> float:
        s = list(seq)[:max_len]
        ids = tf.keras.preprocessing.sequence.pad_sequences(
            [s], maxlen=max_len, padding="post", value=0
        ).astype(np.int32)
        mask = (ids != 0).astype(np.float32)
        logits = detector([ids, mask], training=False)
        prob = tf.nn.softmax(logits, axis=-1).numpy()[0, abnormal_class]
        return float(prob)

    return score_fn


@dataclass
class RLAttack:

    constraint: BenignConstraint
    score_fn: Callable[[Sequence[int]], float]
    ratio: float = 0.1
    tau: float = 0.5
    episodes: int = 200
    gamma: float = 0.99
    success_bonus: float = 1.0
    learning_rate: float = 1e-2
    seed: int = 0

    def __post_init__(self):
        import tensorflow as tf

        self._tf = tf
        self.rng = np.random.default_rng(self.seed)
        tf.random.set_seed(self.seed)
        self.n_actions = 3
        self.state_dim = 4
        # 小型策略网络：state -> 动作概率
        self.policy = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_dim,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(self.n_actions),
        ])
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self._baseline = 0.0

    def _state(self, cur_seq, orig_len, used_budget, budget, cur_score) -> np.ndarray:
        editable = self.constraint.editable_positions(cur_seq)
        return np.array([
            cur_score,
            used_budget / max(budget, 1),
            len(cur_seq) / max(orig_len, 1),
            editable.size / max(len(cur_seq), 1),
        ], dtype=np.float32)

    def _apply_action(self, cur_seq, action) -> Tuple[List[int], int]:
        cur = list(cur_seq)
        units = self.constraint.benign_units
        vocab = self.constraint.benign_vocab_

        if action == 0:  # 插入良性单元
            if len(units) == 0:
                return cur, 0
            unit = [int(v) for v in np.asarray(units[self.rng.integers(0, len(units))])]
            pos = int(self.rng.integers(0, len(cur) + 1))
            new = cur[:pos] + unit + cur[pos:]
            return new, len(unit)

        editable = self.constraint.editable_positions(cur)
        if editable.size == 0:
            return cur, 0

        if action == 1:
            i = int(self.rng.choice(editable))
            new = cur[:i] + cur[i + 1:]
            return new, 1

        if action == 2:
            if vocab.size == 0:
                return cur, 0
            i = int(self.rng.choice(editable))
            choices = vocab[vocab != cur[i]]
            if choices.size == 0:
                choices = vocab
            new = list(cur)
            new[i] = int(self.rng.choice(choices))
            return new, 1

        return cur, 0

    def _rollout(self, orig_seq, train: bool):
        tf = self._tf
        orig_len = len(orig_seq)
        budget = _perturbation_budget(orig_len, self.ratio)
        cur = list(orig_seq)
        cur_score = self.score_fn(cur)

        states, actions, rewards = [], [], []
        best_seq, best_score = list(cur), cur_score
        used = 0

        while used < budget:
            state = self._state(cur, orig_len, used, budget, cur_score)
            logits = self.policy(state[None, :], training=False).numpy()[0]
            probs = tf.nn.softmax(logits).numpy()
            if train:
                action = int(self.rng.choice(self.n_actions, p=probs))
            else:
                action = int(np.argmax(probs))

            new_seq, cost = self._apply_action(cur, action)
            if cost == 0:
                states.append(state); actions.append(action); rewards.append(-0.05)
                if train and self.rng.random() < 0.2:
                    break
                continue

            if not self.constraint.is_semantics_preserved(orig_seq, new_seq):
                states.append(state); actions.append(action); rewards.append(-0.1)
                continue

            new_score = self.score_fn(new_seq)
            reward = cur_score - new_score
            cur, cur_score = new_seq, new_score
            used += cost

            done = cur_score < self.tau
            if done:
                reward += self.success_bonus

            states.append(state); actions.append(action); rewards.append(reward)

            if cur_score < best_score:
                best_seq, best_score = list(cur), cur_score
            if done:
                break

        return states, actions, rewards, best_seq, best_score

    def _update_policy(self, states, actions, rewards):
        tf = self._tf
        if len(states) == 0:
            return
        returns = np.zeros(len(rewards), dtype=np.float32)
        g = 0.0
        for t in reversed(range(len(rewards))):
            g = rewards[t] + self.gamma * g
            returns[t] = g
        self._baseline = 0.95 * self._baseline + 0.05 * float(returns.mean())
        adv = returns - self._baseline
        if adv.std() > 1e-6:
            adv = adv / (adv.std() + 1e-8)

        S = tf.convert_to_tensor(np.array(states), dtype=tf.float32)
        A = tf.convert_to_tensor(np.array(actions), dtype=tf.int32)
        ADV = tf.convert_to_tensor(adv, dtype=tf.float32)

        with tf.GradientTape() as tape:
            logits = self.policy(S, training=True)
            logp = tf.nn.log_softmax(logits, axis=-1)
            idx = tf.stack([tf.range(tf.shape(A)[0]), A], axis=1)
            chosen_logp = tf.gather_nd(logp, idx)
            loss = -tf.reduce_mean(chosen_logp * ADV)  # REINFORCE 目标
        grads = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

    def fit(self, abnormal_sequences: Sequence[Sequence[int]], verbose: int = 1) -> "RLAttack":
        seqs = [list(s) for s in abnormal_sequences if len(s) > 0]
        if len(seqs) == 0:
            return self
        for ep in range(self.episodes):
            s = seqs[self.rng.integers(0, len(seqs))]
            states, actions, rewards, _, best_score = self._rollout(s, train=True)
            self._update_policy(states, actions, rewards)
            if verbose and (ep + 1) % max(1, self.episodes // 10) == 0:
                print(f"[RLAttack] episode {ep + 1}/{self.episodes} "
                      f"best_score={best_score:.3f} baseline={self._baseline:.3f}")
        return self

    def perturb(self, seq: Sequence[int]) -> List[int]:
        _, _, _, best_seq, _ = self._rollout(list(seq), train=False)
        return best_seq

    def attack(
        self, abnormal_sequences: Sequence[Sequence[int]], fit_first: bool = True, verbose: int = 1
    ) -> List[List[int]]:
        if fit_first:
            self.fit(abnormal_sequences, verbose=verbose)
        return [self.perturb(s) for s in abnormal_sequences]


def build_benign_constraint(
    normal_sequences: Sequence[Sequence[int]],
    attack_units: Optional[List[np.ndarray]] = None,
    l_min: int = 2,
    l_max: int = 5,
    metric: str = "matching",
    support_threshold: float = 1.0,
    top_k: Optional[int] = 100,
    match_threshold: float = 0.0,
) -> BenignConstraint:
    benign_units = extract_benign_behavior_units(
        normal_sequences,
        l_min=l_min,
        l_max=l_max,
        metric=metric,
        support_threshold=support_threshold,
        top_k=top_k,
    )
    return BenignConstraint(
        benign_units=benign_units,
        attack_units=attack_units,
        match_threshold=match_threshold,
    )


ATTACK_REGISTRY = {
    "insert": InsertionAttack,
    "drop": DeletionAttack,
    "alter": ReplacementAttack,
}


def generate_adversarial_sequences(
    abnormal_sequences: Sequence[Sequence[int]],
    constraint: BenignConstraint,
    strategy: str,
    ratio: float,
    seed: int = 0,
) -> List[List[int]]:
    if strategy not in ATTACK_REGISTRY:
        raise ValueError(f"未知攻击 (非RL): {strategy}; 可选 {list(ATTACK_REGISTRY)}")
    rng = np.random.default_rng(seed)
    attack = ATTACK_REGISTRY[strategy](constraint=constraint, ratio=ratio, rng=rng)
    return [attack.perturb(s) for s in abnormal_sequences]
