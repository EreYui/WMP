
import math

import pytest
import torch

from wmp_lab.learning.networks import AMPDiscriminator
from wmp_lab.learning.progress_rewards import (
    ForwardProgressGate, checkpoint_reward_revision, gate_positive_task_terms, tracking_velocity_with_progress,
    PROGRESS_REWARD_REVISION, REFERENCE_REWARD_REVISION,
)


def initialized(n=1, dt=.02):
    gate = ForwardProgressGate(n, dt)
    ids = torch.arange(n)
    gate.reset(ids, torch.zeros(n), torch.full((n,), .6), torch.ones(n, dtype=torch.bool))
    return gate


@pytest.mark.parametrize("frequency", [.2, .6, 1., 5.])
def test_bounded_oscillation_and_tiny_drift_do_not_refresh_progress(frequency):
    gate = initialized()
    command = torch.tensor([.6])
    for tick in range(1, 601):
        t = tick * .02
        position = torch.tensor([.25 * math.sin(2 * math.pi * frequency * t) + .001 * t])
        status = gate.update(position, command, torch.tensor([True]))
        if t > 8:
            assert status.stalled.item()


def test_actual_forward_and_reverse_travel_remain_rewarded():
    gate = initialized(2)
    command = torch.tensor([.6, -.6])
    for tick in range(1, 501):
        status = gate.update(command * tick * .02, command, torch.tensor([True, True]))
        assert not status.stalled.any()
    torch.testing.assert_close(status.speed, torch.tensor([.6, .6]), atol=2e-6, rtol=0)


def test_partial_reset_and_terminal_gate_snapshot():
    gate = initialized(2)
    for _ in range(110):
        status = gate.update(torch.zeros(2), torch.full((2,), .6), torch.tensor([True, True]))
    terminal_gate = (~status.stalled).float()
    assert terminal_gate.tolist() == [0., 0.]
    gate.reset(torch.tensor([0]), torch.tensor([100.]), torch.tensor([.6]), torch.tensor([True]))
    status = gate.update(torch.tensor([100., 0.]), torch.full((2,), .6), torch.tensor([True, True]))
    assert status.stalled.tolist() == [False, True]
    assert terminal_gate.tolist() == [0., 0.]


def test_stop_start_and_reversal_get_fresh_grace_without_old_frontier():
    gate = initialized()
    for _ in range(110):
        gate.update(torch.zeros(1), torch.tensor([.6]), torch.tensor([True]))
    assert gate.stalled.item()
    assert not gate.update(torch.zeros(1), torch.zeros(1), torch.tensor([True])).stalled.item()
    assert not gate.update(torch.zeros(1), torch.tensor([.6]), torch.tensor([True])).stalled.item()
    for _ in range(110):
        gate.update(torch.zeros(1), torch.tensor([.6]), torch.tensor([True]))
    assert not gate.update(torch.zeros(1), torch.tensor([-.6]), torch.tensor([True])).stalled.item()


def test_low_commands_ineligible_terrain_and_positive_airtime():
    gate = initialized(3)
    for _ in range(110):
        status = gate.update(torch.zeros(3), torch.tensor([.2, 0., .6]), torch.tensor([True, True, False]))
    assert status.stalled.tolist() == [True, False, False]
    terms = dict(tracking=torch.tensor([.9, 1., .9]), yaw_tracking=torch.full((3,), .5),
                 air_time=torch.full((3,), 2.), stuck=torch.tensor([-1., 0., -1.]),
                 torque=torch.full((3,), -.1))
    result = gate_positive_task_terms(terms, status.stalled)
    assert sum(result.values()).clamp_min(0)[0] == 0
    assert sum(result.values())[1] == sum(terms.values())[1]
    assert sum(result.values())[2] == sum(terms.values())[2]
    assert torch.equal(result["torque"], terms["torque"])


def test_amp_gate_removes_style_only_and_retains_reference_arithmetic():
    disc = AMPDiscriminator((8, 4))
    for parameter in disc.parameters():
        parameter.data.zero_()
    disc.head.bias.data.fill_(1.)
    state = torch.zeros(2, 30)
    task = torch.tensor([0., .02])
    legacy, score = disc.predict_amp_reward(state, state, task)
    gated, gated_score = disc.predict_amp_reward(state, state, task, torch.tensor([0., 1.]))
    torch.testing.assert_close(legacy, torch.tensor([.01, .03]), rtol=0, atol=1e-8)
    torch.testing.assert_close(gated, torch.tensor([0., .03]), rtol=0, atol=1e-8)
    assert torch.equal(score, gated_score)
    with pytest.raises(ValueError, match="gate"):
        disc.predict_amp_reward(state, state, task, torch.ones(2, 1))


def test_reward_revision_compatibility_and_conflict_rejection():
    assert checkpoint_reward_revision({}) == REFERENCE_REWARD_REVISION
    payload = dict(extra=dict(environment_contract=dict(config=dict(reward_revision=PROGRESS_REWARD_REVISION))))
    assert checkpoint_reward_revision(payload) == PROGRESS_REWARD_REVISION
    payload["extra"]["environment"] = dict(config=dict(reward_revision=REFERENCE_REWARD_REVISION))
    with pytest.raises(ValueError, match="disagrees"):
        checkpoint_reward_revision(payload)


@pytest.mark.parametrize("dt,window,speed", [(0, 2, .1), (.02, 0, .1), (.02, 2, float("nan"))])
def test_invalid_progress_timing_is_rejected(dt, window, speed):
    with pytest.raises(ValueError):
        ForwardProgressGate(1, dt, window_s=window, min_speed=speed)


def test_slow_net_progress_cannot_claim_fast_instantaneous_tracking():
    command = torch.tensor([[.6, 0., 0.]])
    value = tracking_velocity_with_progress(
        torch.tensor([[.6, 0., 0.]]), command, torch.tensor([True]),
        torch.tensor([.15]), torch.ones(1), torch.tensor([True]),
    )
    torch.testing.assert_close(value, torch.tensor([math.exp(-.45**2/.25)]))
