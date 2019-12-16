import pytest
import torch

from a3c_pytorch.common import basic_memory


def assert_list_of_tensors(list_a, list_b):
    for tensor_a, tensor_b in zip(list_a, list_b):
        if not torch.equal(tensor_a, tensor_b):
            raise AssertionError(
                f"Tensors are not equal:\ntensor_a: {tensor_a}\ntensor_b: {tensor_b}"
            )


def empty_memory():
    return basic_memory.Memory(0.99)


def example_memory():
    m = basic_memory.Memory(0.99, 2)
    m.actions = [torch.tensor([1]), torch.tensor([0]), torch.tensor([1])]
    m.action_logprobs = [
        torch.tensor([0.53], requires_grad=True) * 0.5,
        torch.tensor([0.12], requires_grad=True) * 0.5,
        torch.tensor([1.53], requires_grad=True) * 0.5,
    ]
    m.state_values = [
        torch.tensor([[0.21]], requires_grad=True) * 0.5,
        torch.tensor([[0.85]], requires_grad=True) * 0.5,
        torch.tensor([[0.12]], requires_grad=True) * 0.5,
    ]
    m.rewards = torch.ones(3, 1)
    m.is_terminals = torch.tensor([[0], [0], [1]], dtype=torch.uint8)
    return m


def example_memory_return():
    m = basic_memory.Memory(0.99)

    m.rewards = torch.ones(5, 1)
    m.is_terminals = torch.tensor([[0], [0], [1], [0], [1]], dtype=torch.uint8)
    return m


@pytest.mark.parametrize(
    "new_val, old_val,expected_result, memory",
    (
        (torch.tensor(1.53), None, torch.tensor([[1.53]]), empty_memory()),
        (
            torch.tensor(1.53),
            torch.tensor([[2.43]]),
            torch.tensor([[2.43], [1.53]]),
            empty_memory(),
        ),
    ),
)
def test_Memory__append(new_val, old_val, expected_result, memory):
    result = memory._append(old_val, new_val)
    assert torch.equal(result, expected_result)


@pytest.mark.parametrize(
    "new_val, expected_result, memory",
    (
        (torch.tensor([1]), [torch.tensor([1])], empty_memory()),
        (
            torch.tensor([1]),
            [
                torch.tensor([1]),
                torch.tensor([0]),
                torch.tensor([1]),
                torch.tensor([1]),
            ],
            example_memory(),
        ),
    ),
)
def test_Memory_update_actions(new_val, expected_result, memory):
    memory.update_actions(new_val)
    assert_list_of_tensors(memory.actions, expected_result)


@pytest.mark.parametrize(
    "new_val, expected_result, memory",
    (
        (
            torch.tensor([0.53], requires_grad=True) * 0.5,
            [torch.tensor([0.53], requires_grad=True) * 0.5],
            empty_memory(),
        ),
        (
            torch.tensor([1.3], requires_grad=True) * 0.5,
            [
                torch.tensor([0.53], requires_grad=True) * 0.5,
                torch.tensor([0.12], requires_grad=True) * 0.5,
                torch.tensor([1.53], requires_grad=True) * 0.5,
                torch.tensor([1.3], requires_grad=True) * 0.5,
            ],
            example_memory(),
        ),
    ),
)
def test_Memory_update_action_logprobs(new_val, expected_result, memory):
    memory.update_action_logprobs(new_val)
    assert_list_of_tensors(memory.action_logprobs, expected_result)
    assert memory.action_logprobs[-1].grad_fn


@pytest.mark.parametrize(
    "new_val, expected_result, memory",
    (
        (
            torch.tensor([0.53], requires_grad=True) * 0.5,
            [torch.tensor([0.53], requires_grad=True) * 0.5],
            empty_memory(),
        ),
        (
            torch.tensor([[1.3]], requires_grad=True) * 0.5,
            [
                torch.tensor([[0.21]], requires_grad=True) * 0.5,
                torch.tensor([[0.85]], requires_grad=True) * 0.5,
                torch.tensor([[0.12]], requires_grad=True) * 0.5,
                torch.tensor([[1.3]], requires_grad=True) * 0.5,
            ],
            example_memory(),
        ),
    ),
)
def test_Memory_update_state_values(new_val, expected_result, memory):
    memory.update_state_values(new_val)
    assert_list_of_tensors(memory.state_values, expected_result)
    assert memory.state_values[-1].grad_fn


@pytest.mark.parametrize(
    "new_val, expected_result, memory",
    (
        (torch.tensor(0.53), torch.tensor([[0.53]]), empty_memory()),
        (torch.tensor(1.0), torch.ones(4, 1), example_memory()),
    ),
)
def test_Memory_update_rewards(new_val, expected_result, memory):
    memory.update_rewards(new_val)
    assert torch.equal(memory.rewards, expected_result)


@pytest.mark.parametrize(
    "new_val, expected_result, memory",
    (
        (
            torch.tensor(0, dtype=torch.uint8),
            torch.tensor([[0]], dtype=torch.uint8),
            empty_memory(),
        ),
        (
            torch.tensor(1, dtype=torch.uint8),
            torch.tensor([[0], [0], [1], [1]], dtype=torch.uint8),
            example_memory(),
        ),
    ),
)
def test_Memory_update_is_terminals(new_val, expected_result, memory):
    memory.update_is_terminals(new_val)
    assert torch.equal(memory.is_terminals, expected_result)


@pytest.mark.parametrize(
    "expected_result, memory",
    (
        (torch.tensor([1 + 1.99 * 0.99, 1 + 1 * 0.99, 1.0]), example_memory()),
        (
            torch.tensor([1 + 1.99 * 0.99, 1 + 1 * 0.99, 1.0, 1 + 1 * 0.99, 1.0]),
            example_memory_return(),
        ),
    ),
)
def test_Memory_calculate_returns(expected_result, memory):
    memory.calculate_returns()
    assert torch.equal(memory.returns, expected_result)


def test_get_batch():
    memory = example_memory()
    memory.calculate_returns()

    expected_batches = [
        {
            "action_logprobs": [
                torch.tensor([0.53], requires_grad=True) * 0.5,
                torch.tensor([0.12], requires_grad=True) * 0.5,
            ],
            "state_values": [
                torch.tensor([[0.21]], requires_grad=True) * 0.5,
                torch.tensor([[0.85]], requires_grad=True) * 0.5,
            ],
            "returns": torch.tensor([1 + 1.99 * 0.99, 1 + 1 * 0.99]),
        },
        {
            "action_logprobs": [torch.tensor([1.53], requires_grad=True) * 0.5],
            "state_values": [torch.tensor([[0.12]], requires_grad=True) * 0.5],
            "returns": torch.tensor([1.0]),
        },
    ]

    for i, batch in enumerate(memory):
        assert_list_of_tensors(
            batch.action_logprobs, expected_batches[i]["action_logprobs"]
        )
        assert_list_of_tensors(batch.state_values, expected_batches[i]["state_values"])

        assert batch.action_logprobs[-1].grad_fn
        assert batch.state_values[-1].grad_fn

        assert torch.equal(batch.returns, expected_batches[i]["returns"])
