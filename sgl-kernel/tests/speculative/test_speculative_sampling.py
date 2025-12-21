import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import (
    chain_speculative_sampling_target_only,
    tree_speculative_sampling_target_only,
)

test_cases = [
    (
        1,
        1,
        [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18],
        [[0, 3, 4, 5], [6, 10, 11, -1]],
        [3, 2],
    ),
    (
        0,  # threshold_single
        0,  # threshold_acc
        [1, 2, 18, -1, -1, -1, 11, -1, -1, -1, 12, 18],
        [[0, 1, 2, -1], [6, 10, 11, -1]],
        [2, 2],
    ),
]


@pytest.mark.parametrize(
    "threshold_single, threshold_acc, expected_predicts, expected_accept_index, expected_accept_token_num",
    test_cases,
)
def test_tree_speculative_sampling_target_only(
    threshold_single,
    threshold_acc,
    expected_predicts,
    expected_accept_index,
    expected_accept_token_num,
):
    """
    Tests the tree_speculative_sampling_target_only function using Pytest parameterization.
    """
    device = "cuda"

    candidates = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [7, 8, 9, 10, 11, 12],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_next_token = torch.tensor(
        [
            [1, 2, -1, 4, 5, -1],
            [4, 2, 3, -1, 5, -1],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_next_sibling = torch.tensor(
        [
            [-1, 3, -1, -1, -1, -1],
            [-1, -1, -1, -1, 1, -1],
        ],
        dtype=torch.int64,
        device=device,
    )

    target_logits = torch.full((2, 6, 20), 1, dtype=torch.float32, device=device)
    target_logits[0, 0, 3] = 10
    target_logits[0, 3, 4] = 10
    target_logits[0, 4, 5] = 10
    target_logits[1, 0, 11] = 10
    target_logits[1, 4, 12] = 10

    for i in range(target_logits.shape[0]):
        for j in range(target_logits.shape[1]):
            if torch.max(target_logits[i, j]) < 10:
                target_logits[i, j, 18] = 10

    temperatures = torch.tensor([0.01, 0.01], dtype=torch.float32, device=device)
    bs, num_draft_tokens = candidates.shape
    num_spec_step = len(expected_accept_index[0])
    predict_shape = (len(expected_predicts),)

    predicts = torch.full(predict_shape, -1, dtype=torch.int32, device=device)
    accept_index = torch.full((bs, num_spec_step), -1, dtype=torch.int32, device=device)
    accept_token_num = torch.full((bs,), 0, dtype=torch.int32, device=device)

    expanded_temperature = temperatures.unsqueeze(1).unsqueeze(1)
    target_probs = F.softmax(target_logits / expanded_temperature, dim=-1)
    coins = torch.rand(bs, num_draft_tokens, device=device, dtype=torch.float32)
    coins_for_final_sampling = torch.rand(bs, device=device).to(torch.float32)

    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coins_for_final_sampling,
        target_probs=target_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )

    assert (
        predicts.tolist() == expected_predicts
    ), f"Predicts mismatch for thresholds ({threshold_single}, {threshold_acc})"
    assert (
        accept_index.tolist() == expected_accept_index
    ), f"Accept index mismatch for thresholds ({threshold_single}, {threshold_acc})"
    assert (
        accept_token_num.tolist() == expected_accept_token_num
    ), f"Accept token num mismatch for thresholds ({threshold_single}, {threshold_acc})"


if __name__ == "__main__":
    pytest.main([__file__])


def test_chain_speculative_sampling_target_only_matches_tree():
    device = "cuda"
    bs = 2
    num_draft_tokens = 4
    vocab_size = 16
    num_spec_tokens = 3

    # topk=1 chain: no siblings
    candidates = torch.tensor(
        [
            [0, 3, 5, 7],
            [0, 4, 6, 8],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_next_token = torch.tensor(
        [
            [1, 2, -1, -1],
            [1, 2, -1, -1],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_next_sibling = torch.full(
        (bs, num_draft_tokens),
        -1,
        dtype=torch.int64,
        device=device,
    )

    # Make target_probs nearly one-hot so randomness doesn't affect results.
    target_logits = torch.full(
        (bs, num_draft_tokens, vocab_size), -10, dtype=torch.float32, device=device
    )
    # root distribution picks first draft token (node 1 token id)
    target_logits[0, 0, candidates[0, 1].item()] = 10
    target_logits[1, 0, candidates[1, 1].item()] = 10
    # node1 distribution picks second draft token (node 2 token id)
    target_logits[0, 1, candidates[0, 2].item()] = 10
    target_logits[1, 1, candidates[1, 2].item()] = 10
    # node2 distribution picks bonus token id 9/10
    target_logits[0, 2, 9] = 10
    target_logits[1, 2, 10] = 10

    target_probs = F.softmax(target_logits, dim=-1)
    coins = torch.rand(bs, num_draft_tokens, device=device, dtype=torch.float32)
    coins_for_final_sampling = torch.rand(bs, device=device, dtype=torch.float32)

    predicts0 = torch.full(
        (bs * num_draft_tokens,), -1, dtype=torch.int32, device=device
    )
    accept_index0 = torch.full(
        (bs, num_spec_tokens), -1, dtype=torch.int32, device=device
    )
    accept_token_num0 = torch.full((bs,), 0, dtype=torch.int32, device=device)

    predicts1 = predicts0.clone()
    accept_index1 = accept_index0.clone()
    accept_token_num1 = accept_token_num0.clone()

    tree_speculative_sampling_target_only(
        predicts=predicts0,
        accept_index=accept_index0,
        accept_token_num=accept_token_num0,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coins_for_final_sampling,
        target_probs=target_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )

    chain_speculative_sampling_target_only(
        predicts=predicts1,
        accept_index=accept_index1,
        accept_token_num=accept_token_num1,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coins_for_final_sampling,
        target_probs=target_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )

    assert torch.equal(predicts0, predicts1)
    assert torch.equal(accept_index0, accept_index1)
    assert torch.equal(accept_token_num0, accept_token_num1)


def test_chain_speculative_sampling_reject_masks_token_in_final_sampling():
    """
    Construct coins to force a rejection at a specific step, then verify the final sampling
    will not sample the rejected token (mask == relu(q - p) equivalence for topk==1).
    """
    device = "cuda"
    bs = 1
    num_draft_tokens = 3  # root + 2 draft nodes in a chain
    vocab_size = 32
    num_spec_tokens = 3

    # Chain: 0 -> 1 -> 2, no siblings.
    t1 = 3  # token at node 1 (will be accepted)
    t2 = 5  # token at node 2 (will be rejected)
    alt = 31  # token that should be sampled after masking t2
    candidates = torch.tensor([[0, t1, t2]], dtype=torch.int64, device=device)
    retrive_index = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
    retrive_next_token = torch.tensor([[1, 2, -1]], dtype=torch.int64, device=device)
    retrive_next_sibling = torch.full(
        (bs, num_draft_tokens), -1, dtype=torch.int64, device=device
    )

    # Build target_probs directly (no softmax) for full control:
    # - At root (slot 0): P(t1)=0.95 => accept by threshold_single.
    # - At node1 (slot 1): P(t2)=0.90, P(alt)=0.10. We will reject t2 (by threshold + coin), then final sampling
    #   must NOT return t2; it should return alt given the chosen coin.
    target_probs = torch.zeros(
        (bs, num_draft_tokens, vocab_size), dtype=torch.float32, device=device
    )
    target_probs[0, 0, 0] = 0.05
    target_probs[0, 0, t1] = 0.95
    target_probs[0, 1, t2] = 0.90
    target_probs[0, 1, alt] = 0.10
    target_probs[0, 2, 0] = 1.0

    # Coins:
    # - First decision uses uniform_samples[0,0] but accept is forced by threshold_single.
    # - Second decision (for node2) uses uniform_samples[0,1] and must reject:
    #   coin=0.99, prob_acc=P(t2)=0.90, threshold_acc=1.0 => coin > prob_acc, and threshold_single > 0.90 => reject.
    coins = torch.tensor([[0.0, 0.99, 0.0]], dtype=torch.float32, device=device)
    # Final sampling coin: coin=0.5
    # - Without masking: u=0.5*1.0=0.5 and since rejected token (id=5) appears before alt (id=31)
    #   with prob 0.9, it would be sampled.
    # - With masking: sum=0.1, u=0.05, only alt has mass => alt is sampled.
    coins_for_final_sampling = torch.tensor([0.5], dtype=torch.float32, device=device)

    predicts0 = torch.full(
        (bs * num_draft_tokens,), -1, dtype=torch.int32, device=device
    )
    accept_index0 = torch.full(
        (bs, num_spec_tokens), -1, dtype=torch.int32, device=device
    )
    accept_token_num0 = torch.full((bs,), 0, dtype=torch.int32, device=device)

    predicts1 = predicts0.clone()
    accept_index1 = accept_index0.clone()
    accept_token_num1 = accept_token_num0.clone()

    # Make sure t1 is accepted (0.95 >= 0.91) but t2 is NOT (0.90 < 0.91).
    threshold_single = 0.91
    threshold_acc = 1.0

    tree_speculative_sampling_target_only(
        predicts=predicts0,
        accept_index=accept_index0,
        accept_token_num=accept_token_num0,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coins_for_final_sampling,
        target_probs=target_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )

    chain_speculative_sampling_target_only(
        predicts=predicts1,
        accept_index=accept_index1,
        accept_token_num=accept_token_num1,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coins_for_final_sampling,
        target_probs=target_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )

    # Sanity: both paths should match for topk==1
    assert torch.equal(predicts0, predicts1)
    assert torch.equal(accept_index0, accept_index1)
    assert torch.equal(accept_token_num0, accept_token_num1)

    # Node 1 token is accepted and written to predicts[0]
    assert predicts0[0].item() == t1
    # Final sampled token is written at last_accepted_retrive_idx (node 1's retrive_index == 1)
    final_token = predicts0[1].item()
    assert final_token != t2
    assert final_token == alt
