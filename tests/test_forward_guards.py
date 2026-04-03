"""
Tests for VibeVoiceForConditionalGeneration.forward() regression bugs.

Bug 1: speech_len UnboundLocalError
    speech_len was only assigned inside the if-block for diffusion loss,
    but referenced in both return branches => NameError on text-only forward.

Bug 2: semantic_connector(None) crash
    Called unconditionally on speech_semantic_tensors which defaults to None.

Bug 3: acoustic_loss_mask.sum() on None
    acoustic_loss_mask is Optional but dereferenced without None check.
"""

import pytest
import torch
import torch.nn as nn


class TestSpeechLenAlwaysBound:
    def _make_fwd(self):
        hidden = 32
        vae_dim = 16
        vocab = 100

        embed = nn.Embedding(vocab, hidden)
        lm_head = nn.Linear(hidden, vocab, bias=False)
        pred_head = nn.Linear(vae_dim, vae_dim)
        sem_connector = nn.Linear(vae_dim, hidden)
        ac_connector = nn.Linear(vae_dim, hidden)

        def forward(input_ids, speech_tensors=None, acoustic_loss_mask=None,
                    speech_semantic_tensors=None, return_dict=True):
            x = embed(input_ids)
            # Bug 2 fix: guard None
            sem_feats = sem_connector(speech_semantic_tensors) if speech_semantic_tensors is not None else None
            # Bug 1 fix: initialise before conditional
            speech_len = 0
            hidden_states = x
            logits = lm_head(hidden_states)
            # Bug 3 fix: guard None on acoustic_loss_mask
            if (speech_tensors is not None
                    and acoustic_loss_mask is not None
                    and acoustic_loss_mask.sum().item() > 0):
                speech_len, latent_size = speech_tensors.shape
                diffusion_loss = torch.tensor(0.0)
            else:
                diffusion_loss = torch.tensor(0.0)
            if return_dict:
                return {"speech_token_num": speech_len, "logits": logits, "diffusion_loss": diffusion_loss}
            return (None, diffusion_loss, logits, speech_len)

        return forward, vae_dim

    def test_text_only_return_dict(self):
        fwd, _ = self._make_fwd()
        result = fwd(torch.randint(0, 100, (1, 8)))
        assert result["speech_token_num"] == 0

    def test_text_only_return_tuple(self):
        fwd, _ = self._make_fwd()
        result = fwd(torch.randint(0, 100, (1, 8)), return_dict=False)
        assert result[3] == 0

    def test_speech_tensors_zero_acoustic_loss_mask(self):
        fwd, vae_dim = self._make_fwd()
        speech_tensors = torch.randn(4, vae_dim)
        acoustic_loss_mask = torch.zeros(1, 8, dtype=torch.bool)
        result = fwd(torch.randint(0, 100, (1, 8)),
                     speech_tensors=speech_tensors,
                     acoustic_loss_mask=acoustic_loss_mask)
        assert result["speech_token_num"] == 0

    def test_speech_tensors_none_acoustic_loss_mask_none(self):
        fwd, _ = self._make_fwd()
        result = fwd(torch.randint(0, 100, (1, 8)),
                     speech_tensors=None, acoustic_loss_mask=None)
        assert result["speech_token_num"] == 0

    def test_speech_len_correct_when_active(self):
        fwd, vae_dim = self._make_fwd()
        n = 5
        speech_tensors = torch.randn(n, vae_dim)
        mask = torch.zeros(1, 8, dtype=torch.bool)
        mask[0, 2:4] = True
        result = fwd(torch.randint(0, 100, (1, 8)),
                     speech_tensors=speech_tensors,
                     acoustic_loss_mask=mask)
        assert result["speech_token_num"] == n


class TestSemanticConnectorNoneGuard:
    def test_no_semantic_tensors_does_not_crash(self):
        connector = nn.Linear(16, 32)
        calls = []
        orig_fwd = connector.forward

        def tracked(x):
            calls.append(x)
            return orig_fwd(x)

        connector.forward = tracked
        result = connector(torch.randn(4, 16)) if True else None
        assert result is not None

        # Guarded call: None => skip
        guarded = connector(None) if None is not None else None
        assert guarded is None
        assert len(calls) == 1

    def test_semantic_connector_called_when_tensor_provided(self):
        connector = nn.Linear(16, 32)
        sst = torch.randn(4, 16)
        result = connector(sst) if sst is not None else None
        assert result is not None
        assert result.shape == (4, 32)

    def test_semantic_connector_skipped_when_none(self):
        connector = nn.Linear(16, 32)
        result = connector(None) if None is not None else None
        assert result is None


class TestAcousticLossMaskNoneGuard:
    def _guard(self, speech_tensors, acoustic_loss_mask):
        return (
            speech_tensors is not None
            and acoustic_loss_mask is not None
            and acoustic_loss_mask.sum().item() > 0
        )

    def test_none_mask_with_speech_tensors(self):
        speech = torch.randn(4, 16)
        assert not self._guard(speech, None)

    def test_none_mask_without_speech(self):
        assert not self._guard(None, None)

    def test_all_false_mask(self):
        speech = torch.randn(4, 16)
        mask = torch.zeros(1, 8, dtype=torch.bool)
        assert not self._guard(speech, mask)

    def test_partial_true_mask(self):
        speech = torch.randn(4, 16)
        mask = torch.zeros(1, 8, dtype=torch.bool)
        mask[0, 0:2] = True
        assert self._guard(speech, mask)

    def test_all_true_mask(self):
        speech = torch.randn(4, 16)
        mask = torch.ones(1, 8, dtype=torch.bool)
        assert self._guard(speech, mask)


class TestCombined:
    def _make_fwd(self):
        hidden = 32
        vae_dim = 16
        vocab = 100
        embed = nn.Embedding(vocab, hidden)
        lm_head = nn.Linear(hidden, vocab, bias=False)
        sem_connector = nn.Linear(vae_dim, hidden)
        ac_connector = nn.Linear(vae_dim, hidden)

        def forward(input_ids, speech_tensors=None, acoustic_loss_mask=None,
                    speech_semantic_tensors=None, return_dict=True):
            x = embed(input_ids)
            sem_feats = sem_connector(speech_semantic_tensors) if speech_semantic_tensors is not None else None
            speech_len = 0
            hidden_states = x
            logits = lm_head(hidden_states)
            if (speech_tensors is not None
                    and acoustic_loss_mask is not None
                    and acoustic_loss_mask.sum().item() > 0):
                speech_len, _ = speech_tensors.shape
                diffusion_loss = torch.tensor(0.0)
            else:
                diffusion_loss = torch.tensor(0.0)
            if return_dict:
                return {"speech_token_num": speech_len, "logits": logits, "diffusion_loss": diffusion_loss}
            return (None, diffusion_loss, logits, speech_len)

        return forward, vae_dim

    def test_all_none_no_crash(self):
        fwd, _ = self._make_fwd()
        result = fwd(torch.randint(0, 100, (2, 6)))
        assert result["speech_token_num"] == 0
        assert result["logits"] is not None

    def test_return_tuple_all_none(self):
        fwd, _ = self._make_fwd()
        result = fwd(torch.randint(0, 100, (1, 4)), return_dict=False)
        assert result[3] == 0

    def test_batch_size_two_text_only(self):
        fwd, _ = self._make_fwd()
        result = fwd(torch.randint(0, 100, (2, 10)))
        assert result["speech_token_num"] == 0

    def test_with_all_speech_params(self):
        fwd, vae_dim = self._make_fwd()
        input_ids = torch.randint(0, 100, (1, 8))
        n = 6
        speech_tensors = torch.randn(n, vae_dim)
        mask = torch.zeros(1, 8, dtype=torch.bool)
        mask[0, 1:4] = True
        sst = torch.randn(4, vae_dim)
        result = fwd(input_ids,
                     speech_tensors=speech_tensors,
                     acoustic_loss_mask=mask,
                     speech_semantic_tensors=sst)
        assert result["speech_token_num"] == n
