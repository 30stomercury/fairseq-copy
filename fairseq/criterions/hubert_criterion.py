# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


def CrossEntropy(pred, target, pred_is_log=False, reduction="none"):
    """cross-entropy loss with soft targets
    args
    ----
    pred : (B, T, N)
        pred probs.
    target: (B, T, N)
        target probs.
    """
    N = target.size(0)
    eps = 1e-12
    # [B, T, D] -> [B, T]
    pred_stable_log = pred
    if not pred_is_log:
        pred_stable = pred.clone().clamp(min=eps, max=1 - eps)
        pred_stable_log = pred_stable.log()

    cross_entropy_masked = torch.where(
        (pred > 0).to(pred.device),
        target * pred_stable_log,
        torch.tensor(0., device=pred.device, dtype=torch.float)
    )
    cross_entropy = - torch.sum(cross_entropy_masked, -1)

    if reduction == "none":
	    return cross_entropy
    elif reduction == "sum":
        return cross_entropy.sum()
    elif reduction == "mean":
        return cross_entropy.mean()
    else:
        raise ValueError(f"{reduction} reduction not implemented")

def vlb(
        pred_outs, 
        target_latent_probs, 
        target_outs, 
        target_losses, 
        y,
        mask,
    ):
    """
    Args
    ----
    pred_outs : (BT, D)
    target_latent_probs : (BT, N)
    target_outs: (BT, D)
    target_losses: (BT, N)
        The reconstruction loss of each mode are collected.
    mask : (BT, N)
    """
    losses = {}
    BT, _ = pred_outs.size()
    num_codes = target_latent_probs.size(-1)

    # cross entropy
    ce_loss_batch = CrossEntropy(
        pred_outs, target_latent_probs, pred_is_log=True, reduction='none'
    ) * mask.squeeze(-1)
    ce_loss = (ce_loss_batch.sum() / mask.sum())

    # entropy
    ent_loss_batch = CrossEntropy(
        target_latent_probs, target_latent_probs, reduction='none'
    ) * mask.squeeze(-1)
    ent_loss = (ent_loss_batch.sum() / mask.sum())

    # reconstruction
    rec_loss_batch = 0.5 * (target_outs - y)**2 
    rec_loss_batch.masked_fill_(mask.unsqueeze(-1)==0, 0.0)
    rec_loss = rec_loss_batch.sum() / mask.sum()

    # collect results
    loss = ce_loss - ent_loss + rec_loss
    losses["loss"] = loss
    losses['ent'] = ent_loss
    losses['ce_loss'] = ce_loss
    losses['rec_loss'] = rec_loss
    return losses



@dataclass
class HubertCriterionConfig(FairseqDataclass):
    pred_masked_weight: float = field(
        default=1.0,
        metadata={"help": "weight for predictive loss for masked frames"},
    )
    pred_nomask_weight: float = field(
        default=0.0,
        metadata={"help": "weight for predictive loss for unmasked frames"},
    )
    loss_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "weights for additional loss terms (not first one)"},
    )
    log_keys: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "output keys to log"},
    )

@dataclass
class HubertvlbCriterionConfig(FairseqDataclass):
    pred_masked_weight: float = field(
        default=1.0,
        metadata={"help": "weight for predictive loss for masked frames"},
    )
    pred_nomask_weight: float = field(
        default=0.0,
        metadata={"help": "weight for predictive loss for unmasked frames"},
    )
    loss_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "weights for additional loss terms (not first one)"},
    )
    marginalization: bool = field(
        default=False,
        metadata={"help": "marginalize rec loss or not"}
    )
    log_keys: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "output keys to log"},
    )


@register_criterion("hubert", dataclass=HubertCriterionConfig)
class HubertCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        pred_masked_weight,
        pred_nomask_weight,
        loss_weights=None,
        log_keys=None,
    ):
        super().__init__(task)
        self.pred_masked_weight = pred_masked_weight
        self.pred_nomask_weight = pred_nomask_weight
        self.loss_weights = loss_weights
        self.log_keys = [] if log_keys is None else log_keys

    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(target_list=sample["target_list"], **sample["net_input"])
        loss = 0.0
        sample_size = 0
        logging_output = {}
        reduction = "sum" if reduce else "none"

        loss_m_list = []
        logp_m_list = model.get_logits(net_output, True)
        targ_m_list = model.get_targets(net_output, True)
        assert self.pred_masked_weight == 0 or len(logp_m_list) > 0
        for i, (logp_m, targ_m) in enumerate(zip(logp_m_list, targ_m_list)):
            loss_m = F.cross_entropy(logp_m, targ_m, reduction=reduction)
            loss_m_list.append(loss_m)
            logging_output[f"loss_m_{i}"] = loss_m.detach().item()
        if self.pred_masked_weight > 0:
            loss += self.pred_masked_weight * sum(loss_m_list)
            sample_size += targ_m_list[0].numel()

        loss_u_list = []
        logp_u_list = model.get_logits(net_output, False)
        targ_u_list = model.get_targets(net_output, False)
        assert self.pred_nomask_weight == 0 or len(logp_u_list) > 0
        for i, (logp_u, targ_u) in enumerate(zip(logp_u_list, targ_u_list)):
            loss_u = F.cross_entropy(logp_u, targ_u, reduction=reduction)
            loss_u_list.append(loss_u)
            logging_output[f"loss_u_{i}"] = loss_u.detach().item()
        if self.pred_nomask_weight > 0:
            loss += self.pred_nomask_weight * sum(loss_u_list)
            sample_size += targ_u_list[0].numel()

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses, names = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
                names = [names]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(
                self.loss_weights
            ), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, n, coef in zip(extra_losses, names, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    logging_output[f"loss_{n}"] = p.item()

        logging_output = {
            "loss": loss.item() if reduce else loss,
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            **logging_output,
        }

        for lk in self.log_keys:
            if lk in net_output:
                logging_output[lk] = float((net_output[lk]))

        def compute_correct(logits):
            if logits.numel() == 0:
                return 0, 0
            else:
                assert logits.dim() > 1, logits.shape
                max = logits.argmax(-1) == 0
                min = logits.argmin(-1) == 0
                both = max & min
                corr = max.long().sum().item() - both.long().sum().item()
                count = max.numel()
                return corr, count

        with torch.no_grad():
            for i, logp_m in enumerate(logp_m_list):
                corr_m, count_m = compute_correct(logp_m)
                logging_output[f"correct_m_{i}"] = corr_m
                logging_output[f"count_m_{i}"] = count_m

            for i, logp_u in enumerate(logp_u_list):
                corr_u, count_u = compute_correct(logp_u)
                logging_output[f"correct_u_{i}"] = corr_u
                logging_output[f"count_u_{i}"] = count_u

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training (copied from normal cross entropy)."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

        counts = {}
        for lk in logging_outputs[0].keys():
            if lk.startswith("count_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val)
                counts[lk] = val

        for lk in logging_outputs[0].keys():
            if lk.startswith("loss_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / sample_size / math.log(2), round=3)
            elif lk.startswith("correct_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / counts[re.sub("correct", "count", lk)])

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError()

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False


@register_criterion("hubertvlb", dataclass=HubertvlbCriterionConfig)
class HubertCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        pred_masked_weight,
        pred_nomask_weight,
        loss_weights=None,
        marginalization=False,
        log_keys=None,
    ):
        super().__init__(task)
        self.pred_masked_weight = pred_masked_weight
        self.pred_nomask_weight = pred_nomask_weight
        self.loss_weights = loss_weights
        self.marginalization = marginalization
        self.log_keys = [] if log_keys is None else log_keys

    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(target_list=sample["target_list"], **sample["net_input"])
        loss = 0.0
        sample_size = 0
        logging_output = {}
        reduction = "sum" if reduce else "none"

        loss_m_list = []
        logp_m_list = model.get_logits(net_output, True)
        targ_m_list = model.get_targets(net_output, True)
        assert self.pred_masked_weight == 0 or len(logp_m_list) > 0
        for i, (logp_m, targ_m) in enumerate(zip(logp_m_list, targ_m_list)):
            target_probs = net_output['q_results']['latent_probs']
            N = target_probs.size(-1)
            # kl term
            loss_m = CrossEntropy(
                    logp_m[..., :N].softmax(-1),
                    target_probs, reduction=reduction)
            loss_ent = CrossEntropy(
                    target_probs, 
                    target_probs, reduction=reduction)
            loss_m_list.append(loss_m)


        if self.pred_masked_weight > 0:
            sample_size += targ_m_list[0].numel()
            #loss_m = self.pred_masked_weight * sum(loss_m_list) / sample_size
            #loss_ent = self.pred_masked_weight * loss_ent / sample_size
            loss_m = self.pred_masked_weight * sum(loss_m_list)
            loss_ent = self.pred_masked_weight * loss_ent
            loss += loss_m
            loss -= loss_ent
            logging_output[f"loss_m_{i}"] = loss_m.detach().item()
            logging_output[f"loss_ent_{i}"] = loss_ent.detach().item()


        if self.marginalization:
            # (B * T, 1, N) x (B * T, N, 1)
            target_losses = -0.5 * net_output['q_results']['downstream_losses']
            # marginalization
            rec_loss_batch = target_probs.unsqueeze(1).bmm(
                target_losses.unsqueeze(-1)
            ).squeeze()
            rec_loss = rec_loss_batch.sum()
        else:
            # reconstruction
            fbanks = net_output['fbanks']
            rec_loss_batch = 0.5 * (
                    net_output['q_results']['q']- fbanks
            ).pow(2).sum(-1)
            rec_loss = (rec_loss_batch).sum() 
            #rec_loss = (rec_loss_batch / sample_size).sum() 

        loss += rec_loss
        logging_output[f"loss_rec"] = rec_loss.detach().item() 

        loss_u_list = []
        logp_u_list = model.get_logits(net_output, False)
        targ_u_list = model.get_targets(net_output, False)
        assert self.pred_nomask_weight == 0 or len(logp_u_list) > 0
        for i, (logp_u, targ_u) in enumerate(zip(logp_u_list, targ_u_list)):
            loss_u = F.cross_entropy(logp_u, targ_u, reduction=reduction)
            loss_u_list.append(loss_u)
            logging_output[f"loss_u_{i}"] = loss_u.detach().item()
        # always 0
        if self.pred_nomask_weight > 0:
            loss += self.pred_nomask_weight * sum(loss_u_list)
            sample_size += targ_u_list[0].numel()

        # sample_size
        #sample_size = 1
        #logging_output[f"sample_size"] = 1
        #logging_output[f"ntokens"] = 1

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses, names = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
                names = [names]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(
                self.loss_weights
            ), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, n, coef in zip(extra_losses, names, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() 
                    loss += p * sample_size
                    logging_output[f"loss_{n}"] = p.item()

        logging_output = {
            "loss": loss.item() if reduce else loss,
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            **logging_output,
        }

        for lk in self.log_keys:
            if lk in net_output:
                logging_output[lk] = float((net_output[lk]))

        def compute_correct(logits):
            if logits.numel() == 0:
                return 0, 0
            else:
                assert logits.dim() > 1, logits.shape
                max = logits.argmax(-1) == 0
                min = logits.argmin(-1) == 0
                both = max & min
                corr = max.long().sum().item() - both.long().sum().item()
                count = max.numel()
                return corr, count

        with torch.no_grad():
            for i, logp_m in enumerate(logp_m_list):
                corr_m, count_m = compute_correct(logp_m)
                logging_output[f"correct_m_{i}"] = corr_m
                logging_output[f"count_m_{i}"] = count_m

            for i, logp_u in enumerate(logp_u_list):
                corr_u, count_u = compute_correct(logp_u)
                logging_output[f"correct_u_{i}"] = corr_u
                logging_output[f"count_u_{i}"] = count_u

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training (copied from normal cross entropy)."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

        counts = {}
        for lk in logging_outputs[0].keys():
            if lk.startswith("count_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val)
                counts[lk] = val

        for lk in logging_outputs[0].keys():
            if lk.startswith("loss_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / sample_size / math.log(2), round=3)
            elif lk.startswith("correct_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / counts[re.sub("correct", "count", lk)])

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError()

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
