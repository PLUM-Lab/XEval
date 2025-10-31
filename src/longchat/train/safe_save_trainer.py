import os
from pathlib import Path
from packaging import version
from transformers import Trainer, is_torch_tpu_available
from trl import SFTTrainer
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_sagemaker_mp_enabled, WEIGHTS_NAME, logging
from transformers.trainer_utils import ShardedDDPOption
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
from typing import Optional
from transformers.trainer_pt_utils import get_parameter_names
from transformers.dependency_versions_check import dep_version_check
from transformers.integrations import (
    get_reporting_integration_callbacks,
    hp_params,
    is_fairscale_available,
)
if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.nn.wrap import auto_wrap
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

logger = logging.get_logger(__name__)

class SafeSaveTrainer(Trainer):
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.
        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif (
            ShardedDDPOption.ZERO_DP_2 in self.args.sharded_ddp
            or ShardedDDPOption.ZERO_DP_3 in self.args.sharded_ddp
            or self.fsdp is not None
        ):
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = self.model.state_dict()

            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
        elif self.deepspeed:
            # this takes care of everything as long as we aren't under zero3
            if self.args.should_save:
                self._save(output_dir)

            if is_deepspeed_zero3_enabled():
                # It's too complicated to try to override different places where the weights dump gets
                # saved, so since under zero3 the file is bogus, simply delete it. The user should
                # either user deepspeed checkpoint to resume or to recover full weights use
                # zero_to_fp32.py stored in the checkpoint.
                if self.args.should_save:
                    file = os.path.join(output_dir, WEIGHTS_NAME)
                    if os.path.isfile(file):
                        # logger.info(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")
                        os.remove(file)

                # now save the real model if stage3_gather_16bit_weights_on_model_save=True
                # if false it will not be saved.
                # This must be called on all ranks
                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    logger.warning(
                        "deepspeed.save_16bit_model didn't save the model, since"
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    self.deepspeed.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")
    
    # for now let's not use different lr
    # def create_optimizer(self):
    #     """
    #     Setup the optimizer.

    #     We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    #     Trainer's init through `optimizers`, or subclass and override this method in a subclass.
    #     """
    #     opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

    #     if self.optimizer is None:
    #         decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
    #         decay_parameters = [name for name in decay_parameters if "bias" not in name]
    #         emb_parameters = [name for name in decay_parameters if "embed_tokens" in name]
    #         optimizer_grouped_parameters = [
    #             {
    #                 "params": [
    #                     p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and not n in emb_parameters)
    #                 ],
    #                 "weight_decay": self.args.weight_decay,
    #             },
    #             {
    #                 "params": [
    #                     p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and n in emb_parameters)
    #                 ],
    #                 "weight_decay": self.args.weight_decay,
    #                 "lr": 2e-4
    #             },
    #             {
    #                 "params": [
    #                     p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
    #                 ],
    #                 "weight_decay": 0.0,
    #             },
    #         ]

    #         optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

    #         if self.sharded_ddp == ShardedDDPOption.SIMPLE:
    #             self.optimizer = OSS(
    #                 params=optimizer_grouped_parameters,
    #                 optim=optimizer_cls,
    #                 **optimizer_kwargs,
    #             )
    #         else:
    #             self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    #             if optimizer_cls.__name__ == "Adam8bit":
    #                 import bitsandbytes

    #                 manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

    #                 skipped = 0
    #                 for module in opt_model.modules():
    #                     if isinstance(module, nn.Embedding):
    #                         skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
    #                         logger.info(f"skipped {module}: {skipped/2**20}M params")
    #                         manager.register_module_override(module, "weight", {"optim_bits": 32})
    #                         logger.debug(f"bitsandbytes: will optimize {module} in fp32")
    #                 logger.info(f"skipped: {skipped/2**20}M params")

    #     if is_sagemaker_mp_enabled():
    #         self.optimizer = smp.DistributedOptimizer(self.optimizer)

    #     return self.optimizer