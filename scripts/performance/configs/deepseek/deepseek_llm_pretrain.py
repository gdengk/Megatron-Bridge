# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from utils.overrides import set_workload_base_configs
from utils.precision import get_precision_config
from utils.utils import get_workload_base_config

from megatron.bridge.recipes.deepseek.deepseek_v3 import deepseek_v3_pretrain_config as pretrain_config
from megatron.bridge.training.config import ConfigContainer


logger = logging.getLogger(__name__)


def set_deepseek_v3_common_configs(cfg: ConfigContainer, moe_a2a_overlap: bool = False) -> None:
    """Set common performance configurations for all DeepSeek-V3 configs."""
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.moe_router_force_load_balancing = True
    cfg.model.moe_router_dtype = 'bf16'


def deepseek_v3_pretrain_config_gb300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="gb300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config(
        mock=mock,
        precision_config=precision_config,
        pipeline_model_parallel_size=base_cfg.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=base_cfg.virtual_pipeline_model_parallel_size,
        moe_flex_dispatcher_backend=base_cfg.moe_flex_dispatcher_backend,
        layout="Et*4|(t*4|)*14tmL",
    )
    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap.overlap_grad_reduce = True

    # Setting num_workers and pin_memory to 0 and False respectively gives better performance.
    # we are debugging this and might change this in the future.
    def get_data():
        # Parameters
        seq_length = 4096
        seed = 1234
        # tmpdatapath="/lustre/share/coreai_mlperf_training/data/c4/dsv3_8b"
        val_test_path = "/lustre/share/coreai_mlperf_training/data/c4/dsv3_8b/c4-validation-91205-samples.en_text_document"

        is_8b_dataset = True
        r = [6] if is_8b_dataset else [6, 7]
        train_datasets = [f"/lustre/share/coreai_mlperf_training/data/c4/dsv3_8b/c4-train.en_{idx}_text_document" for idx in r]
        train_datasets_weights = [50] * len(r)

        data_paths = [(train_datasets, train_datasets_weights), ([val_test_path], None), ([val_test_path], None)]

        from megatron.bridge.training.config import GPTDatasetConfig

        return GPTDatasetConfig(
            dataloader_type="single",
            blend_per_split=data_paths,
            sequence_length=seq_length,
            random_seed=seed,
            num_workers=8,
            path_to_cache="/lustre/fsw/coreai_dlalgo_llm/gdeng/mbridge/workspace/dsv3test_nemo2602-gb300/cache",
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
        )
    cfg.dataset = get_data()
    cfg.dataset.num_workers = 0
    cfg.dataset.pin_memory = False
    cfg.model.cuda_graph_warmup_steps = 0

    if precision == "fp8_mx":  # keeping this eanbled causes NaN grad norm
        cfg.comm_overlap.overlap_param_gather = True
        cfg.ddp.overlap_param_gather = True
        cfg.optimizer.overlap_param_gather = True
        cfg.optimizer.overlap_param_gather_with_optimizer_step = False
        cfg.comm_overlap.overlap_param_gather_with_optimizer_step = False

    return cfg


def deepseek_v3_pretrain_config_gb200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="gb200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config(
        mock=mock,
        precision_config=precision_config,
        pipeline_model_parallel_size=base_cfg.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=base_cfg.virtual_pipeline_model_parallel_size,
        moe_flex_dispatcher_backend=base_cfg.moe_flex_dispatcher_backend,
        layout="Et*4|(t*4|)*14tmL",
    )
    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap.overlap_grad_reduce = True

    # Setting num_workers and pin_memory to 0 and False respectively gives better performance.
    # we are debugging this and might change this in the future.
    def get_data():
        # Parameters
        seq_length = 4096
        seed = 1234
        # tmpdatapath="/lustre/share/coreai_mlperf_training/data/c4/dsv3_8b"
        val_test_path = "/lustre/share/coreai_mlperf_training/data/c4/dsv3_8b/c4-validation-91205-samples.en_text_document"

        is_8b_dataset = True
        r = [6] if is_8b_dataset else [6, 7]
        train_datasets = [f"/lustre/share/coreai_mlperf_training/data/c4/dsv3_8b/c4-train.en_{idx}_text_document" for idx in r]
        train_datasets_weights = [50] * len(r)

        data_paths = [(train_datasets, train_datasets_weights), ([val_test_path], None), ([val_test_path], None)]

        from megatron.bridge.training.config import GPTDatasetConfig

        return GPTDatasetConfig(
            dataloader_type="single",
            blend_per_split=data_paths,
            sequence_length=seq_length,
            random_seed=seed,
            num_workers=8,
            path_to_cache="/lustre/fsw/coreai_dlalgo_llm/gdeng/mbridge/workspace/dsv3test_nemo2602-gb300/cache",
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
        )
    cfg.dataset = get_data()
    cfg.dataset.num_workers = 0
    cfg.dataset.pin_memory = False
    cfg.model.cuda_graph_warmup_steps = 0

    if precision == "fp8_mx":  # keeping this eanbled causes NaN grad norm
        cfg.comm_overlap.overlap_param_gather = True
        cfg.ddp.overlap_param_gather = True
        cfg.optimizer.overlap_param_gather = True
        cfg.optimizer.overlap_param_gather_with_optimizer_step = False
        cfg.comm_overlap.overlap_param_gather_with_optimizer_step = False

    return cfg


def deepseek_v3_pretrain_config_b300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="b300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config(
        mock=mock,
        precision_config=precision_config,
        pipeline_model_parallel_size=base_cfg.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=base_cfg.virtual_pipeline_model_parallel_size,
        moe_flex_dispatcher_backend=base_cfg.moe_flex_dispatcher_backend,
        layout=None,
    )
    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap.overlap_grad_reduce = True

    return cfg


def deepseek_v3_pretrain_config_b200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="b200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config(
        mock=mock,
        precision_config=precision_config,
        pipeline_model_parallel_size=base_cfg.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=base_cfg.virtual_pipeline_model_parallel_size,
        moe_flex_dispatcher_backend=base_cfg.moe_flex_dispatcher_backend,
        layout=None,
    )
    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap.overlap_grad_reduce = True

    return cfg


def deepseek_v3_pretrain_config_singlenode(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """Single-node (4-GPU), baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="singlenode",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config(
        mock=mock,
        precision_config=precision_config,
        pipeline_model_parallel_size=base_cfg.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=base_cfg.virtual_pipeline_model_parallel_size,
        moe_flex_dispatcher_backend=base_cfg.moe_flex_dispatcher_backend,
        layout=None,
    )
    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    def get_data():
        # Parameters
        seq_length = 4096
        seed = 1234
        val_test_path = "/lustre/share/coreai_mlperf_training/data/c4/dsv3_8b/c4-validation-91205-samples.en_text_document"

        is_8b_dataset = True
        r = [6] if is_8b_dataset else [6, 7]
        train_datasets = [f"/lustre/share/coreai_mlperf_training/data/c4/dsv3_8b/c4-train.en_{idx}_text_document" for idx in r]
        train_datasets_weights = [50] * len(r)

        data_paths = [(train_datasets, train_datasets_weights), ([val_test_path], None), ([val_test_path], None)]

        from megatron.bridge.training.config import GPTDatasetConfig

        return GPTDatasetConfig(
            dataloader_type="single",
            blend_per_split=data_paths,
            sequence_length=seq_length,
            random_seed=seed,
            num_workers=8,
            path_to_cache="/lustre/fsw/coreai_dlalgo_llm/gdeng/mbridge/workspace/dsv3test_nemo2602-gb300/cache",
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
        )
    cfg.dataset = get_data()
    cfg.dataset.num_workers = 0
    cfg.dataset.pin_memory = False
    cfg.model.cuda_graph_warmup_steps = 0

    # single node setup
    cfg.model.num_layers = 4
    cfg.model.num_moe_experts = 8
    cfg.model.moe_layer_freq=[0] + [1] * 3
    cfg.model.moe_router_group_topk=1
    cfg.model.moe_router_num_groups=1



    return cfg


def deepseek_v3_pretrain_config_h100(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """H100, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="h100",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config(
        mock=mock,
        precision_config=precision_config,
        pipeline_model_parallel_size=base_cfg.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=base_cfg.virtual_pipeline_model_parallel_size,
        moe_flex_dispatcher_backend=base_cfg.moe_flex_dispatcher_backend,
        layout="Et|(tt|)*30mL",
    )
    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    # Disabling to avoid functional errors. TODO: Test with it enabled and keep it enabled if it works.
    cfg.comm_overlap.overlap_grad_reduce = False

    return cfg
