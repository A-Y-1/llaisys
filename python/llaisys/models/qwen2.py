from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..libllaisys.models import load_qwen2
from ..libllaisys.models import Qwen2Meta
from ..libllaisys import llaisysTensor_t

import ctypes
import json
import torch
from pathlib import Path
import safetensors

load_qwen2(LIB_LLAISYS)

class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        # path check

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist.")
        else:
            self.model_path = model_path

        config_path = model_path / "config.json" 
        weights_path = model_path / "model.safetensors"
        if not config_path.exists():
            raise FileNotFoundError(f"Config path {config_path} does not exist.")
        elif not weights_path.exists():
            raise FileNotFoundError(f"Weights path {weights_path} does not exist.")
        
        # model initialize
        self.device = device
        self.device_id = 0
        with open(config_path, "r") as f:
            config = json.load(f)
        
        self.eos_token_id = config.get("eos_token_id")
        self.hidden_size = config.get("hidden_size")
        self.intermediate_size = config.get("intermediate_size")
        self.max_position_embeddings = config.get("max_position_embeddings")
        self.num_attention_heads = config.get("num_attention_heads")
        self.num_hidden_layers = config.get("num_hidden_layers")
        self.num_key_value_heads = config.get("num_key_value_heads")
        self.rms_norm_eps = config.get("rms_norm_eps")
        self.rope_theta = config.get("rope_theta")
        self.torch_dtype = config.get("torch_dtype")
        self.vocab_size = config.get("vocab_size")
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.kvhead_dimv = self.head_dim

        assert self.torch_dtype == "bfloat16", "Bf16 support only."

        if self.device == DeviceType.CPU:
            self.data_type = DataType.F32
            print("[Llaisys] Running on CPU, use FP32 for better performance.")
        else:
            self.data_type = DataType.BF16

        META = Qwen2Meta(
            dtype=self.data_type,
            nlayer=self.num_hidden_layers,
            hs=self.hidden_size,
            nh=self.num_attention_heads,
            nkvh=self.num_key_value_heads,
            dh=self.head_dim,
            di=self.intermediate_size,
            maxseq=self.max_position_embeddings,
            voc=self.vocab_size,
            epsilon=self.rms_norm_eps,
            theta=self.rope_theta,
            end_token=self.eos_token_id
        )

        device_ids = (ctypes.c_int * 1)(0)
        self.model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(META),
            ctypes.c_int(device),
            device_ids,
            ctypes.c_int(1)
        )
        if not self.model:
            raise RuntimeError("Failed to create Qwen2 model.")
        
        print("[Llaisys] Model initialize done.", flush=True)
        
        weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model)

        # Loading weights
        for file in sorted(model_path.glob("*.safetensors")):
            def cast_tensor_to_f32(tensor): 
                if self.device == DeviceType.CPU:
                    return tensor.to(torch.float32).contiguous()
                return tensor
            
            embed_tensor = {
                "model.embed_tokens.weight": "in_embed",
                "lm_head.weight": "out_embed",
                "model.norm.weight": "out_norm_w"
            }

            data_ = safetensors.safe_open(file, framework="torch", device="cpu")
            for name_ in embed_tensor.keys():
                filed = embed_tensor[name_]
                load_tensor = cast_tensor_to_f32(data_.get_tensor(name_))   
                LIB_LLAISYS.tensorLoad(getattr(weights.contents, filed), load_tensor.data_ptr())

            # load layer weights
            def load_layer_weights(filed, name):
                arr_ptr = getattr(weights.contents, filed)
                arr_type = llaisysTensor_t * self.num_hidden_layers
                arr = ctypes.cast(arr_ptr, ctypes.POINTER(arr_type)).contents
                for i in range(self.num_hidden_layers):
                    tensor_name = f"model.layers.{i}.{name}"
                    load_tensor = cast_tensor_to_f32(data_.get_tensor(tensor_name))
                    LIB_LLAISYS.tensorLoad(arr[i], load_tensor.data_ptr())

            layer_tasks = [
                ("attn_norm_w", "input_layernorm.weight"),
                ("attn_q_w", "self_attn.q_proj.weight"),
                ("attn_q_b", "self_attn.q_proj.bias"),
                ("attn_k_w", "self_attn.k_proj.weight"),
                ("attn_k_b", "self_attn.k_proj.bias"),
                ("attn_v_w", "self_attn.v_proj.weight"),
                ("attn_v_b", "self_attn.v_proj.bias"),
                ("attn_o_w", "self_attn.o_proj.weight"),
                ("mlp_norm_w", "post_attention_layernorm.weight"),
                ("mlp_gate_w", "mlp.gate_proj.weight"),
                ("mlp_up_w", "mlp.up_proj.weight"),
                ("mlp_down_w", "mlp.down_proj.weight"),
            ]

            for filed, suffix in layer_tasks:
                #print(f"[Llaisys] Processing layer-group: {suffix}...", flush=True)
                load_layer_weights(filed, suffix)

            print("[Llaisys] All weights loaded successfully.", flush=True)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        inputs = list(inputs)
        generated = inputs
        ntokens = len(generated)
        tokens = (ctypes.c_int64 * ntokens)(*generated)

        '''
        KV Cache 
        '''
        self.kcache = (llaisysTensor_t * self.num_hidden_layers)()
        self.vcache = (llaisysTensor_t * self.num_hidden_layers)()
        shape = (ctypes.c_size_t * 3)(
            max_new_tokens + len(generated),
            self.num_key_value_heads,
            self.kvhead_dimv
        )

        for i in range(self.num_hidden_layers):
            self.kcache[i] = LIB_LLAISYS.tensorCreate(shape, 3, self.data_type, self.device, self.device_id)
            self.vcache[i] = LIB_LLAISYS.tensorCreate(shape, 3, self.data_type, self.device, self.device_id)

        '''
        Prefilling
        '''

        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self.model,
            tokens,
            ctypes.c_size_t(ntokens),
            ctypes.cast(self.kcache, ctypes.POINTER(llaisysTensor_t)),
            ctypes.cast(self.vcache, ctypes.POINTER(llaisysTensor_t)),
            ctypes.c_size_t(0)
        )
        generated.append(next_token)
        print("[Llaisys] Prefill done", flush=True)

        '''
        Decoding
        '''
        for _ in range(max_new_tokens - 1):
            past_len = len(generated) - 1
            if next_token == self.eos_token_id:
                break
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.model,
                (ctypes.c_int64 * 1)(next_token),
                ctypes.c_size_t(1),
                ctypes.cast(self.kcache, ctypes.POINTER(llaisysTensor_t)),
                ctypes.cast(self.vcache, ctypes.POINTER(llaisysTensor_t)),
                past_len
            )
            generated.append(next_token)

        return generated
