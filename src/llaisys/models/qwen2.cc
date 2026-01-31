#include "llaisys/models/qwen2.h"
#include "llaisys/ops.h"
#include "../../utils.hpp"

#include <cstring>
#include <cmath>
#include <iostream>
#include <vector> 
#include <cstdio>
using std::cout;
using std::endl;

__C {

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice){
        struct LlaisysQwen2Model *qwen2Model = (struct LlaisysQwen2Model *)malloc(sizeof(LlaisysQwen2Model));
        qwen2Model->device = device;
        qwen2Model->ndevice = ndevice;

        if (meta == nullptr || device_ids == nullptr) {
            fprintf(stderr, "[ERROR] Qwen2ModelCreate: invalid ptr.\n");
            return nullptr; 
        }
        qwen2Model->meta = (LlaisysQwen2Meta *)malloc(sizeof(LlaisysQwen2Meta));
        qwen2Model->device_ids = (int *)malloc(sizeof(int) * ndevice);
        qwen2Model->weights = (LlaisysQwen2Weights *)malloc(sizeof(LlaisysQwen2Weights));

        memcpy(qwen2Model->meta, meta, sizeof(LlaisysQwen2Meta));
        memcpy(qwen2Model->device_ids, device_ids, sizeof(int) * ndevice);
        
        //Create tensor for in_embed, out_embed, out_norm_w
        size_t shape_in_embed[2] = {meta->voc, meta->hs};
        size_t shape_out_embed[2] = {meta->voc, meta->hs};
        size_t shape_out_norm_w[1] = {meta->hs};
        qwen2Model->weights->in_embed = tensorCreate(shape_in_embed, 2, meta->dtype, device, device_ids[0]);
        qwen2Model->weights->out_embed = tensorCreate(shape_out_embed, 2, meta->dtype, device, device_ids[0]);
        qwen2Model->weights->out_norm_w = tensorCreate(shape_out_norm_w, 1, meta->dtype, device, device_ids[0]);

        //Create tensor for attn_norm_w, attn_q_w, attn_k_w, attn_v_w, attn_q_b, attn_k_b, attn_v_b, attn_o_w
        auto create_tensor_for_layer_2d = [&](llaisysTensor_t *&ptr, size_t dim0, size_t dim1){
            ptr = (llaisysTensor_t *)malloc(sizeof(llaisysTensor_t) * meta->nlayer);
            size_t shape[2] = {dim0, dim1};
            for(size_t i=0;i<meta->nlayer;i++){
                ptr[i] = tensorCreate(shape, 2, meta->dtype, device, device_ids[0]);
            }
        };

        auto create_tensor_for_layer_1d = [&](llaisysTensor_t *&ptr, size_t dim0){
            ptr = (llaisysTensor_t *)malloc(sizeof(llaisysTensor_t) * meta->nlayer);
            size_t shape[1] = {dim0};
            for(size_t i=0;i<meta->nlayer;i++){
                ptr[i] = tensorCreate(shape, 1, meta->dtype, device, device_ids[0]);
            }
        };

        create_tensor_for_layer_1d(qwen2Model->weights->attn_norm_w, meta->hs);
        create_tensor_for_layer_2d(qwen2Model->weights->attn_q_w, meta->nh * meta->dh, meta->hs);
        create_tensor_for_layer_2d(qwen2Model->weights->attn_k_w, meta->nkvh * meta->dh, meta->hs);
        create_tensor_for_layer_2d(qwen2Model->weights->attn_v_w, meta->nkvh * meta->dh, meta->hs);
        create_tensor_for_layer_1d(qwen2Model->weights->attn_q_b, meta->nh * meta->dh);
        create_tensor_for_layer_1d(qwen2Model->weights->attn_k_b, meta->nkvh * meta->dh);
        create_tensor_for_layer_1d(qwen2Model->weights->attn_v_b, meta->nkvh * meta->dh);
        create_tensor_for_layer_2d(qwen2Model->weights->attn_o_w, meta->nh * meta->dh, meta->hs);
        create_tensor_for_layer_1d(qwen2Model->weights->mlp_norm_w, meta->hs);
        create_tensor_for_layer_2d(qwen2Model->weights->mlp_gate_w, meta->di, meta->hs);
        create_tensor_for_layer_2d(qwen2Model->weights->mlp_up_w, meta->di, meta->hs);
        create_tensor_for_layer_2d(qwen2Model->weights->mlp_down_w, meta->hs, meta->di);

        return qwen2Model;
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model){
        if(model == nullptr){
            return;
        }
        tensorDestroy(model->weights->in_embed);
        tensorDestroy(model->weights->out_embed);
        tensorDestroy(model->weights->out_norm_w);

        auto destroy_tensor_for_layer = [&](llaisysTensor_t *ptr){
            if(ptr == nullptr) {
                return;
            }
            for(size_t i=0;i<model->meta->nlayer;i++){
                tensorDestroy(ptr[i]);
            }
            free(ptr);
        };

        if(model->weights == nullptr || model->meta == nullptr){
            if(model->meta) {
                free(model->meta);
            }
            if(model->device_ids) {
                free(model->device_ids);
            }
            if(model->weights){
                free(model->weights);
            }
            free(model);
            return;
        }
        destroy_tensor_for_layer(model->weights->attn_norm_w);
        destroy_tensor_for_layer(model->weights->attn_q_w);
        destroy_tensor_for_layer(model->weights->attn_k_w);
        destroy_tensor_for_layer(model->weights->attn_v_w);
        destroy_tensor_for_layer(model->weights->attn_q_b);
        destroy_tensor_for_layer(model->weights->attn_k_b);
        destroy_tensor_for_layer(model->weights->attn_v_b);
        destroy_tensor_for_layer(model->weights->attn_o_w);
        destroy_tensor_for_layer(model->weights->mlp_norm_w);
        destroy_tensor_for_layer(model->weights->mlp_gate_w);
        destroy_tensor_for_layer(model->weights->mlp_up_w);
        destroy_tensor_for_layer(model->weights->mlp_down_w);

        free(model->weights);
        if(model->meta) {
            free(model->meta);
        }
        if(model->device_ids) {
            free(model->device_ids);
        }
        free(model);
    }

    LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model){
        return model->weights;
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken, llaisysTensor_t *kcache, llaisysTensor_t *vcache, size_t past_len){
        if(!model || !token_ids || ntoken == 0) return -1;

        size_t seq_len = ntoken;
        size_t nlayer = model->meta->nlayer;
        size_t hs = model->meta->hs;
        size_t nh = model->meta->nh;
        size_t nkvh = model->meta->nkvh;
        size_t dh = model->meta->dh;
        size_t di = model->meta->di;
        size_t voc = model->meta->voc;
        float theta = (float)model->meta->theta;
        float epsilon = (float)model->meta->epsilon;
        float scale = (float)(1.0f / sqrt(dh));

        //Embedding layer
        //printf("[Infer] Creating input tensors...\n");
        size_t input_shape[1] = {seq_len};
        llaisysTensor_t input = tensorCreate(input_shape, 1, llaisysDataType_t::LLAISYS_DTYPE_I64, model->device, model->device_ids[0]);
        tensorLoad(input, token_ids);

        //printf("[Infer] Creating embedding tensors...\n");
        size_t embedding_shape[2] = {seq_len, hs};
        llaisysTensor_t embedding = tensorCreate(embedding_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
        //printf("[Infer] Embedding...\n");
        llaisysEmbedding(embedding, input, model->weights->in_embed);

        //Generate pos_ids
        //printf("[Infer] Embedding done.\n");
        size_t pos_id_shape[1] = {seq_len};
        llaisysTensor_t pos_ids = tensorCreate(pos_id_shape, 1, llaisysDataType_t::LLAISYS_DTYPE_I64, model->device, model->device_ids[0]);
        std::vector<int64_t> tmp_ids(seq_len);
        for(size_t i=0;i<seq_len;i++){
            tmp_ids[i] = ((int64_t)past_len+(int64_t)i);
        }
        tensorLoad(pos_ids, tmp_ids.data());

        //Hidden layer
        llaisysTensor_t layer_input = embedding;
        for(size_t i=0;i<nlayer;i++){
            //RMS norm
            size_t rms_output_shape[2] = {seq_len, hs};
            llaisysTensor_t rms_output = tensorCreate(rms_output_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysRmsNorm(rms_output, layer_input, model->weights->attn_norm_w[i], epsilon);
            
            //Linear Q, K, V
            size_t q_shape[2] = {seq_len, nh * dh};
            llaisysTensor_t q = tensorCreate(q_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysLinear(q, rms_output, model->weights->attn_q_w[i], model->weights->attn_q_b[i]);

            size_t k_shape[2] = {seq_len, nkvh * dh};
            llaisysTensor_t k = tensorCreate(k_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysLinear(k, rms_output, model->weights->attn_k_w[i], model->weights->attn_k_b[i]);

            size_t v_shape[2] = {seq_len, nkvh * dh};
            //llaisysTensor_t v = tensorCreate(v_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysTensor_t v = tensorView(tensorSlice(vcache[i], 0, past_len, past_len + seq_len),
                v_shape, 2);

            llaisysLinear(v, rms_output, model->weights->attn_v_w[i], model->weights->attn_v_b[i]);

            //RoPE
            size_t q_shape_3d[3] = {seq_len, nh, dh};
            q = tensorView(q, q_shape_3d, 3);
            size_t k_shape_3d[3] = {seq_len, nkvh, dh};
            k = tensorView(k, k_shape_3d, 3);
            //v = tensorView(v, k_shape_3d, 3);

            llaisysTensor_t q_rope = tensorCreate(q_shape_3d, 3, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysROPE(q_rope, q, pos_ids, theta);

            //llaisysTensor_t k_rope = tensorCreate(k_shape_3d, 3, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysTensor_t k_rope = tensorSlice(kcache[i], 0, past_len, past_len + seq_len);

            llaisysROPE(k_rope, k, pos_ids, theta);

            //Self Attention
            size_t attn_shape[3] = {seq_len, nh, dh};
            llaisysTensor_t attn_val = tensorCreate(attn_shape, 3, model->meta->dtype, model->device, model->device_ids[0]);
            k_rope = tensorSlice(kcache[i], 0, 0, past_len + seq_len);
            v = tensorSlice(vcache[i], 0, 0, past_len + seq_len);
            llaisysSelfAttention(attn_val, q_rope, k_rope, v, scale);
            size_t attn_shape_2d[2] = {seq_len, nh * dh};
            attn_val = tensorView(attn_val, attn_shape_2d, 2);
            //printf("[Infer] self attention done.\n");

            //Attention Output projection
            size_t attn_out_shape[2] = {seq_len, hs};
            llaisysTensor_t attn_out = tensorCreate(attn_out_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            //printf("[Infer] attn_out successfully create.\n");
            llaisysLinear(attn_out, attn_val, model->weights->attn_o_w[i], nullptr);
            //printf("[Infer] Attention output projection done.\n");

            //Residual connection
            size_t residual_shape[2] = {seq_len, hs};
            llaisysTensor_t residual_out = tensorCreate(residual_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysAdd(residual_out, layer_input, attn_out);

            //MLP
            //printf("[Infer] residual connection done.\n");
            size_t mlp_input_shape[2] = {seq_len, hs};
            llaisysTensor_t mlp_input = tensorCreate(mlp_input_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysRmsNorm(mlp_input, residual_out, model->weights->mlp_norm_w[i], epsilon);

            //SwiGLU
            size_t mlp_gate_shape[2] = {seq_len, di};
            llaisysTensor_t mlp_gate = tensorCreate(mlp_gate_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysLinear(mlp_gate, mlp_input, model->weights->mlp_gate_w[i], nullptr);

            size_t mlp_up_shape[2] = {seq_len, di};
            llaisysTensor_t mlp_up = tensorCreate(mlp_up_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysLinear(mlp_up, mlp_input, model->weights->mlp_up_w[i], nullptr);

            llaisysTensor_t swiGlu_out = tensorCreate(mlp_up_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysSwiGLU(swiGlu_out, mlp_gate, mlp_up);

            size_t mlp_down_shape[2] = {seq_len, hs};
            llaisysTensor_t mlp_down = tensorCreate(mlp_down_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysLinear(mlp_down, swiGlu_out, model->weights->mlp_down_w[i], nullptr);

            llaisysAdd(layer_input, residual_out, mlp_down);

            tensorDestroy(q_rope);
            // tensorDestroy(k_rope);
            tensorDestroy(q);
            tensorDestroy(k);
            // tensorDestroy(v);
            tensorDestroy(attn_val);
            tensorDestroy(attn_out);
            tensorDestroy(residual_out);
            tensorDestroy(mlp_input);
            tensorDestroy(mlp_gate);
            tensorDestroy(mlp_up);
            tensorDestroy(swiGlu_out);
            tensorDestroy(mlp_down);
            tensorDestroy(rms_output);
        }
        //Layer norm
        size_t output_final_shape[2] = {seq_len, hs};
        llaisysTensor_t output_final = tensorCreate(output_final_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
        llaisysRmsNorm(output_final, layer_input, model->weights->out_norm_w, epsilon);

        //Out embedding
        size_t output_shape[2] = {seq_len, voc};
        llaisysTensor_t output = tensorCreate(output_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
        llaisysLinear(output, output_final, model->weights->out_embed, nullptr);
        
        //Slice
        llaisysTensor_t output_slice = tensorSlice(output, 0, seq_len-1, seq_len);

        //Argmax
        size_t index_shape[1] = {1};
        llaisysTensor_t index = tensorCreate(index_shape, 1, llaisysDataType_t::LLAISYS_DTYPE_I64, model->device, model->device_ids[0]);
        llaisysTensor_t value = tensorCreate(index_shape, 1, model->meta->dtype, model->device, model->device_ids[0]);
        llaisysArgmax(index, value, output_slice);

        int64_t index_result = *((int64_t *)tensorGetData(index));

        tensorDestroy(index);
        tensorDestroy(value);
        tensorDestroy(output);
        tensorDestroy(output_final);
        tensorDestroy(layer_input);
        tensorDestroy(input);
        tensorDestroy(pos_ids);

        return index_result;
    }
}