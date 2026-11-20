#include "multimodal_engine.h"
#include <cmath>
#include <cstring>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <android/log.h>

#define LOG_TAG "MultimodalEngineNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace ops;

namespace {

// Deterministic Pseudo-Vision Feature Generator per unique Image Path
// Guarantees that the SAME image always produces the EXACT same visual embedding tensor across all steps/epochs.
TensorPtr get_deterministic_image_features(const std::string& image_path, int64_t num_patches, int64_t vision_dim) {
    auto features = zeros({1, num_patches, vision_dim}, kFloat32, kCPU);
    float* data = features->data<float>();

    // 64-bit FNV-1a Hash of image path string
    uint64_t hash = 14695981039346656037ULL;
    for (char c : image_path) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ULL;
    }

    // Deterministic pseudo-random sequence seeded by image hash
    uint64_t state = hash;
    for (int64_t i = 0; i < num_patches * vision_dim; ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        float val = static_cast<float>((state >> 33) & 0x7FFFFFFF) / static_cast<float>(0x7FFFFFFF);
        data[i] = (val - 0.5f) * 0.2f; // Scaled in [-0.1, 0.1]
    }

    features->set_requires_grad(false);
    return features;
}

} // namespace

MultimodalTrainerState* init_multimodal_trainer(const char* model_path) {
    LOGI("init_multimodal_trainer: initializing QwenMultimodalModel with path '%s'", model_path ? model_path : "default");
    auto* state = new MultimodalTrainerState();
    state->model_path = model_path ? model_path : "qwen2.5-0.5b-multimodal";

    state->config.text_config.vocab_size = 256;
    state->config.text_config.hidden_size = 64;
    state->config.text_config.intermediate_size = 128;
    state->config.text_config.num_hidden_layers = 2;
    state->config.text_config.num_attention_heads = 4;
    state->config.text_config.num_key_value_heads = 2;
    state->config.text_config.max_position_embeddings = 512;
    state->config.text_config.rms_norm_eps = 1e-6f;
    state->config.vision_dim = 32;
    state->config.use_projector_bias = false;
    state->config.seed = 42;

    state->vision_dim = state->config.vision_dim;
    state->num_patches = 4;
    state->max_seq_len = 16;

    state->model = std::make_shared<QwenMultimodalModel>(state->config);
    state->tokenizer = std::make_shared<SimpleTokenizer>(state->config.text_config.vocab_size);

    // Initialize base weights deterministically
    for (auto& p : state->model->base_model()->parameters()) {
        if (!p) continue;
        float* d = p->data<float>();
        for (int64_t i = 0; i < p->numel(); ++i) {
            d[i] = ((i % 17) - 8.0f) * 0.02f;
        }
    }
    state->model->base_model()->assign_weight("final_norm.weight", full({state->config.text_config.hidden_size}, 1.0f));
    for (int l = 0; l < state->config.text_config.num_hidden_layers; ++l) {
        state->model->base_model()->assign_weight("layers." + std::to_string(l) + ".input_norm.weight", full({state->config.text_config.hidden_size}, 1.0f));
        state->model->base_model()->assign_weight("layers." + std::to_string(l) + ".post_norm.weight", full({state->config.text_config.hidden_size}, 1.0f));
    }
    state->model->freeze_base();

    MultimodalTrainerConfig trainer_cfg;
    trainer_cfg.learning_rate = 1e-5f; // Standard research LR (1e-5)
    trainer_cfg.max_grad_norm = 1.0f;
    trainer_cfg.weight_decay = 0.0f;
    state->trainer = std::make_shared<MultimodalTrainer>(*state->model, trainer_cfg);

    state->is_initialized = true;
    LOGI("init_multimodal_trainer: QwenMultimodalModel initialized successfully (hidden=%d, vocab=%d, vision_dim=%d)", 
         state->config.text_config.hidden_size, state->config.text_config.vocab_size, state->config.vision_dim);
    return state;
}

void free_multimodal_trainer(MultimodalTrainerState* state) {
    if (!state) return;
    LOGI("free_multimodal_trainer: destroying state");
    delete state;
}

std::string get_multimodal_status_json(MultimodalTrainerState* state) {
    if (!state || !state->is_initialized) {
        return "{\"status\":\"uninitialized\",\"ready\":false}";
    }

    std::ostringstream ss;
    ss << "{"
       << "\"status\":\"ready\","
       << "\"model\":\"" << state->model_path << "\","
       << "\"hidden_size\":" << state->config.text_config.hidden_size << ","
       << "\"vocab_size\":" << state->config.text_config.vocab_size << ","
       << "\"vision_dim\":" << state->config.vision_dim << ","
       << "\"projector_trainable\":true,"
       << "\"base_model_frozen\":true,"
       << "\"steps_completed\":" << (state->trainer ? state->trainer->step_count() : 0)
       << "}";
    return ss.str();
}

MultimodalStepResult execute_multimodal_training_step(
    MultimodalTrainerState* state,
    const char* image_path,
    const char* prompt,
    const char* response_label,
    float learning_rate
) {
    MultimodalStepResult res{0.0f, 0.0f, 0.0f, false};

    if (!state || !state->is_initialized || !state->model || !state->trainer) {
        LOGE("execute_multimodal_training_step: state not initialized!");
        return res;
    }

    try {
        const int64_t vision_dim = state->vision_dim;
        const int64_t num_patches = state->num_patches;

        std::string img_key = (image_path && std::strlen(image_path) > 0) ? image_path : "default_image";
        auto raw_image_patches = get_deterministic_image_features(img_key, num_patches, vision_dim);

        std::string prompt_str = prompt ? prompt : "";
        std::string resp_str = response_label ? response_label : "";
        
        std::vector<int> prompt_tokens = state->tokenizer->encode(prompt_str);
        std::vector<int> resp_tokens = state->tokenizer->encode(resp_str);

        if (prompt_tokens.empty()) prompt_tokens.push_back(1);
        if (resp_tokens.empty()) resp_tokens.push_back(2);

        int64_t max_text_tokens = state->max_seq_len - num_patches;
        if (max_text_tokens < 2) max_text_tokens = 2;

        std::vector<int32_t> text_input_ids;
        for (int tid : prompt_tokens) {
            text_input_ids.push_back(static_cast<int32_t>(tid % state->config.text_config.vocab_size));
        }
        for (int tid : resp_tokens) {
            text_input_ids.push_back(static_cast<int32_t>(tid % state->config.text_config.vocab_size));
        }
        if (static_cast<int64_t>(text_input_ids.size()) > max_text_tokens) {
            text_input_ids.resize(max_text_tokens);
        }

        const int64_t num_text_tokens = static_cast<int64_t>(text_input_ids.size());
        auto input_ids_tensor = zeros({1, num_text_tokens}, kInt32, kCPU);
        for (int64_t s = 0; s < num_text_tokens; ++s) {
            input_ids_tensor->data<int32_t>()[s] = text_input_ids[s];
        }

        MultimodalBatch batch;
        batch.vision_features = raw_image_patches;
        batch.input_ids = input_ids_tensor;
        batch.attention_mask = full({1, num_text_tokens}, 1.0f, kFloat32, kCPU);
        batch.labels = nullptr;

        if (learning_rate > 0.0f) {
            state->trainer->optimizer()->set_learning_rate(learning_rate);
        }

        float loss = state->trainer->train_step(batch);
        res.loss = loss;

        // Compute projector gradient norm
        float grad_norm = 0.0f;
        const auto& proj_w = state->model->projector_weight();
        if (proj_w && proj_w->grad()) {
            const float* g = proj_w->grad()->data<float>();
            float sum_sq = 0.0f;
            for (int64_t i = 0; i < proj_w->numel(); ++i) {
                sum_sq += g[i] * g[i];
            }
            grad_norm = std::sqrt(sum_sq);
        }
        res.gradient_norm = grad_norm;
        res.accuracy = 0.0f; // Multi-class causal LM
        res.success = true;

        double rss_mb = 0.0;
        {
            std::ifstream status_file("/proc/self/status");
            std::string line;
            while (std::getline(status_file, line)) {
                if (line.rfind("VmRSS:", 0) == 0) {
                    std::stringstream ss(line.substr(6));
                    double kb = 0.0;
                    ss >> kb;
                    rss_mb = kb / 1024.0;
                    break;
                }
            }
        }

        LOGI("[DEVICE_STEP] step=%zu loss=%.4f grad_norm=%.4f lr=%.5f rss=%.1fMB",
             state->trainer->step_count(), res.loss, res.gradient_norm,
             state->trainer->optimizer()->get_learning_rate(), rss_mb);

    } catch (const std::exception& e) {
        LOGE("execute_multimodal_training_step exception: %s", e.what());
        res.success = false;
    } catch (...) {
        LOGE("execute_multimodal_training_step: unknown exception");
        res.success = false;
    }

    return res;
}
