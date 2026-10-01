#pragma once

#include <cstdint>
#include <cstddef>
#include <array>
#ifdef EFRARL_ENABLE_FILESYSTEM
#include <fstream>
#endif

namespace EfRaRL {

// ===========================================================================
// 1. INTERNAL MATH AND RANDOM NUMBER GENERATOR (FastRNG)
// ===========================================================================
class FastRNG {
    uint32_t state = 123456789;
public:
    void setSeed(uint32_t seed) { state = seed == 0 ? 1 : seed; }
    uint32_t next() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }
    float nextFloat() { return (next() % 2000000 / 1000000.0f) - 1.0f; }
    float nextGaussian() {
        float sum = 0.0f;
        for (int i = 0; i < 12; ++i) {
            sum += (next() & 0xFFFF) / 65535.0f;
        }
        return sum - 6.0f;
    }
};

// ===========================================================================
// 2. HARDWARE-AGNOSTIC FAST MATH (No <cmath> dependency)
// ===========================================================================
inline bool is_nan(float x) {
    union { float f; uint32_t i; } u = {x};
    return (u.i & 0x7FFFFFFF) > 0x7F800000;
}

inline bool is_inf(float x) {
    union { float f; uint32_t i; } u = {x};
    return (u.i & 0x7FFFFFFF) == 0x7F800000;
}

inline float fast_inv_sqrt(float number) {
    union { float f; uint32_t i; } conv = {number};
    conv.i  = 0x5f3759df - ( conv.i >> 1 );
    float x2 = number * 0.5F;
    float y = conv.f;
    return y * ( 1.5F - ( x2 * y * y ) );
}

inline float fast_tanh(float x) {
    if (x > 3.0f) return 1.0f;
    if (x < -3.0f) return -1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

inline float relu(float x) { return x > 0.0f ? x : 0.0f; }
inline float relu_grad(float x) { return x > 0.0f ? 1.0f : 0.0f; }
inline float tanh_grad(float x) { float t = fast_tanh(x); return 1.0f - t * t; }
inline float clamp(float x, float min_v, float max_v) { return x < min_v ? min_v : (x > max_v ? max_v : x); }

// ===========================================================================
// 2. CORE STRUCTURES
// ===========================================================================
template <size_t NumSensors, size_t NumActions>
struct Transition {
    std::array<float, NumSensors> state;
    std::array<float, NumActions> action;
    std::array<float, NumSensors> next_state;
    float reward;
    float done;
};

// ===========================================================================
// 3. NEURAL NETWORK LAYERS (Weights, Adam, Double Buffering)
// ===========================================================================
template <size_t In, size_t Out>
struct DenseLayer {
    std::array<std::array<float, Out>, In> w{};
    std::array<float, Out> b{};
    
    std::array<std::array<float, Out>, In> m_w{};
    std::array<std::array<float, Out>, In> v_w{};
    std::array<float, Out> m_b{};
    std::array<float, Out> v_b{};
    
    std::array<std::array<float, Out>, In> grad_w{};
    std::array<float, Out> grad_b{};

    void init_he(FastRNG& rng) {
        float stddev = 1.41421356f * fast_inv_sqrt(static_cast<float>(In));
        for (size_t i = 0; i < In; ++i) {
            for (size_t j = 0; j < Out; ++j) {
                w[i][j] = rng.nextGaussian() * stddev;
                m_w[i][j] = 0.0f; v_w[i][j] = 0.0f;
                grad_w[i][j] = 0.0f;
            }
        }
        for (size_t j = 0; j < Out; ++j) {
            b[j] = 0.0f; m_b[j] = 0.0f; v_b[j] = 0.0f;
            grad_b[j] = 0.0f;
        }
    }

    void reset_gradients() {
        for(size_t i=0; i<In; ++i) for(size_t j=0; j<Out; ++j) grad_w[i][j] = 0.0f;
        for(size_t j=0; j<Out; ++j) grad_b[j] = 0.0f;
    }

    void copy_from(const DenseLayer<In, Out>& other) {
        *this = other;
    }

    size_t serialize(float* buffer) const {
        size_t idx = 0;
        for (size_t i = 0; i < In; ++i) {
            for (size_t j = 0; j < Out; ++j) buffer[idx++] = w[i][j];
        }
        for (size_t j = 0; j < Out; ++j) buffer[idx++] = b[j];
        return idx;
    }

    size_t deserialize(const float* buffer) {
        size_t idx = 0;
        for (size_t i = 0; i < In; ++i) {
            for (size_t j = 0; j < Out; ++j) w[i][j] = buffer[idx++];
        }
        for (size_t j = 0; j < Out; ++j) b[j] = buffer[idx++];
        return idx;
    }

    void soft_update(const DenseLayer<In, Out>& online, float tau) {
        for (size_t i = 0; i < In; ++i) {
            for (size_t j = 0; j < Out; ++j) {
                w[i][j] = tau * online.w[i][j] + (1.0f - tau) * w[i][j];
            }
        }
        for (size_t j = 0; j < Out; ++j) {
            b[j] = tau * online.b[j] + (1.0f - tau) * b[j];
        }
    }

    void apply_adam(float lr, float beta1, float beta2, float epsilon, float max_grad_norm, size_t batch_size) {
        for (size_t i = 0; i < In; ++i) {
            for (size_t j = 0; j < Out; ++j) {
                float g = grad_w[i][j] / batch_size;
                g = clamp(g, -max_grad_norm, max_grad_norm);
                m_w[i][j] = beta1 * m_w[i][j] + (1.0f - beta1) * g;
                v_w[i][j] = beta2 * v_w[i][j] + (1.0f - beta2) * g * g;
                float m_hat = m_w[i][j] / (1.0f - beta1);
                float v_hat = v_w[i][j] / (1.0f - beta2);
                w[i][j] -= lr * m_hat * fast_inv_sqrt(v_hat + epsilon * epsilon);
            }
        }
        for (size_t j = 0; j < Out; ++j) {
            float g = grad_b[j] / batch_size;
            g = clamp(g, -max_grad_norm, max_grad_norm);
            m_b[j] = beta1 * m_b[j] + (1.0f - beta1) * g;
            v_b[j] = beta2 * v_b[j] + (1.0f - beta2) * g * g;
            float m_hat = m_b[j] / (1.0f - beta1);
            float v_hat = v_b[j] / (1.0f - beta2);
            b[j] -= lr * m_hat * fast_inv_sqrt(v_hat + epsilon * epsilon);
        }
    }
};

template <size_t S, size_t H, size_t A>
struct ActorNetwork {
    DenseLayer<S, H> l1;
    DenseLayer<H, A> l2;
    void init(FastRNG& rng) { l1.init_he(rng); l2.init_he(rng); }
    void copy_from(const ActorNetwork& o) { l1.copy_from(o.l1); l2.copy_from(o.l2); }
    size_t serialize(float* buffer) const { return l1.serialize(buffer) + l2.serialize(buffer + (S * H + H)); }
    size_t deserialize(const float* buffer) { return l1.deserialize(buffer) + l2.deserialize(buffer + (S * H + H)); }
    void soft_update(const ActorNetwork& o, float tau) { l1.soft_update(o.l1, tau); l2.soft_update(o.l2, tau); }
    void reset_gradients() { l1.reset_gradients(); l2.reset_gradients(); }
    void apply_adam(float lr, float b1, float b2, float eps, float clip, size_t batch) {
        l1.apply_adam(lr, b1, b2, eps, clip, batch);
        l2.apply_adam(lr, b1, b2, eps, clip, batch);
    }
};

template <size_t S, size_t H, size_t A>
struct CriticNetwork {
    DenseLayer<S + A, H> l1;
    DenseLayer<H, 1> l2;
    void init(FastRNG& rng) { l1.init_he(rng); l2.init_he(rng); }
    void copy_from(const CriticNetwork& o) { l1.copy_from(o.l1); l2.copy_from(o.l2); }
    size_t serialize(float* buffer) const { return l1.serialize(buffer) + l2.serialize(buffer + ((S + A) * H + H)); }
    size_t deserialize(const float* buffer) { return l1.deserialize(buffer) + l2.deserialize(buffer + ((S + A) * H + H)); }
    void soft_update(const CriticNetwork& o, float tau) { l1.soft_update(o.l1, tau); l2.soft_update(o.l2, tau); }
    void reset_gradients() { l1.reset_gradients(); l2.reset_gradients(); }
    void apply_adam(float lr, float b1, float b2, float eps, float clip, size_t batch) {
        l1.apply_adam(lr, b1, b2, eps, clip, batch);
        l2.apply_adam(lr, b1, b2, eps, clip, batch);
    }
};

// ===========================================================================
// 4. MICRO RL - DDPG CONTINUOUS AGENT
// ===========================================================================
template <size_t NumSensors, size_t NumHiddenNeurons, size_t NumActions, size_t BufferCapacity = 500, size_t BatchSize = 32>
class ContinuousAgent {
private:
    std::array<Transition<NumSensors, NumActions>, BufferCapacity> replay_buffer{};
    size_t buffer_index = 0;
    size_t current_buffer_size = 0;

    std::array<ActorNetwork<NumSensors, NumHiddenNeurons, NumActions>, 2> actors;
    volatile uint8_t active_actor_index = 0;
    
    ActorNetwork<NumSensors, NumHiddenNeurons, NumActions> target_actor;
    
    CriticNetwork<NumSensors, NumHiddenNeurons, NumActions> critic;
    CriticNetwork<NumSensors, NumHiddenNeurons, NumActions> target_critic;

public:
    FastRNG rng;
    bool is_safe_mode = false; 
    bool is_training = true;

    std::array<float, NumActions> ou_noise_state{};
    float ou_theta = 0.15f;
    float ou_sigma = 0.2f;

    float exploration_noise = 1.0f; 
    float gamma = 0.99f;            
    float tau = 0.005f;             
    float max_grad_norm = 1.0f;     
    float reward_scale = 1.0f;
    float lr_actor = 0.001f;
    float lr_critic = 0.005f;

    constexpr ContinuousAgent() = default;

    void begin(uint32_t seed) {
        rng.setSeed(seed);
        actors[0].init(rng);
        actors[1].copy_from(actors[0]);
        target_actor.copy_from(actors[0]);
        
        critic.init(rng);
        target_critic.copy_from(critic);

        for(size_t i=0; i<NumActions; ++i) ou_noise_state[i] = 0.0f;
    }

    // ===========================================================================
    // API 1: INFERENCE AND CONTROL (SIM-TO-REAL TRANSFER)
    // ===========================================================================
    void start_training() { is_training = true; }
    void stop_training() { is_training = false; }

    static constexpr size_t get_model_size() {
        return (NumSensors * NumHiddenNeurons + NumHiddenNeurons) + 
               (NumHiddenNeurons * NumActions + NumActions) + 
               ((NumSensors + NumActions) * NumHiddenNeurons + NumHiddenNeurons) + 
               (NumHiddenNeurons * 1 + 1);
    }

    size_t serialize_model(float* buffer) const {
        size_t idx = 0;
        idx += actors[active_actor_index].serialize(buffer + idx);
        idx += critic.serialize(buffer + idx);
        return idx;
    }

    size_t deserialize_model(const float* buffer) {
        size_t idx = 0;
        idx += actors[active_actor_index].deserialize(buffer + idx);
        actors[1 - active_actor_index].copy_from(actors[active_actor_index]);
        target_actor.copy_from(actors[active_actor_index]);
        
        idx += critic.deserialize(buffer + idx);
        target_critic.copy_from(critic);
        return idx;
    }

#ifdef EFRARL_ENABLE_FILESYSTEM
    bool export_model(const char* filepath) {
        std::ofstream out(filepath, std::ios::binary);
        if (!out.is_open()) return false;
        out.write(reinterpret_cast<const char*>(&actors[active_actor_index]), sizeof(ActorNetwork<NumSensors, NumHiddenNeurons, NumActions>));
        out.write(reinterpret_cast<const char*>(&critic), sizeof(CriticNetwork<NumSensors, NumHiddenNeurons, NumActions>));
        out.close();
        return true;
    }

    bool import_model(const char* filepath) {
        std::ifstream in(filepath, std::ios::binary);
        if (!in.is_open()) return false;
        in.read(reinterpret_cast<char*>(&actors[active_actor_index]), sizeof(ActorNetwork<NumSensors, NumHiddenNeurons, NumActions>));
        in.read(reinterpret_cast<char*>(&critic), sizeof(CriticNetwork<NumSensors, NumHiddenNeurons, NumActions>));
        in.close();
        
        actors[1 - active_actor_index] = actors[active_actor_index];
        target_actor = actors[active_actor_index];
        target_critic = critic;
        return true;
    }
#endif

    std::array<float, NumActions> use_model(const std::array<float, NumSensors>& state) {
        if (is_safe_mode) return safe_action();
        return forward_actor(state, active_actor_index, nullptr, nullptr);
    }

    std::array<float, NumActions> get_action(const std::array<float, NumSensors>& state) {
        if (is_safe_mode) return safe_action();

        std::array<float, NumActions> action = forward_actor(state, active_actor_index, nullptr, nullptr);

        for(size_t i = 0; i < NumActions; ++i) {
            float noise = ou_theta * (0.0f - ou_noise_state[i]) + ou_sigma * rng.nextGaussian();
            ou_noise_state[i] += noise;
            
            action[i] += ou_noise_state[i] * exploration_noise;
            action[i] = clamp(action[i], -1.0f, 1.0f);
            
            if (is_nan(action[i]) || is_inf(action[i])) {
                is_safe_mode = true;
                return safe_action();
            }
        }
        return action;
    }

    // ===========================================================================
    // API 2: LEARNING (HEAVY MATH - BACKGROUND TASK)
    // ===========================================================================
    void train_step(const std::array<float, NumSensors>& state, 
                    const std::array<float, NumActions>& action, 
                    float reward, 
                    const std::array<float, NumSensors>& next_state, 
                    float done) {
                        
        if (is_safe_mode || !is_training) return;

        float scaled_reward = clamp(reward * reward_scale, -1.0f, 1.0f);
        save_to_buffer(state, action, scaled_reward, next_state, done);
        
        if (current_buffer_size < BatchSize) return;

        uint8_t train_actor_idx = 1 - active_actor_index;
        
        actors[train_actor_idx].reset_gradients();
        critic.reset_gradients();

        for (size_t b = 0; b < BatchSize; ++b) {
            size_t rnd_idx = rng.next() % current_buffer_size;
            const auto& t = replay_buffer[rnd_idx];

            // -----------------------------------------------------
            // 1. CRITIC UPDATE
            // -----------------------------------------------------
            std::array<float, NumActions> target_act = forward_target_actor(t.next_state);
            float target_q = forward_target_critic(t.next_state, target_act);
            float y = t.reward + gamma * (1.0f - t.done) * target_q;

            std::array<float, NumHiddenNeurons> c_h_pre{}, c_h_act{};
            float q_pred = forward_critic(t.state, t.action, &c_h_pre, &c_h_act);

            float critic_loss_grad = q_pred - y;
            critic_loss_grad = clamp(critic_loss_grad, -max_grad_norm, max_grad_norm);

            std::array<float, NumHiddenNeurons> d_c_h{};
            for(size_t i=0; i<NumHiddenNeurons; ++i) {
                critic.l2.grad_w[i][0] += c_h_act[i] * critic_loss_grad;
                d_c_h[i] = critic_loss_grad * critic.l2.w[i][0] * relu_grad(c_h_pre[i]);
            }
            critic.l2.grad_b[0] += critic_loss_grad;

            for(size_t i=0; i<NumSensors; ++i) {
                for(size_t j=0; j<NumHiddenNeurons; ++j) {
                    critic.l1.grad_w[i][j] += t.state[i] * d_c_h[j];
                }
            }
            for(size_t i=0; i<NumActions; ++i) {
                for(size_t j=0; j<NumHiddenNeurons; ++j) {
                    critic.l1.grad_w[NumSensors + i][j] += t.action[i] * d_c_h[j];
                }
            }
            for(size_t j=0; j<NumHiddenNeurons; ++j) {
                critic.l1.grad_b[j] += d_c_h[j];
            }

            // -----------------------------------------------------
            // 2. ACTOR UPDATE
            // -----------------------------------------------------
            std::array<float, NumHiddenNeurons> a_h_pre{}, a_h_act{};
            std::array<float, NumActions> a_out_pre{};
            std::array<float, NumActions> pred_action = forward_actor(t.state, train_actor_idx, &a_h_pre, &a_h_act, &a_out_pre);

            std::array<float, NumHiddenNeurons> c_h_pre_actor{}, c_h_act_actor{};
            forward_critic(t.state, pred_action, &c_h_pre_actor, &c_h_act_actor);

            std::array<float, NumActions> d_action{};
            for(size_t i=0; i<NumActions; ++i) {
                float sum = 0.0f;
                for(size_t j=0; j<NumHiddenNeurons; ++j) {
                    float d_h = -1.0f * critic.l2.w[j][0] * relu_grad(c_h_pre_actor[j]); 
                    sum += critic.l1.w[NumSensors + i][j] * d_h;
                }
                d_action[i] = sum;
            }

            std::array<float, NumHiddenNeurons> d_a_h{};
            for(size_t i=0; i<NumHiddenNeurons; ++i) d_a_h[i] = 0.0f;

            for(size_t j=0; j<NumActions; ++j) {
                float d_out = d_action[j] * tanh_grad(a_out_pre[j]);
                actors[train_actor_idx].l2.grad_b[j] += d_out;
                for(size_t i=0; i<NumHiddenNeurons; ++i) {
                    actors[train_actor_idx].l2.grad_w[i][j] += a_h_act[i] * d_out;
                    d_a_h[i] += d_out * actors[train_actor_idx].l2.w[i][j];
                }
            }
            
            for(size_t j=0; j<NumHiddenNeurons; ++j) {
                float d_h = d_a_h[j] * relu_grad(a_h_pre[j]);
                actors[train_actor_idx].l1.grad_b[j] += d_h;
                for(size_t i=0; i<NumSensors; ++i) {
                    actors[train_actor_idx].l1.grad_w[i][j] += t.state[i] * d_h;
                }
            }
        }

        actors[train_actor_idx].apply_adam(lr_actor, 0.9f, 0.999f, 1e-8f, max_grad_norm, BatchSize);
        critic.apply_adam(lr_critic, 0.9f, 0.999f, 1e-8f, max_grad_norm, BatchSize);

        active_actor_index = train_actor_idx;
        
        actors[1 - active_actor_index] = actors[active_actor_index];
        
        target_actor.soft_update(actors[active_actor_index], tau);
        target_critic.soft_update(critic, tau);
    }

private:
    std::array<float, NumActions> safe_action() const { return std::array<float, NumActions>{}; }

    void save_to_buffer(const std::array<float, NumSensors>& s, const std::array<float, NumActions>& a, float r, const std::array<float, NumSensors>& s_next, float d) {
        for(size_t i = 0; i < NumSensors; ++i) {
            if (is_nan(s[i]) || is_nan(s_next[i])) return;
        }
        for(size_t i = 0; i < NumActions; ++i) {
            if (is_nan(a[i])) return;
        }
        if (is_nan(r)) return;

        replay_buffer[buffer_index] = {s, a, s_next, r, d};
        buffer_index = (buffer_index + 1) % BufferCapacity;
        if (current_buffer_size < BufferCapacity) current_buffer_size++;
    }

    std::array<float, NumActions> forward_actor(const std::array<float, NumSensors>& state, uint8_t idx, 
                                                std::array<float, NumHiddenNeurons>* h_pre = nullptr,
                                                std::array<float, NumHiddenNeurons>* h_act = nullptr,
                                                std::array<float, NumActions>* out_pre = nullptr) {
        std::array<float, NumHiddenNeurons> h{};
        for(size_t j=0; j<NumHiddenNeurons; ++j) {
            h[j] = actors[idx].l1.b[j];
            for(size_t i=0; i<NumSensors; ++i) h[j] += state[i] * actors[idx].l1.w[i][j];
            if (h_pre) (*h_pre)[j] = h[j];
            h[j] = relu(h[j]);
            if (h_act) (*h_act)[j] = h[j];
        }
        
        std::array<float, NumActions> out{};
        for(size_t j=0; j<NumActions; ++j) {
            out[j] = actors[idx].l2.b[j];
            for(size_t i=0; i<NumHiddenNeurons; ++i) out[j] += h[i] * actors[idx].l2.w[i][j];
            if (out_pre) (*out_pre)[j] = out[j];
            out[j] = fast_tanh(out[j]);
        }
        return out;
    }

    std::array<float, NumActions> forward_target_actor(const std::array<float, NumSensors>& state) {
        std::array<float, NumHiddenNeurons> h{};
        for(size_t j=0; j<NumHiddenNeurons; ++j) {
            h[j] = target_actor.l1.b[j];
            for(size_t i=0; i<NumSensors; ++i) h[j] += state[i] * target_actor.l1.w[i][j];
            h[j] = relu(h[j]);
        }
        std::array<float, NumActions> out{};
        for(size_t j=0; j<NumActions; ++j) {
            out[j] = target_actor.l2.b[j];
            for(size_t i=0; i<NumHiddenNeurons; ++i) out[j] += h[i] * target_actor.l2.w[i][j];
            out[j] = fast_tanh(out[j]);
        }
        return out;
    }

    float forward_critic(const std::array<float, NumSensors>& state, const std::array<float, NumActions>& action,
                         std::array<float, NumHiddenNeurons>* h_pre = nullptr,
                         std::array<float, NumHiddenNeurons>* h_act = nullptr) {
        std::array<float, NumHiddenNeurons> h{};
        for(size_t j=0; j<NumHiddenNeurons; ++j) {
            h[j] = critic.l1.b[j];
            for(size_t i=0; i<NumSensors; ++i) h[j] += state[i] * critic.l1.w[i][j];
            for(size_t i=0; i<NumActions; ++i) h[j] += action[i] * critic.l1.w[NumSensors + i][j];
            if (h_pre) (*h_pre)[j] = h[j];
            h[j] = relu(h[j]);
            if (h_act) (*h_act)[j] = h[j];
        }
        
        float out = critic.l2.b[0];
        for(size_t i=0; i<NumHiddenNeurons; ++i) out += h[i] * critic.l2.w[i][0];
        return out;
    }

    float forward_target_critic(const std::array<float, NumSensors>& state, const std::array<float, NumActions>& action) {
        std::array<float, NumHiddenNeurons> h{};
        for(size_t j=0; j<NumHiddenNeurons; ++j) {
            h[j] = target_critic.l1.b[j];
            for(size_t i=0; i<NumSensors; ++i) h[j] += state[i] * target_critic.l1.w[i][j];
            for(size_t i=0; i<NumActions; ++i) h[j] += action[i] * target_critic.l1.w[NumSensors + i][j];
            h[j] = relu(h[j]);
        }
        float out = target_critic.l2.b[0];
        for(size_t i=0; i<NumHiddenNeurons; ++i) out += h[i] * target_critic.l2.w[i][0];
        return out;
    }
};

}
