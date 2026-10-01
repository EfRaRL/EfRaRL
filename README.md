# EfRaRL: Embedded Failsafe Reinforcement Learning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23090783.svg)](https://doi.org/10.5281/zenodo.23090783)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![RAM](https://img.shields.io/badge/RAM-16.0%20KB-brightgreen.svg)]()

**EfRaRL** is a lightweight, zero-allocation, header-only C++17 Deep Reinforcement Learning (DRL) library explicitly designed for resource-constrained microcontrollers (e.g., STM32, ESP32). It implements a continuous control DDPG (Deep Deterministic Policy Gradient) agent capable of on-device training and real-time inference within a **strictly bounded 16.0 KB static RAM footprint**.

## Paper & Citation
If you use EfRaRL in your research or project, please cite our paper:

> **EfRaRL: Embedded Failsafe Robust Actor Reinforcement Learning for Resource-Constrained Microcontrollers**  
> *Efe Ramazan Örnek* (Independent Researcher)  
> 📖 **Official Paper on Zenodo:** [https://zenodo.org/records/23090783](https://zenodo.org/records/23090783)  
> 🔗 **DOI:** [10.5281/zenodo.23090783](https://doi.org/10.5281/zenodo.23090783)

```bibtex
@article{ornek2026efrarl,
  title={EfRaRL: Embedded Failsafe Robust Actor Reinforcement Learning for Resource-Constrained Microcontrollers},
  author={{\"{O}}rnek, Efe Ramazan},
  journal={Zenodo Preprint},
  doi={10.5281/zenodo.23090783},
  year={2026}
}
```

---

## Core Features
- **Zero Dynamic Allocation (0 Bytes Heap):** All neural networks, weight matrices, and replay buffers are resolved at compile-time via C++17 non-type templates. Operates entirely in the `.bss` and `.data` sections to prevent MCU heap fragmentation.
- **Interrupt-Safe Double Buffering:** The architecture isolates the active policy from the backpropagation process. High-frequency ISRs (Interrupt Service Routines) can query the Active Network with zero lock-contention while the MCU trains the Target Network in the background.
- **Failsafe Design:** Built-in gradient clipping and Huber loss calculations mathematically prevent `NaN` divergence and gradient explosion, even under extreme sensor dropout (10%) and hardware noise.
- **Sim-to-Real Transfer (.EfRa format):** Train on a powerful desktop, export the raw active weights (only ~3.3 KB) as a `.EfRa` file, and flash it directly to the MCU for instant real-world inference.

---

## 🚀 Quick Start & Usage

EfRaRL is header-only. Simply include `EfRaRL.hpp` in your C++ project. No external dependencies (not even STL vectors or dynamic arrays) are required.

### 1. Initializing the Agent
You define the architecture of the neural network using C++ template parameters. This guarantees that all memory is statically allocated at compile-time.

```cpp
#include "EfRaRL.hpp"

// Template: <Sensors, HiddenNeurons, Actions, BufferCapacity, BatchSize>
// Example: 3 Sensors, 16 Hidden Neurons, 2 Actions, 200 Replay Buffer, 16 Batch
EfRaRL::ContinuousAgent<3, 16, 2, 200, 16> agent;

int main() {
    // Initialize weights with a random seed (FastRNG)
    agent.begin(12345);
    
    // Optional: Tune hyperparameters
    agent.lr_actor = 0.002f;
    agent.lr_critic = 0.005f;
    agent.exploration_noise = 1.0f;
    
    // ... main control loop ...
}
```

### 2. Defining the Reward Function
EfRaRL leaves the reward logic entirely to the developer, allowing seamless integration with any robotic plant or sensor array. A common approach for reference tracking is negative Mean Squared Error (MSE).

```cpp
// 1. Read sensors (e.g., current angle, velocity, target angle)
std::array<float, 3> state = { read_angle(), read_velocity(), target_angle };

// 2. Calculate the Reward (Penalty) based on the error
float error = state[0] - state[2];
float mse = (error * error);
float reward = -mse; // The agent will naturally try to maximize this (push to 0.0)
```

### 3. The Control Loop (Training vs Inference)
EfRaRL provides two distinct paths for controlling your hardware: **Training** and **Inference**.

```cpp
// --- A. GETTING ACTION (With Exploration Noise for Training) ---
std::array<float, 2> action = agent.get_action(state);
apply_motor_pwm(action[0], action[1]);

// --- B. TRAINING STEP (Backpropagation in Background) ---
// Note: This function is computationally heavy. On an MCU, 
// call this in your main loop, NOT inside a timer interrupt!
agent.train_step(state, action, reward, next_state, done_flag);
```

For pure **Inference** (e.g., when the model is already trained and you want zero exploration noise):
```cpp
// Pure greedy inference (Extremely fast, safe for ISR)
std::array<float, 2> greedy_action = agent.use_model(state);
apply_motor_pwm(greedy_action[0], greedy_action[1]);
```

### 4. Sim-to-Real Transfer (.EfRa)
You can train the agent heavily in a PC simulation (C++ benchmark), serialize the weights to a hardware-agnostic `float` array, and export it as a tiny `.efra` file.

**On Desktop (Export):**
```cpp
// 1. Get model size
size_t model_size = agent.get_model_size();
float* buffer = new float[model_size];

// 2. Serialize weights to float array
agent.serialize_model(buffer);

// 3. Save to file
std::ofstream out("model.efra", std::ios::binary);
out.write(reinterpret_cast<char*>(buffer), model_size * sizeof(float));
out.close();
```

**On MCU (Import):**
```cpp
// 1. Load the buffer (e.g., from EEPROM, Flash, or Serial)
float buffer[MODEL_SIZE]; 
load_from_eeprom(buffer);

// 2. Inject weights instantly
agent.deserialize_model(buffer);

// 3. Disable backpropagation for pure inference
agent.stop_training(); 
agent.exploration_noise = 0.0f;
```

---

## Performance Metrics (Hardware-in-the-Loop)
*Measured on STM32 Nucleo-F401RE (84 MHz) and Intel Core Ultra 7 255HX.*

| Metric | PC Simulation (x86_64) | STM32 MCU (ARM Cortex-M4) |
| :--- | :--- | :--- |
| **Static RAM Footprint** | 16.4 KB | 16.4 KB |
| **Dynamic Heap Usage** | 0 Bytes | 0 Bytes |
| **Peak IPS (Iterations/sec)** | 262,674 IPS | 298 IPS |
| **Average Step Time** | < 1 $\mu$s | ~3.3 ms |
| **Failsafe Status** | Stable under 10% dropout | Stable under 10% dropout |

## License
MIT License. See `LICENSE` for more details.
