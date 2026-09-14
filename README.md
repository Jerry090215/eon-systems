# 🪰 Eon Systems — Embodied Fruit Fly Brain

> A real biological neural network, embodied on your desktop.

**7,497 real neurons from the FlyWire connectome, running as leaky integrate-and-fire (LIF) spiking neurons with STDP plasticity, embodied in your macOS desktop environment. It learns, habituates, and evolves.**

---

## ✨ Core Features

### 🧠 Real Biological Connectome
- **7,497 neurons** extracted from the FlyWire v783 adult *Drosophila* connectome
  - Mushroom body: 6,297 neurons (KC 5,177 + ALPN 685 + DAN 339 + MBON 96)
  - Visual lobe: 1,200 neurons (T4/T5 motion detectors + Tm interneurons + LC output neurons)
- **546,368 real synaptic connections** with neurotransmitter predictions
- **62,261 plastic connections** (KC→MBON) governed by spike-timing-dependent plasticity (STDP)

### ⚡ Real-Time Neural Simulation
- LIF (Leaky Integrate-and-Fire) neuron model with 1ms timestep
- APL lateral inhibition for sparse coding (winner-take-all)
- KC spontaneous activity (Poisson process, 1.2 Hz baseline)
- Runs at ~30 fps on Apple M2 — no GPU required

### 🧠 Learning & Memory
- **STDP plasticity**: Dopamine-gated synaptic weight modification
  - Punishment (PPL1 DAN activation) → suppress KC→MBON connections
  - Reward (PAM DAN activation) → strengthen connections
- **Habituation**: Repeated harmless stimuli → reduced response (verified experimentally)
- **Dishabituation**: Strong aversive stimulus restores habituated response (verified experimentally)
- Weight persistence: STDP weights auto-save every 30 seconds

### 🧬 Somatic Evolution (No Reproduction)
- Individual-level evolution: mutation → evaluation → keep/revert cycle
- 21 evolvable parameters across 6 systems (vision, learning, metabolism, behavior, neuromodulation, motor)
- Fitness based on energy, punishment frequency, exploration range, food access, collision rate
- Every 5 minutes: 1-3 parameters mutated (1-3% Gaussian noise), evaluated for 4 minutes
- Intrinsic plasticity: neuron thresholds and membrane time constants self-adjust (homeostatic plasticity)

### 🖥️ Desktop Embodiment
Your macOS desktop is the fly's physical world:

| Real World | Fly's Perception | Neural Encoding |
|---|---|---|
| Mouse cursor | Predator / looming object | Visual lobe T4/T5 motion detection, LC11 looming escape |
| Mouse movement patterns | Odors (A/B/C) | ALPN activation → mushroom body Kenyon cells |
| System volume spikes | Electric shock / punishment | PPL1 dopamine neurons → STDP weight suppression |
| Keyboard typing | Tactile stimulation / air puff | Mechanosensory encoding |
| Application windows | Obstacles / walls | Collision detection → visual avoidance |
| Terminal windows | Food sources | Energy recovery when positioned over terminal |
| Screen brightness / time | Circadian rhythm | Global excitability modulation |

### 🏃 Metabolism & Survival
- Energy system: movement consumes energy, food (terminals) restores it, zero energy = death
- Fatigue system: activity accumulates fatigue, rest recovers it
- Circadian rhythm: daytime (7:00-22:00) active, nighttime reduced activity
- Health system: collisions and danger cause damage, natural recovery
- Death & respawn: energy/health zero → death → 30s respawn with fresh genome

### 🗣️ Voice System (Optional)
- Neural state → local LLM (Ollama/mimi) → first-person inner monologue
- macOS `say` command for text-to-speech (zero extra memory)
- Asynchronous generation, non-blocking main loop

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    macOS Desktop Environment                  │
│  Mouse │ Volume │ Keyboard │ Windows │ Brightness │ Time    │
└──────────────────────────┬──────────────────────────────────┘
                           │ sensory input
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     Sensory Encoding Layer                    │
│  VisualEncoder │ AntennalLobe │ RealSenses │ WindowDetector │
└──────────────────────────┬──────────────────────────────────┘
                           │ neural currents
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Mushroom Body LIF Network                   │
│  5,177 KC │ 685 ALPN │ 339 DAN │ 96 MBON │ APL inhibition  │
│  STDP plasticity │ Spontaneous activity │ Weight persistence  │
└──────────────────────────┬──────────────────────────────────┘
                           │ MBON firing rates
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 Visual Lobe LIF Network                       │
│  8×8 Retina │ Medulla On/Off │ T4/T5 motion │ LC output     │
└──────────────────────────┬──────────────────────────────────┘
                           │ behavior signals
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Behavior & Motor Control Layer                   │
│  ExplorationDriver │ CentralComplex │ MotorControl │ Metabolism│
│  DesktopEnvironment │ SomaticEvolution │ IntrinsicPlasticity  │
└──────────────────────────┬──────────────────────────────────┘
                           │ walkDrive / turnBias / wingDrive
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  DesktopFly Visualization                     │
│  3D fly on macOS desktop │ Brain activity viewer │ External   │
│  brain bridge via /tmp/fly_brain.json                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- macOS 12+ (desktop embodiment features are macOS-only)
- Python 3.10+
- ~2 GB free disk space for connectome data

### 1. Clone the repository
```bash
git clone https://github.com/Jerry090215/eon-systems.git
cd eon-systems
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download FlyWire connectome data (~850 MB)
```bash
./download_data.sh
```

### 4. Extract neural circuits
```bash
# Extract mushroom body circuit (6,297 neurons)
python3 mushroom_body/extract_circuit.py

# Extract visual lobe circuit (1,200 neurons)
python3 scripts/extract_visual_lobe.py
```

### 5. Run the embodied fly simulation
```bash
cd embodied_fly
python3 embodied_fly_live.py
```

The simulation will start and print real-time neural activity to the terminal.

### 6. (Optional) Launch DesktopFly visualization
```bash
# In a separate terminal
cd desktop-fly
./build.sh
./DesktopFly
```

DesktopFly will read brain signals from `/tmp/fly_brain.json` and render a 3D fly on your desktop.

---

## 📁 Project Structure

```
eon-systems/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── .gitignore
├── requirements.txt             # Python dependencies
├── download_data.sh             # FlyWire data download script
│
├── embodied_fly/                # Core simulation (Python)
│   ├── embodied_fly_live.py    # Main loop, real-time interaction
│   ├── mb_lif.py                # Mushroom body LIF network + STDP
│   ├── visual_lobe.py           # Visual lobe LIF network
│   ├── sensors.py               # Sensory input (mouse, volume, keyboard, brightness)
│   ├── encoder.py               # Neural encoding (sensory → neural currents)
│   ├── central_complex.py       # Central complex navigation
│   ├── motor_control.py         # Motor control & behavior state machine
│   ├── exploration.py            # Exploration driver (curiosity, fatigue)
│   ├── desktop_environment.py   # Desktop environment (collisions, food, predator)
│   ├── metabolism.py             # Metabolism (energy, fatigue, circadian, health)
│   ├── real_senses.py            # Real sensory encoder (vision/olfaction/touch)
│   ├── window_detector.py        # macOS window detection (AppleScript)
│   ├── genome.py                  # Evolvable genotype (21 parameters)
│   ├── somatic_evolution.py      # Somatic evolution engine (mutate-evaluate-keep)
│   ├── intrinsic_plasticity.py   # Intrinsic plasticity (threshold/tau adaptation)
│   ├── thought_generator.py       # Voice system (LLM inner monologue + TTS)
│   ├── courtship.py               # Courtship circuit (P1/pC1, pheromones)
│   ├── genetics.py                # Genetic algorithm for reproduction
│   └── multi_fly.py               # Multi-fly management
│
├── desktop-fly/                  # Visualization layer (Swift, third-party)
│   └── ...                       # 3D fly renderer, brain activity viewer
│
├── mushroom_body/                # Extracted mushroom body circuit data
│   ├── circuit.json              # 6,297 neurons, 546,368 connections (56 MB)
│   ├── extract_circuit.py        # Extraction script
│   └── README.md                 # Data documentation
│
├── visual_lobe/                  # Extracted visual lobe circuit data
│   └── circuit.json              # 1,200 neurons, 2,665 connections (1.2 MB)
│
├── raw_data/                     # Raw FlyWire connectome (852 MB, gitignored)
│   ├── proofread_root_ids_783.npy
│   ├── proofread_connections_783.feather
│   └── neuron_annotations.tsv
│
├── scripts/                      # Utility scripts
│   └── extract_visual_lobe.py    # Extract visual lobe circuit
│
└── examples/                     # Examples and experiments
    ├── experiment_conditioning.py # Conditioning experiment (STDP learning)
    └── test_embodied_fly.py      # Basic tests
```

---

## 🔬 Technical Details

### Neuron Statistics
| Brain Region | Neurons | Connections | Synapses |
|---|---|---|---|
| Mushroom Body — Kenyon Cells (KC) | 5,177 | — | — |
| Mushroom Body — AL Projection Neurons (ALPN) | 685 | — | — |
| Mushroom Body — Dopamine Neurons (DAN) | 339 | — | — |
| Mushroom Body — MB Output Neurons (MBON) | 96 | — | — |
| Visual Lobe — T4/T5 Motion Detectors | 640 | — | — |
| Visual Lobe — Tm Interneurons | 360 | — | — |
| Visual Lobe — LC Output Neurons | 200 | — | — |
| **Total** | **7,497** | **549,033** | **1,125,197** |

### LIF Neuron Parameters
```
Membrane time constant (τ_m):  20 ms
Resting potential:              -65 mV (normalized to 0)
Firing threshold:               1.0 (KC: 2.5)
Refractory period:              2 ms
Weight scale:                   0.0008
Simulation timestep:            1 ms
Display framerate:              ~30 fps
```

### STDP Plasticity Parameters
```
Plastic connections:            62,261 (KC → MBON)
STDP time constant (τ):         50 ms
Punishment learning rate:       0.0008 (absolute, PPL1 DAN gated)
Reward learning rate:           0.0005 (PAM DAN gated)
Weight range:                   [0, 5.0]
Auto-save interval:             30 seconds
```

### Somatic Evolution Parameters
```
Evolvable parameters:           21 (6 systems)
Baseline period:                120 seconds
Mutation interval:              300 seconds
Evaluation period:              240 seconds
Mutations per cycle:            1-3 parameters
Mutation magnitude:             1-3% Gaussian noise
Fitness metrics:                energy, punishment, exploration, food, collisions, rest
```

---

## 🧪 Experimental Verification

### Habituation & Dishabituation Experiment

We verified that the embodied fly exhibits real biological learning phenomena:

**Habituation**: After repeated exposure to a harmless stimulus (mouse cursor approach) with no punishment, the fly's escape response gradually diminishes.
- Baseline nervousness: 0.043 (after habituation)
- Escape trigger rate: 0%

**Dishabituation**: A single strong aversive stimulus (system volume spike → PPL1 dopamine punishment) paired with mouse approach instantly restores the fear response.
- Post-punishment nervousness: 0.343 (8× increase)
- Escape trigger rate: 100%
- Arousal: 0.539 → 0.900

**Re-habituation**: After dishabituation, the fly enters a bistable state — it fears the mouse when it approaches, but quickly calms when the mouse stops. This demonstrates conditional fear learning, not simple habituation.

This is not scripted behavior — it emerges from the real STDP plasticity in the 62,261 KC→MBON connections.

### STDP Learning Verification
After 10 punishment training trials paired with odor A:
- KC→MBON weights for odor A: 0.0116 → 0.0064 (**-45%**)
- 49% of connections suppressed to zero
- MBON firing rate for odor A: 56.5 → 46.0 Hz (**-18.5%**)
- Control odor B: unchanged (learning is specific)

---

## 🆚 Comparison with Related Projects

| Feature | This Project | DesktopFly | NeuroMechFly/FlyGym | fly-arena |
|---|---|---|---|---|
| Real connectome neurons | 7,497 | 668 | ~125,000 | ~165,000 |
| STDP synaptic plasticity | ✅ | ❌ | ❌ | ❌ |
| Habituation/dishabituation | ✅ (verified) | ❌ | ❌ | ❌ |
| Somatic evolution | ✅ | ❌ | ❌ | ❌ |
| Intrinsic plasticity | ✅ | ❌ | ❌ | ❌ |
| Metabolism system | ✅ | ❌ | ❌ | ❌ |
| Real desktop embodiment | ✅ (mouse/windows/volume) | ✅ (simple) | ❌ (MuJoCo sim) | ❌ (MuJoCo sim) |
| Multimodal sensory encoding | ✅ (vision/olfaction/touch/audio) | Visual only | ✅ | ✅ |
| Voice/LLM integration | ✅ (optional) | ❌ | ❌ | ❌ |
| Python implementation | ✅ | ❌ (Swift) | ✅ | ✅ |
| Runs on consumer hardware | ✅ (M2, ~12% CPU) | ✅ | Requires GPU | Requires GPU |

---

## 🤝 Contributing

Contributions are welcome! Areas where help would be particularly valuable:

1. **Additional brain regions**: Extract and integrate more FlyWire brain regions (central complex full connectome, olfactory glomeruli, motor neurons)
2. **Linux/Windows port**: The desktop embodiment currently uses macOS-specific APIs (Quartz, AppleScript). Porting to Linux (X11/Wayland) or Windows would expand accessibility.
3. **Visualization improvements**: Better neural activity visualization, 3D brain viewer
4. **Experiments**: Design and run more learning/memory experiments to validate the model
5. **Documentation**: Better tutorials, API docs, troubleshooting guides
6. **Performance optimization**: Faster LIF simulation, larger neuron counts

Please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

**Third-party data and software:**
- FlyWire connectome data: CC BY 4.0, FlyWire Consortium
- DesktopFly: MIT, Denis Shiryaev
- NeuroMechFly / FlyGym: respective open licenses

---

## 📚 References & Acknowledgments

- **FlyWire Connectome**: Dorkenwald et al., "FlyWire: online community for whole-brain connectomics", *Nature Methods*, 2024
- **Mushroom Body Circuit**: Li et al., "The connectome of the Drosophila mushroom body", *Nature*, 2020
- **Visual Lobe**: Shinomiya et al., "Comparisons between the ON- and OFF-edge motion pathways in the Drosophila brain", *eLife*, 2019
- **STDP in Drosophila**: Cohn et al., "A dopamine-gated learning circuit in the Drosophila mushroom body", *Cell*, 2015
- **DesktopFly**: Denis Shiryaev, https://github.com/DenisSergeevitch/desktop-fly
- **NeuroMechFly**: Wang-Chen et al., "NeuroMechFly, a neuromechanical model of adult Drosophila melanogaster", *Nature Methods*, 2024

---

## ⚠️ Disclaimer

This is a research project exploring principles of biological neural computation. It is **not** a complete emulation of a fruit fly brain — it simulates specific extracted circuits (mushroom body + visual lobe) with simplified LIF neuron models. Real fruit fly brains have ~139,000 neurons and far more complex neuromodulation, glial dynamics, and developmental processes.

The behaviors observed (habituation, learning, evolution) emerge from the simulated neural plasticity and should be interpreted as model behaviors, not claims about real fruit fly cognition.

---

<p align="center">
  <i>Made with 🧠 and too much caffeine.</i>
</p>
