# High Energy Physics & AI Engineering Research 🚀
[![HSF Training](https://img.shields.io/badge/HSF-Software_Training-blue?logo=github)](https://hsf-training.org/)
[![OS: Arch Linux](https://img.shields.io/badge/OS-Arch_Linux-1793D1?logo=arch-linux&logoColor=white)](https://archlinux.org/)
[![Env: AlmaLinux 9](https://img.shields.io/badge/Container-AlmaLinux_9-FF8A00?logo=almalinux&logoColor=white)](https://almalinux.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

I am an Artificial Intelligence student at the **Universidade de Vigo (UVigo)**. This repository serves as a specialized research laboratory focusing on the intersection of **High-Performance Computing (HPC)**, **Deterministic AI**, and **Particle Physics**. This work is pecifically designed as a technical portfolio for **GSoC 2026**

---

## 📚 Project Structure

### [01] Foundations: HSF Training & Mathematical Physics
Following the **HEP Software Foundation (HSF)** standards, this module establishes the groundwork for professional-grade scientific software development.
* **Software Sustainability:** Implementation of clean code principles, version control (Conventional Commits), and modular Python development.
* **HSF Curriculum:** Transitioning through the HSF training modules including ML for HEP, Python analysis (Scikit-HEP), and Unix shell optimization.
* **Physics Modeling:** Theoretical study of the Standard Model (SM), focusing on particle-matter interaction and signal induction in calorimeters.

### [02] ATLAS TileCal: Signal Reconstruction & Pulse Analysis
The **Tile Calorimeter (TileCal)** is crucial for measuring hadron energy at ATLAS. This module explores Deep Learning as an alternative to traditional **Optimal Filtering (OF)**.
* **The Challenge:** In the HL-LHC, the high collision rate causes **Out-of-time pile-up**, where pulses from different events overlap within the same 150ns window.
* **Research Pipeline:**
    1. **Data Generation:** Simulating realistic TileCal pulses with Gaussian noise and multi-pulse pile-up.
    2. **Benchmark:** Proving the failure of standard MLP architectures under high-noise conditions.
    3. **Innovation:** Implementation of a **Temporal Recurrent Network (LSTM)** in PyTorch that treats the 7 ADC samples as a sequence, significantly improving energy resolution.



### [03] CMS Archi: Decoupled Agentic Workflows with MCP
Applying modern systems engineering to physics operations by implementing the **Model Context Protocol (MCP)**.
* **Objective:** Building a reliable AI "Research Assistant" for the CMS experiment.
* **Architecture:** Using a decoupled approach where the LLM (Reasoning) is separated from the Physics Tools (Execution).
* **Deterministic Tools:** Custom MCP servers providing real-time data access, invariant mass calculations, and system monitoring within an AlmaLinux 9 environment.



---

## 🛠️ Technical Stack & Standards

| Component | Technology |
| :--- | :--- |
| **Scientific Stack** | Scikit-HEP, PyTorch, Scikit-Learn |
| **Training Standards** | HSF (HEP Software Foundation) |
| **Runtime Environment** | AlmaLinux 9 (CERN Standard) |
| **Development** | Arch Linux / Podman Containers |
| **AI Protocols** | Model Context Protocol (MCP) for tool-use and agentic reasoning |

---

## 🚀 Installation & Reproducibility

To ensure a deterministic environment identical to CERN's production nodes:

```bash
# 1. Start the AlmaLinux 9 container
podman run -it almalinux:9 /bin/bash

# 2. Clone the repository
git clone [https://github.com/hugo-martinez-hme/hep-data-analysis.git](https://github.com/hugo-martinez-hme/hep-data-analysis.git)
cd hep-data-analysis

# 3. Install dependencies from the requirements manifest
pip install -r requirements.txt
