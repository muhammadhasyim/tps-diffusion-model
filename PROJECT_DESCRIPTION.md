## Project Description: Fine-Tuning and Evaluation Framework for Protein Co-Folding Models

This project aims to develop a **fine-tuning algorithm** and a corresponding set of **evaluation protocols** for protein co-folding models.

### Overview

The core idea is to generate high-quality reference data from molecular simulations, use that data to fine-tune a co-folding model, and then evaluate whether the trained model can reproduce meaningful thermodynamic behavior, such as free energy surfaces.

### Reference Data Generation

Reference data will be generated using:

* **OpenMM molecular dynamics simulations**
* **OPES**: *On-the-fly Probability Enhanced Sampling*

These simulations will produce:

* Protein complex structures
* Reweighting factors

The reweighting factors can later be used for post-processing tasks, such as estimating **free energy surfaces**.

### Model Fine-Tuning

Once the simulation data has been generated, the next step is to use it to **supervised fine-tune a co-folding model**.

The current plan is to use the standard **score-function diffusion model training objective**. This is a simulation-free training approach, meaning the model is trained directly from generated data rather than requiring molecular dynamics during training.

### Evaluation

After fine-tuning, the model will be evaluated by checking whether it can generate samples that reproduce the expected thermodynamic landscape.

The main evaluation target is:

* Constructing a **free energy surface** from model-generated structures

To do this, the project will use the **TPS-OPES algorithm** that already exists in this folder.

### Out of Scope for Now

The reinforcement learning algorithm currently in the folder should be ignored for this stage of the project.

It will be set aside for now and revisited later if needed.

### Current Workflow

1. Generate reference data using OpenMM molecular dynamics and OPES.
2. Save structures and reweighting factors from the simulations.
3. Use the generated simulation data to supervised fine-tune a protein co-folding model.
4. Train the model using a score-function diffusion objective.
5. Generate structures from the fine-tuned model.
6. Use TPS-OPES to construct free energy surfaces from model-generated samples.
7. Compare these free energy surfaces against the reference simulation-derived surfaces.
