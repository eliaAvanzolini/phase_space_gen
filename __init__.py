# phase_space_gen/__init__.py
"""
Phase Space Generative Models — Physics-Informed Generative AI for Medical MC Simulation

Package structure:
    data/           -- data generation, loading, normalization
    models/         -- NSF, CFM, GAN architectures
    configs/        -- YAML configurations for reproducible experiments
    outputs/        -- training outputs (auto-created)

Quick start:
    from data.synthetic_linac import generate_phase_space
    from models.cfm import PhaseSpaceCFM, CFMTrainer

    ps = generate_phase_space(n_samples=100_000, E_nom=6.0, jaw_x=5.0, jaw_y=5.0)
"""
