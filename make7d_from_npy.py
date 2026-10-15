import numpy as np
from train_gen_energy_1d import generate_full_7d_with_condE

for name in ["resq", "shift101"]:
    E = np.load(f"gen/E_3.5_{name}.npy")
    generate_full_7d_with_condE("runs/cfm_condE_s42", 3.5, E,
                                f"gen/gen_3.5_neural_{name}_condE.h5")
