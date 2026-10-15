import h5py
import numpy as np

def get_data(filepath):
    with h5py.File(filepath, 'r') as f:
        key = 'phase_space' if 'phase_space' in f else ('data' if 'data' in f else list(f.keys())[0])
        return f[key][:], key

cfm_data, key_cfm = get_data("gen/gen_3.5_hybrid.h5")
shift_data, _ = get_data("gen/gen_3.5_shift_k101.h5")

n = min(len(cfm_data), len(shift_data))
cfm_data = cfm_data[:n]
shift_data = shift_data[:n]

# Target Spearman correlazione interpolata a 3.5 cm
rho_target = -0.22

# 1. Calcolo rank dell'angolo di scattering theta del CFM
theta = np.hypot(cfm_data[:, 3], cfm_data[:, 4])
rank_theta = np.argsort(np.argsort(theta)).astype(np.float64) / n

# 2. Generazione variabile di accoppiamento con correlazione target
rng = np.random.default_rng(42)
noise = rng.standard_normal(n)
rank_noise = np.argsort(np.argsort(noise)).astype(np.float64) / n

# Variabile latente con correlazione rank voluta
latent = rho_target * rank_theta + np.sqrt(1.0 - rho_target**2) * rank_noise
idx_order = np.argsort(latent)

# 3. Assegnazione dell'energia ordinata della baseline analitica
e_sorted = np.sort(shift_data[:, 6])
e_matched = np.empty_like(e_sorted)
e_matched[idx_order] = e_sorted

# 4. Composizione del phase space ibrido
data_hybrid = np.copy(cfm_data)
data_hybrid[:, 6] = e_matched

out_path = "gen/gen_3.5_copula_hybrid.h5"
with h5py.File(out_path, 'w') as f:
    f.create_dataset(key_cfm, data=data_hybrid)

print(f"Ibrido con copula salvato in {out_path}")
print(f"Verifica Spearman(theta, E): {np.corrcoef(rank_theta, np.argsort(np.argsort(e_matched))/n)[0,1]:.3f}")
