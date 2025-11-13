import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import gc

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import imageio
import numpy as np

#from ray_trax.create_turbulent_3D import generate_correlated_lognormal_field_3D
from ray_trax.utils import gaussian_emissivity 
from ray_trax.ray_trax_3D_tdep import (
    compute_radiation_field_from_multiple_sources_with_time_step
)

# Config
Nx, Ny, Nz = 128, 128, 128
key = random.PRNGKey(111)


# Load your saved 3D field (replace with your actual load logic)
kappa_npz = np.load("./binary_fields/final_state128.npz", allow_pickle=False)

print(kappa_npz.files)

kappa = np.asarray((kappa_npz["arr"][0]))
print(np.shape(kappa))


num_cells = 128

# Process it
#star_positions = jnp.array([
#    ((box_size / 2 - R_forced / 2) / box_size * num_cells, num_cells // 2, num_cells // 2),
#    ((box_size / 2 + R_forced / 2) / box_size * num_cells, num_cells // 2, num_cells // 2),
    
#])

star_positions = [[42, 64, 64], [86, 64, 64]]

emissivity = jnp.zeros((Nx, Ny, Nz), dtype=jnp.float32)
emissivity = gaussian_emissivity(Nx, Ny, Nz, center=(42, 64, 64), amplitude=.1, width=1.) + gaussian_emissivity(Nx, Ny, Nz, center=(86, 64, 64), amplitude=1., width=1.)



dt = 128.0 #time step
c = 1.0  # Speed of light in code units

output_dir = 'plots_binary'
os.makedirs(output_dir, exist_ok=True)

plt.imshow(np.log10(kappa[:,:,64]), origin='lower')
plt.colorbar()
plt.savefig(output_dir+'/kappa_field.png')
plt.close()

plt.imshow(np.log10(emissivity[:,:,64]+1e-10), origin = "lower", cmap = "jet")
plt.colorbar()
plt.savefig(output_dir+'/em_field.png')

mid_z = Nz // 2

emissivity = jnp.asarray(emissivity, dtype=jnp.float32)
kappa = jnp.asarray(kappa, dtype=jnp.float32)

tstart = time.time()
filenames = []

    

J = compute_radiation_field_from_multiple_sources_with_time_step(
        emissivity, kappa, jnp.array([[42., 64., 64.], [86., 64., 64.]]),
        num_rays=int(16*1024),  
        step_size=0.5,
        radiation_velocity=c,
        time_step=dt,
        use_sharding=False,
        unroll = False,
        ray_batch_size = 1024
    )

    # Force evaluation and break JAX graph
J.block_until_ready()
J = np.array(J)

# ----- per-step snapshot (mid-Z, total field) -----
plt.figure(figsize=(6, 5))
plt.imshow(np.log10(J[:, :, mid_z] + 1e-6), origin='lower', cmap='inferno')
plt.title(f"X-Y plane at Z={mid_z}")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar(label="log10(I)")
plt.tight_layout()
filename = os.path.join(output_dir, 'binary.png')
plt.savefig(filename)
filenames.append(filename)
plt.close()


tend = time.time()



