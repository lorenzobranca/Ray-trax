import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnames=["num_rays", "step_size", "max_steps"])
def compute_radiation_field_from_source(j_map, kappa_map, source_pos, num_rays=360, step_size=0.5, max_steps=1000):
    Nx, Ny = j_map.shape
    angles = jnp.linspace(0, 2 * jnp.pi, num_rays, endpoint=False)
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)



    def bilinear_op(grid, x, y, value=None, mode="interp"):
        """
        Unified bilinear operator for interpolation or deposition.

        Parameters:
        - grid: 2D array
        - x, y: float coordinates
        - value: value to deposit (only needed in mode="deposit")
        - mode: "interp" or "deposit"

        Returns:
        - interpolated value if mode == "interp"
        - updated grid if mode == "deposit"
        """
        Nx, Ny = grid.shape
        x0 = jnp.floor(x).astype(int)
        y0 = jnp.floor(y).astype(int)
        x1 = jnp.clip(x0 + 1, 0, Nx - 1)
        y1 = jnp.clip(y0 + 1, 0, Ny - 1)
        x0 = jnp.clip(x0, 0, Nx - 1)
        y0 = jnp.clip(y0, 0, Ny - 1)

        dx = x - x0
        dy = y - y0

        w00 = (1 - dx) * (1 - dy)
        w01 = (1 - dx) * dy
        w10 = dx * (1 - dy)
        w11 = dx * dy

        if mode == "interp":
            return (grid[x0, y0] * w00 +
                    grid[x0, y1] * w01 +
                    grid[x1, y0] * w10 +
                    grid[x1, y1] * w11)
        
        

        elif mode == "deposit":
            assert value is not None, "Value must be provided for deposition"
            updates = jnp.zeros_like(grid)
            updates = updates.at[x0, y0].add(w00 * value)
            updates = updates.at[x0, y1].add(w01 * value)
            updates = updates.at[x1, y0].add(w10 * value)
            updates = updates.at[x1, y1].add(w11 * value)

            return grid + updates
        else:
            raise ValueError("mode must be either 'interp' or 'deposit'")
    
    def trace_single_ray(direction):

        def body_fn(i, state):
            x, y, I, tau, J = state
            j_val = bilinear_op(j_map, x, y, mode="interp")
            kappa_val = bilinear_op(kappa_map, x, y, mode="interp")
            ds = step_size
            d_tau = kappa_val * ds
            dI = j_val * jnp.exp(-tau) * ds

            I_new = I*jnp.exp(-d_tau) + dI
            tau_new = tau + d_tau
            J = bilinear_op(J, x, y, value=I_new, mode="deposit")
            x_new = x + direction[0] * ds
            y_new = y + direction[1] * ds
            return (x_new, y_new, I_new, tau_new, J)
            
        x0, y0 = source_pos
        initial = (x0, y0, 0.0, 0.0, jnp.zeros_like(j_map))
        _, _, _, _, J_ray = jax.lax.fori_loop(0, max_steps, body_fn, initial)

        return J_ray
    
 
    
    J_all = jax.vmap(trace_single_ray)(directions)
    return jnp.sum(J_all, axis=0)* (2 * jnp.pi / num_rays)




def compute_radiation_field_from_multiple_sources(
    j_map, kappa_map, source_positions,
    num_rays=360, step_size=0.5, max_steps=1000
    ):
    J_total = jnp.zeros_like(j_map)
    for source_pos in source_positions:
        J_total += compute_radiation_field_from_star(
            j_map, kappa_map, source_pos,
            num_rays=num_rays,
            step_size=step_size,
            max_steps=max_steps
        )
    return J_total



