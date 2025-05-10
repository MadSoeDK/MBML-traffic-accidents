
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO, autoguide, Predictive
from pyro.optim import Adam, ClippedAdam
from pyro.contrib.gp.models.model import GPModel as gpr
import torch
import pyro
from torch.distributions import constraints
from torch.linalg import cholesky
from torch.distributions import MultivariateNormal
from pyro.infer.autoguide import AutoDiagonalNormal
#set random seed:
import random
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
pyro.set_rng_seed(0)
import os
os.makedirs("Outputs", exist_ok=True)


#preparing the coords 
print("Preparing Coordinates")
df = pd.read_csv('df_hourly_january.csv', parse_dates=['datetime'])
df.set_index(['datetime', 'cell_id'], inplace=True)

df_day = df.copy()


# Convert UTM (metres) to kilometres for nicer kernel length‑scale priors
coords = df_day[["X", "Y"]].dropna().values / 1_000.0  # shape (n_points, 2)

# Re‑centre so the south‑west corner is (0,0)
coords = coords - coords.min(0, keepdims=True)
coords_orig = df_day[["X","Y"]].dropna().values / 1000.0           # km
offset_km = coords_orig.min(0)    
offset_km = torch.tensor(offset_km, dtype=torch.float32)

# coords: your accident locations in km after centering, shape (N,2)
coords = torch.tensor(coords, dtype=torch.float32).detach().clone()  # reuse your coords array

# Define M quadrature points uniformly over the bounding box of coords
N, _ = coords.shape
M = 1000  # Numper of samples 

# Compute bounding‐box mins and ranges as tensors
x_min, y_min = coords.min(0).values          # both floats
x_max, y_max = coords.max(0).values          # both floats
mins   = torch.tensor([x_min, y_min])        # shape (2,)
ranges = torch.tensor([x_max - x_min,
                       y_max - y_min])      # shape (2,)

# Draw M uniform points in [0,1]^2, then scale & shift
U = torch.rand(M, 2) * ranges[None, :] + mins[None, :]  # shape (M,2)

# Compute domain area in km² for the bounding box
area = (x_max - x_min) * (y_max - y_min)

# Pre-stack all locations for the GP: first the observed points, then quadrature
X_all = torch.cat([coords, U], dim=0)  # shape (N+M, 2)



print("Defining Model")
# RBF kernel function
def rbf_kernel(X1, X2, ls, var):
    d = torch.cdist(X1, X2)
    return var * torch.exp(-0.5 * (d/ls)**2)


def model(coords, U, area):
    """
    coords:   (N,2) tensor of observed accident locations
    U:        (M,2) tensor of quadrature locations (inside study area)
    area:     float, total area of study region
    """
    N = coords.shape[0]
    M = U.shape[0]

    # Sample kernel hyper‐parameters (priors)
    ls    = pyro.sample("lengthscale", dist.LogNormal(0., 0.5))
    var   = pyro.sample("variance",    dist.LogNormal(0., 1.0))
    noise = pyro.sample("noise",       dist.LogNormal(-3., 0.3))

    # Build full covariance over all points
    X_all = torch.cat([coords, U], dim=0)       # shape (N+M, 2)
    K = rbf_kernel(X_all, X_all, ls, var)
    K = 0.5*(K + K.T)                            # enforce symmetry
    K = K + (noise + 1e-4) * torch.eye(N+M)     # jitter + noise
    L = cholesky(K)                             # lower‐Cholesky

    # Sample the latent GP at all points
    f_all = pyro.sample(
        "f_all",
        dist.MultivariateNormal(
            loc=torch.zeros(N+M),
            scale_tril=L
        )
    )

    # Split into observed vs quadrature
    f_obs  = f_all[:N]      # latent at each accident
    f_quad = f_all[N:]      # latent at each quadrature site

    # Log‐likelihood via Monte‐Carlo quadrature 
    with pyro.plate("obs", N):
        # each observed point contributes +f(x_i) to the log‐likelihood
        pyro.factor("obs_ll", f_obs)

    with pyro.plate("quad", M):
        # each quadrature point contributes −(area/M) * exp(f(u_j))
        pyro.factor("quad_ll", - (area / M) * torch.exp(f_quad))


guide = AutoDiagonalNormal(model)

# Inference SVI
pyro.clear_param_store()
optimizer = ClippedAdam({"lr": 0.02})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

print("Running Model")
num_steps = 1000
for step in range(1, num_steps+1):
    loss = svi.step(coords, U, area)
    if step % 100 == 0:
        print(f"[{step:4d}]  ELBO = {-loss:.1f}")



#Getting the postierior distribution out-----------
predictive = Predictive(model, guide=guide, num_samples=1000, return_sites=["f_all"])
samples    = predictive(coords, U, area)

# samples["f_all"] has shape [200, N+M]
f_all_post = samples["f_all"].mean(0)         # Monte-Carlo posterior mean, shape (N+M,)

#-----saving for lator
print("Saving Posterior Means")
np.save("Outputs/f_all_post.npy", f_all_post.numpy())

# slice into obs vs quad
N = coords.size(0)
f_obs_mean  = f_all_post[:N]
f_quad_mean = f_all_post[:,N:]

# compute intensities
λ_obs      = torch.exp(f_obs_mean)
lambda_quad = torch.exp(f_quad_mean).squeeze()


#plotting the posterior:
print("Plotting Posterior Mean Accident Intensity Map")
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point

# Compute absolute UTM for quadrature points
abs_U_km = U + offset_km                   # (M,2) in km
abs_U_m  = abs_U_km * 1000                 # to metres

# Build GeoDataFrame
gdf = gpd.GeoDataFrame(
    {"intensity": lambda_quad.numpy()},
    geometry=[Point(x, y) for x, y in abs_U_m.numpy()],
    crs="EPSG:25832"
)

# Reproject for map tiles
gdf_web = gdf.to_crs(epsg=3857)

# Plot heatmap points over OSM basemap
fig, ax = plt.subplots(figsize=(8,8))
gdf_web.plot(
    column="intensity",
    ax=ax,
    markersize=8,
    alpha=0.7,
    cmap="inferno",
    legend=True,
    legend_kwds={"label":"Predicted intensity λ̂", "shrink":0.6}
)
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
ax.set_axis_off()
ax.set_title("LGCP Posterior Mean Accident Intensity")
plt.tight_layout()


print("Saving Intensity Map")
plt.savefig("Outputs/accident_intensity.png")


