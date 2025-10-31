# utils/render.py
import torch
import torch.nn.functional as F

EPS = 1e-10

def sample_stratified(rays_o, rays_d, near, far, N_samples, perturb=True, lindisp=False, device='cpu'):
    N_rays = rays_o.shape[0]
    t_lin = torch.linspace(0.0, 1.0, N_samples, device=device).expand(N_rays, N_samples)

    if lindisp:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_lin) + 1.0 / far * t_lin)
    else:
        z_vals = near * (1.0 - t_lin) + far * t_lin

    if perturb:
        mids = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
        upper = torch.cat([mids, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], mids], -1)
        z_vals = lower + (upper - lower) * torch.rand_like(z_vals)

    pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)
    return z_vals, pts

def raw2outputs(rgb, sigma, z_vals, rays_d, white_bkgd=False):
    device = rgb.device
    
    # ====== CRITICAL FIXES ======
    # 1. Apply sigmoid to RGB to bound it to [0, 1]
    rgb = torch.sigmoid(rgb)
    
    # 2. Apply ReLU to sigma to ensure positive density
    sigma = F.relu(sigma)
    # ============================
    sigma = sigma * 0.00001

    deltas = z_vals[:, 1:] - z_vals[:, :-1]
    deltas = torch.cat([deltas, 1e10 * torch.ones_like(deltas[:, :1], device=device)], dim=-1)

    ray_norm = torch.norm(rays_d, dim=-1, keepdim=True)
    deltas = deltas * ray_norm  # shape [N_rays, N_samples]

    # Make sure sigma is [N_rays, N_samples]
    sigma = sigma.squeeze(-1)

    alpha = 1.0 - torch.exp(-sigma * deltas)  # [N_rays, N_samples]
    trans = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1.0 - alpha + EPS], -1), -1)[:, :-1]
    weights = alpha * trans  # [N_rays, N_samples]

    # Ensure rgb is [N_rays, N_samples, 3]
    if rgb.shape[-1] != 3:
        rgb = rgb.repeat(1, 1, 3)

    comp_rgb = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)

    if white_bkgd:
        comp_rgb = comp_rgb + (1.0 - acc_map.unsqueeze(-1))

    return comp_rgb, depth_map, acc_map, weights


def sample_pdf(bins, weights, N_samples, det=False):
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)

    N_rays = cdf.shape[0]
    if det:
        u = torch.linspace(0.0, 1.0 - 1.0 / N_samples, steps=N_samples, device=bins.device)
        u = u.expand(N_rays, N_samples)
    else:
        u = torch.rand(N_rays, N_samples, device=bins.device)

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)

    cdf_below = torch.gather(cdf, 1, below)
    cdf_above = torch.gather(cdf, 1, above)
    bins_below = torch.gather(bins, 1, below)
    bins_above = torch.gather(bins, 1, above)

    denom = cdf_above - cdf_below
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_below) / denom
    samples = bins_below + t * (bins_above - bins_below)
    return samples


def render_rays(models, pos_enc, dir_enc, rays_o, rays_d, 
                N_samples=64, N_importance=128, near=0.0, far=1.0,
                perturb=True, lindisp=False, white_bkgd=False, **kwargs):
    
    device = rays_o.device
    N_rays = rays_o.shape[0]

    # --- Coarse sampling ---
    z_vals_coarse, pts_coarse = sample_stratified(rays_o, rays_d, near, far, N_samples, perturb, lindisp, device)

    pts_flat = pts_coarse.reshape(-1, 3)
    dirs = rays_d.unsqueeze(1).expand_as(pts_coarse)
    dirs_flat = dirs.reshape(-1, 3)

    encoded_pts = pos_enc(pts_flat)
    encoded_dirs = dir_enc(dirs_flat)
    
    sigma_coarse_flat, rgb_coarse_flat = models['coarse'](encoded_pts, encoded_dirs)
    
    sigma_coarse = sigma_coarse_flat.view(N_rays, N_samples, -1)
    rgb_coarse = rgb_coarse_flat.view(N_rays, N_samples, 3)
    
    comp_rgb_coarse, depth_coarse, acc_coarse, weights_coarse = raw2outputs(
        rgb_coarse, sigma_coarse, z_vals_coarse, rays_d, white_bkgd
    )

    results = { 
        'rgb_map0': comp_rgb_coarse,
        'acc_map0': acc_coarse,  # ← ADD THIS
        'depth_map0': depth_coarse  # ← ADD THIS
    }

    # --- Fine sampling ---
    if N_importance > 0:
        z_vals_mid = 0.5 * (z_vals_coarse[:, :-1] + z_vals_coarse[:, 1:])
        z_samples = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1], N_importance, det=not perturb)
        z_vals_fine, _ = torch.sort(torch.cat([z_vals_coarse, z_samples], -1), -1)
        
        pts_fine = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals_fine.unsqueeze(-1)

        pts_flat_fine = pts_fine.reshape(-1, 3)
        dirs_fine = rays_d.unsqueeze(1).expand_as(pts_fine)
        dirs_flat_fine = dirs_fine.reshape(-1, 3)

        encoded_pts_fine = pos_enc(pts_flat_fine)
        encoded_dirs_fine = dir_enc(dirs_flat_fine)
        
        sigma_fine_flat, rgb_fine_flat = models['fine'](encoded_pts_fine, encoded_dirs_fine)
        
        sigma_fine = sigma_fine_flat.view(N_rays, N_samples + N_importance, -1)
        rgb_fine = rgb_fine_flat.view(N_rays, N_samples + N_importance, 3)
        
        comp_rgb_fine, depth_fine, acc_fine, _ = raw2outputs(
            rgb_fine, sigma_fine, z_vals_fine, rays_d, white_bkgd
        )
        
        results['rgb_map'] = comp_rgb_fine
        results['acc_map'] = acc_fine      # ← ADD THIS
        results['depth_map'] = depth_fine  # ← ADD THIS
        
    return results