import abc
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math


def differential_entropy(pdf, x_pdf):
    # pdf is a vector because we want to perform a numerical integration
    pdf = pdf + 1e-4
    f = -1 * pdf * torch.log(pdf)
    # Integrate using the composite trapezoidal rule
    ans = torch.trapz(f, x_pdf, dim=-1).mean()
    return ans


class GenerativeModel(abc.ABC, nn.Module):
    """Base class inherited by all generative models in pytorch-generative.
    Provides:
        * An abstract `sample()` method which is implemented by subclasses that support
          generating samples.
        * Variables `self._c, self._h, self._w` which store the shape of the (first)
          image Tensor the model was trained with. Note that `forward()` must have been
          called at least once and the input must be an image for these variables to be
          available.
        * A `device` property which returns the device of the model's parameters.
    """

    def __call__(self, *args, **kwargs):
        if getattr(self, "_c", None) is None and len(args[0].shape) == 4:
            _, self._c, self._h, self._w = args[0].shape
        return super().__call__(*args, **kwargs)

    @property
    def device(self):
        return next(self.parameters()).device

    @abc.abstractmethod
    def sample(self, n_samples):
        ...


class Kernel(abc.ABC, nn.Module):
    """Base class which defines the interface for all kernels."""

    def __init__(self, bandwidth=0.01):
        """Initializes a new Kernel.
        Args:
            bandwidth: The kernel's (band)width.
        """
        super().__init__()
        self.bandwidth = bandwidth

    def _diffs(self, test_Xs, train_Xs):
        """Computes difference between each x in test_Xs with all train_Xs."""
        test_Xs = test_Xs.view(test_Xs.shape[0], test_Xs.shape[1], 1)
        train_Xs = train_Xs.view(train_Xs.shape[0], 1, *train_Xs.shape[1:])
        return test_Xs - train_Xs

    @abc.abstractmethod
    def forward(self, test_Xs, train_Xs):
        """Computes p(x) for each x in test_Xs given train_Xs."""

    @abc.abstractmethod
    def sample(self, train_Xs):
        """Generates samples from the kernel distribution."""


class GaussianKernel(Kernel):
    """Implementation of the Gaussian kernel."""

    def forward(self, test_Xs, train_Xs):
        diffs = self._diffs(test_Xs, train_Xs)
        dims = tuple(range(len(diffs.shape))[2:])
        var = self.bandwidth ** 2
        exp = torch.exp(- torch.pow(diffs,2) / (2 * var))
        coef = 1 / torch.sqrt(torch.tensor(2 * np.pi * var))
        return (coef * exp).mean(dim=-1)

    def sample(self, train_Xs):
        device = train_Xs.device
        noise = torch.randn(train_Xs.shape) * self.bandwidth
        return train_Xs + noise


class KernelDensityEstimator(GenerativeModel):
    """The KernelDensityEstimator model."""

    def __init__(self, train_Xs, kernel=None):
        """Initializes a new KernelDensityEstimator.
        Args:
            train_Xs: The "training" data to use when estimating probabilities.
            kernel: The kernel to place on each of the train_Xs.
        """
        super().__init__()
        self.kernel = kernel or GaussianKernel()
        self.train_Xs = train_Xs

    @property
    def device(self):
        return self.train_Xs.device

    # TODO(eugenhotaj): This method consumes O(train_Xs * x) memory. Implement an
    # iterative version instead.
    def forward(self, x):
        return self.kernel(x, self.train_Xs)

    def sample(self, n_samples):
        idxs = np.random.choice(range(len(self.train_Xs)), size=n_samples)
        return self.kernel.sample(self.train_Xs[idxs])


def gen_rand_img(cali_data):
    """
    Generate gaussian images.
    """
    if cali_data.ndim == 4:
        image_shape = cali_data.shape[1:]
        num_samples = cali_data.shape[0]
    else:
        image_shape = (3, 224, 224)
        num_samples = 32

    cali_data = torch.randn(num_samples, *image_shape)
    return cali_data


def random_crop_params(B, H, W, min_ratio=0.3, max_ratio=0.7, device='cpu'):
    """
    Generate crop coordinate (top, left, h, w) for each image.
    """
    params = []
    for _ in range(B):
        r = torch.empty(1, device=device).uniform_(min_ratio, max_ratio).item()
        win_area = int(r * H * W)
        win_h = int((win_area * H / W) ** 0.5)
        win_w = int(win_h * W / H)
        top = torch.randint(0, H - win_h + 1, (1,), device=device).item()
        left = torch.randint(0, W - win_w + 1, (1,), device=device).item()
        params.append((top, left, win_h, win_w))
    return params


def crop_and_resize(batch, crop_params, out_size=224):
    """
    Crop batch images according to the crop parameters and resize to 224x224.
    """
    patches = []
    for img, (top, left, h, w) in zip(batch, crop_params):
        patch = img[:, top:top + h, left:left + w]
        patch = F.interpolate(
            patch.unsqueeze(0), size=(out_size, out_size),
            mode='bilinear', align_corners=False
        ).squeeze(0)
        patches.append(patch)
    return torch.stack(patches, dim=0)


def gen_perturb(img,
                p_flip=0.5,
                max_deg=15,
                jitter_strength=0.4,
                p_blur=0.3,
                p_erase=0.3):
    """
    Perturbation for generation.
    """
    B, C, H, W = img.shape
    device = img.device
    dtype = img.dtype

    # ---- 1) Random Horizontal Flip -----------------------------------
    if p_flip > 0:
        flip_mask = (torch.rand(B, device=device) < p_flip).view(-1, 1, 1, 1)
        img = torch.where(flip_mask, img.flip(-1), img)

    # ---- 2) Random Affine -------------------
    deg = (torch.rand(B, device=device) * 2 - 1) * max_deg
    theta = deg * math.pi / 180.
    scale = 1.0 + (torch.rand(B, device=device) * 0.1 - 0.05)  # ±5%
    tx = (torch.rand(B, device=device) * 0.1 - 0.05)           # ±5% Shift
    ty = (torch.rand(B, device=device) * 0.1 - 0.05)

    rot = torch.stack([torch.cos(theta), -torch.sin(theta), torch.sin(theta),  torch.cos(theta)], dim=1).view(B, 2, 2)
    sc = rot * scale.view(-1, 1, 1)
    trans = torch.stack([tx, ty], dim=1).view(B, 2, 1)

    affine = torch.cat([sc, trans], dim=2)
    grid = F.affine_grid(affine, img.size(), align_corners=False)
    img = F.grid_sample(img, grid, align_corners=False)

    # ---- 3) Color Jitter --------------------
    # Bright & Contrast
    if jitter_strength > 0:
        bright = 1 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 2 * jitter_strength
        contrast = 1 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 2 * jitter_strength
        img = (img * contrast) * bright

        gray = img.mean(dim=1, keepdim=True)
        sat = 1 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 2 * jitter_strength
        img = img * sat + gray * (1 - sat)

    # ---- 4) Random Gaussian Blur -------------------------------------
    if p_blur > 0:
        blur_mask = torch.rand(B, device=device) < p_blur
        if blur_mask.any():
            # 3×3 Gaussian Kernel [1,2,1]/16
            kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=dtype, device=device) / 16.0
            kernel = kernel.expand(C, 1, 3, 3)
            img_blur = F.conv2d(img, kernel, padding=1, groups=C)
            img = torch.where(blur_mask.view(-1, 1, 1, 1), img_blur, img)

    # ---- 5) Random Erasing --------------------------------
    if p_erase > 0:
        erase_mask = torch.rand(B, device=device) < p_erase
        if erase_mask.any():
            erase_size = int(0.2 * H)  # 20% side length
            for i in torch.where(erase_mask)[0]:
                cx = torch.randint(0, H, (1,), device=device).item()
                cy = torch.randint(0, W, (1,), device=device).item()
                x1 = max(cx - erase_size // 2, 0)
                y1 = max(cy - erase_size // 2, 0)
                x2 = min(x1 + erase_size, H)
                y2 = min(y1 + erase_size, W)
                img[i, :, x1:x2, y1:y2] = 0.0        # Random Erasing

    return img


def total_variation(img):
    B, C, H, W = img.shape
    dh = img[:, :, :, 1:] - img[:, :, :, :-1]
    dv = img[:, :, 1:, :] - img[:, :, :-1, :]

    N_h = C * H * (W - 1)
    N_v = C * (H - 1) * W

    loss_h = dh.pow(2).sum(dim=[1, 2, 3]) / N_h
    loss_v = dv.pow(2).sum(dim=[1, 2, 3]) / N_v
    loss_per_image = loss_h + loss_v

    return loss_per_image.mean()


def train_fake_img_d4c(model, cali_data, gen_text, d4c_config, gen_iter, gen_batch_size, gen_lr, device, tau=0.1, lambda_nce=1.0):
    B, C, H, W = cali_data.shape
    cali_data = cali_data.to(device)
    gen_text = gen_text.to(device)

    cali_data.requires_grad = True
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    with torch.no_grad():
        text_features = model.encode_text(gen_text.to(device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Generate pseudo label for generation
    num_classes = gen_text.shape[0]
    target_labels_all = torch.randint(0, num_classes, (B,), device=device)

    optimizer = torch.optim.Adam([cali_data], lr=gen_lr)

    if d4c_config in (1, 3):
        # Initialize foreground crop coordinates
        crop_params_all = random_crop_params(B, H, W, min_ratio=0.4, max_ratio=0.7, device=device)
        # Generate masks
        mask = torch.zeros(B, 1, H, W, device=device)
        for i, (top, left, h_, w_) in enumerate(crop_params_all):
            mask[i, :, top:top + h_, left:left + w_] = 1.0
        mask = mask.expand(-1, C, -1, -1)  # (B, C, H, W)

    for it in range(gen_iter):
        for start in range(0, B, gen_batch_size):
            end = min(B, start + gen_batch_size)
            batch = cali_data[start:end]
            optimizer.zero_grad()

            if d4c_config in (1, 3):
                # Crop and resize foreground to 224x224
                fg = crop_and_resize(batch, crop_params_all[start:end], out_size=H)
                if d4c_config == 3:
                    fg = gen_perturb(fg)
                # Separate background
                m = mask[start:end]
                bg = batch * (1 - m) + torch.randn_like(batch) * 0.1

                # Encode feature and normalize
                fg_feat = model.encode_image(fg)
                fg_feat = fg_feat / fg_feat.norm(dim=-1, keepdim=True)
                bg_feat = model.encode_image(bg)
                bg_feat = bg_feat / bg_feat.norm(dim=-1, keepdim=True)

                # Method 1 and Method 2
                fg_logits = fg_feat @ text_features.t() / tau
                bg_logits = fg_feat @ bg_feat.t() / tau
                logits = torch.cat([fg_logits, bg_logits], dim=1)

            elif d4c_config in (0, 2):
                if d4c_config == 2:
                    batch = gen_perturb(batch)
                feat = model.encode_image(batch)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                logits = feat @ text_features.t() / tau

            else:
                raise NotImplementedError("Specify d4c_config from {0, 1, 2, 3} to generate images.")

            target_labels = target_labels_all[start:end]
            loss_nce = F.cross_entropy(logits, target_labels)
            tv_loss = total_variation(batch)
            loss = lambda_nce * loss_nce + 0.1 * tv_loss
            loss.backward()
            optimizer.step()

        if it % 10 == 0:
            print(f"[Iter {it}/{gen_iter}] loss={loss.item():.4f}")


def train_bn_loss(model, cali_data, gen_iter, gen_batch_size, gen_lr, device):
    def bn_hook(module, input, output):
        x = input[0]
        dims = [0] + list(range(2, x.ndim))
        batch_mean = x.mean(dim=dims)
        batch_var = x.var(dim=dims, unbiased=False)
        loss_bn = ((batch_mean - module.running_mean) ** 2).mean() + ((batch_var - module.running_var) ** 2).mean()
        bn_loss_values.append(loss_bn)

    B, C, H, W = cali_data.shape
    cali_data = cali_data.to(device)

    bn_loss_values = []
    bn_hooks = []

    cali_data.requires_grad = True
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            h = module.register_forward_hook(bn_hook)
            bn_hooks.append(h)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    optimizer = torch.optim.Adam([cali_data], lr=gen_lr)

    for it in range(gen_iter):
        for start in range(0, B, gen_batch_size):
            end = min(B, start + gen_batch_size)
            batch = cali_data[start:end]
            optimizer.zero_grad()

            bn_loss_values.clear()
            model.encode_image(batch)
            bn_loss = sum(bn_loss_values) if bn_loss_values else torch.tensor(0., device=device)
            tv_loss = total_variation(batch)

            loss = bn_loss + 0.1 * tv_loss
            loss.backward()
            optimizer.step()

        if it % 10 == 0:
            print(f"[Iter {it}/{gen_iter}] loss={loss.item():.4f}")

    for h in bn_hooks:
        h.remove()


def train_pse_loss(model, cali_data, gen_iter, gen_batch_size, gen_lr, device):
    B, C, H, W = cali_data.shape
    cali_data = cali_data.to(device)

    attn_maps = []

    cali_data.requires_grad = True
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    for _, m in model.visual.named_modules():
        if type(m) is nn.MultiheadAttention:
            orig_forward = m.forward

            def patched_forward(self, query, key, value,
                                key_padding_mask=None,
                                need_weights=False,
                                attn_mask=None,
                                average_attn_weights=True,
                                _orig=orig_forward):
                attn_output, attn_weights = _orig(query, key, value,
                                                  key_padding_mask,
                                                  True,
                                                  attn_mask,
                                                  False)
                v = value.permute(1, 0, 2)
                context = torch.einsum('bhij,bje->bhie', attn_weights, v)
                attn_maps.append(context)
                return attn_output, attn_weights

            m.forward = patched_forward.__get__(m, m.__class__)

    optimizer = torch.optim.Adam([cali_data], lr=gen_lr)

    for it in range(gen_iter):
        for start in range(0, B, gen_batch_size):
            end = min(B, start + gen_batch_size)
            batch = cali_data[start:end]
            optimizer.zero_grad()

            attn_maps.clear()
            model.encode_image(batch)
            entropy_loss = 0.0
            if attn_maps:
                for ctx in attn_maps:
                    attention_p = ctx.mean(dim=1)[:, 1:, :]
                    sims = torch.cosine_similarity(attention_p.unsqueeze(1), attention_p.unsqueeze(2), dim=3)

                    kde = KernelDensityEstimator(sims.view(gen_batch_size, -1))
                    start_p = sims.min().item()
                    end_p = sims.max().item()
                    x_plot = torch.linspace(start_p, end_p, steps=10).repeat(gen_batch_size, 1).cuda()
                    kde_estimate = kde(x_plot)
                    dif_entropy_estimated = differential_entropy(kde_estimate, x_plot)
                    entropy_loss -= dif_entropy_estimated
            tv_loss = total_variation(batch)

            loss = entropy_loss + 0.1 * tv_loss
            loss.backward()
            optimizer.step()

        if it % 10 == 0:
            print(f"[Iter {it}/{gen_iter}] loss={loss.item():.4f}")


def train_fake_img_baseline(model, cali_data, is_transformer, gen_iter, gen_batch_size, gen_lr, device):
    if is_transformer:
        train_pse_loss(model, cali_data, gen_iter, gen_batch_size, gen_lr, device)
    else:
        train_bn_loss(model, cali_data, gen_iter, gen_batch_size, gen_lr, device)
