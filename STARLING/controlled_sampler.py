"""
Controlled DDPM sampler for STARLING (idptools/starling).

Implements the reduced two-parameter noise controller

    nu(t) = alpha * g(t)   on   t in [0, t_w],

from "Inference-time noise control of diffusion models" (Behjoo et al.).
In a DDPM codebase the controlled reverse step is a two-line modification
(SI Appendix, app:ddpm of the paper):

    1. score coefficient:  beta_t  ->  (beta_t + nu_t^2) / 2
    2. injected noise std: sqrt(beta_t)  ->  nu_t

Time convention: t is FORWARD (noising) time. The sampling loop runs
timestep = T-1 ... 0, so the low-noise (late-reverse) window [0, t_w]
corresponds to DDPM indices  timestep/T <= t_w , i.e. the LAST fraction
t_w of sampling steps. The theory localizes productive control action to
this window (kernel envelope ~ 1/Sigma_t^4).

Reference-schedule note: the paper's reference is nu = g, i.e. the
sigma_t^2 = beta_t variance choice of DDPM. STARLING's stock sampler
uses the exact posterior variance beta_tilde_t instead (the two choices
agree to O(beta_t); see Ho et al. 2020 and the SI). This class therefore
uses the continuous-time reference g(t) = sqrt(beta_t) on control steps
-- which also avoids the degenerate beta_tilde -> 0 limit at the last
steps -- and leaves all out-of-window steps EXACTLY as STARLING's stock
sampler computes them. To get a pure baseline for comparison, use
starling.samplers.ddpm_sampler.DDPMSampler directly (or set window to a
value below 1/T, which puts every step out of window).
"""

import torch

from starling.samplers.ddpm_sampler import DDPMSampler, extract


class ControlledDDPMSampler(DDPMSampler):
    """DDPM sampler with the reduced inference-time noise controller.

    Parameters
    ----------
    ddpm_model : starling DiffusionModel
        The pretrained latent diffusion model (frozen, unchanged).
    encoder_model : starling VAE wrapper
        Used only for decoding latents to distance maps (unchanged).
    alpha : float
        Noise amplification factor. alpha > 1 boosts injected noise
        (over-sharp-score regime; the default for protein systems);
        0 < alpha < 1 damps it (over-smoothed regime); alpha = 1
        reproduces the continuous-time reference sampler nu = g.
    window : float
        Control window t_w as a fraction of forward time, in (0, 1].
        The controller acts on steps with timestep/T <= window, i.e.
        the late-reverse (low-noise) part of sampling. Validated
        protein default: 0.2.
    nu_min : float
        Admissible floor for nu_t (safety clip; default 0.0).
    nu_schedule : torch.Tensor, optional
        Full per-timestep schedule with num_timesteps entries. If
        given, overrides (alpha, window): every step uses
        nu_t = nu_schedule[timestep] for the noise std and
        (beta_t + nu_t^2)/2 for the score coefficient. Use this to
        plug in a schedule produced by the exact fixed-point solver
        of the paper.
    ionic_strength : float
        Passed through to the conditioning labels (default 150 mM).
    """

    def __init__(
        self,
        ddpm_model,
        encoder_model,
        alpha: float = 2.5,
        window: float = 0.2,
        nu_min: float = 0.0,
        nu_schedule: torch.Tensor = None,
        ionic_strength: float = 150,
    ):
        super().__init__(
            ddpm_model, encoder_model, ionic_strength=ionic_strength
        )
        if not 0.0 < window <= 1.0:
            raise ValueError(f"window must be in (0, 1], got {window}")
        if alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.alpha = float(alpha)
        self.window = float(window)
        self.nu_min = float(nu_min)
        if nu_schedule is not None:
            nu_schedule = torch.as_tensor(
                nu_schedule, dtype=torch.float32, device=self.device
            )
            if nu_schedule.numel() != self.n_steps:
                raise ValueError(
                    f"nu_schedule must have {self.n_steps} entries, "
                    f"got {nu_schedule.numel()}"
                )
        self.nu_schedule = nu_schedule

    def _control_params(self, batched_timestamps, shape):
        """Return (score_coeff, noise_std) for the controlled update.

        Broadcastable to `shape`. Out-of-window steps get the stock
        STARLING values (score coefficient beta_t, noise std
        sqrt(beta_tilde_t)); in-window steps get the controlled values
        ((beta_t + nu_t^2)/2, nu_t) with nu_t = alpha * sqrt(beta_t).
        """
        betas_t = extract(self.betas, batched_timestamps, shape)
        posterior_std = torch.sqrt(
            extract(self.posterior_variance, batched_timestamps, shape)
        )

        if self.nu_schedule is not None:
            nu = extract(self.nu_schedule, batched_timestamps, shape)
            nu = torch.clamp(nu, min=self.nu_min)
            return 0.5 * (betas_t + nu**2), nu

        nu = torch.clamp(self.alpha * torch.sqrt(betas_t), min=self.nu_min)

        in_window = (
            batched_timestamps.float() / float(self.n_steps) <= self.window
        ).reshape(batched_timestamps.shape[0], *((1,) * (len(shape) - 1)))

        score_coeff = torch.where(in_window, 0.5 * (betas_t + nu**2), betas_t)
        noise_std = torch.where(in_window, nu, posterior_std)
        return score_coeff, noise_std

    def p_sample(
        self,
        x: torch.Tensor,
        timestamp: int,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """One controlled denoising step (two-line modification).

        Identical to DDPMSampler.p_sample except:
          * score coefficient beta_t -> (beta_t + nu_t^2) / 2
          * injected noise std sqrt(beta_tilde_t) -> nu_t
        on control-window steps. Out-of-window steps are bit-identical
        to the stock sampler.
        """
        b, *_, device = *x.shape, x.device

        batched_timestamps = torch.full(
            (b,), timestamp, device=device, dtype=torch.long
        )

        preds = self.ddpm_model.model(x, batched_timestamps, labels, attention_mask)

        sqrt_recip_alphas_t = extract(
            self.sqrt_recip_alphas, batched_timestamps, x.shape
        )
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, batched_timestamps, x.shape
        )

        # --- line 1 of the modification: beta_t -> (beta_t + nu_t^2)/2 ---
        score_coeff, noise_std = self._control_params(batched_timestamps, x.shape)

        predicted_mean = sqrt_recip_alphas_t * (
            x - score_coeff * preds / sqrt_one_minus_alphas_cumprod_t
        )

        if timestamp == 0:
            return predicted_mean
        else:
            # --- line 2 of the modification: noise std -> nu_t ---
            noise = torch.randn_like(x)
            return predicted_mean + noise_std * noise
