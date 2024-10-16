import torch
import torch.nn as nn

from neuralop.models import FNO
from neuralop.models.tfno import Projection
from neuralop.models.spectral_convolution import FactorizedSpectralConv


class TransportFNO(FNO):
    def __init__(
        self,
        n_modes,
        hidden_channels,
        in_channels=4,
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        incremental_n_modes=None,
        use_mlp=False,
        mlp=None,
        non_linearity=torch.nn.functional.gelu,
        norm=None,
        preactivation=False,
        fno_skip="linear",
        mlp_skip="soft-gating",
        separable=False,
        factorization=None,
        rank=1,
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="one-sided",
        fft_norm="forward",
        SpectralConv=FactorizedSpectralConv,
        **kwargs
    ):
        FNO.__init__(
            self,
            n_modes,
            hidden_channels,
            in_channels,
            out_channels,
            lifting_channels,
            projection_channels,
            n_layers,
            incremental_n_modes,
            use_mlp,
            mlp,
            non_linearity,
            norm,
            preactivation,
            fno_skip,
            mlp_skip,
            separable,
            factorization,
            rank,
            joint_factorization,
            fixed_rank_modes,
            implementation,
            decomposition_kwargs,
            domain_padding,
            domain_padding_mode,
            fft_norm,
            SpectralConv,
            **kwargs
        )

    
        self.projection = Projection(
                    in_channels=self.hidden_channels,
                    out_channels=out_channels,
                    hidden_channels=projection_channels,
                    non_linearity=non_linearity,
                    n_dim=1,
                )

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device

    def forward(self, transports, couplings):
        """TFNO's forward pass"""
        transports = self.lifting(transports)

        if self.domain_padding is not None:
            transports = self.domain_padding.pad(transports)

        for layer_idx in range(self.n_layers):
            transports = self.fno_blocks(transports, layer_idx)

        if self.domain_padding is not None:
            transports = self.domain_padding.unpad(transports)


        couplings = couplings[0].to(transports.device).permute(1,0)
        transports = transports.reshape(self.hidden_channels, -1).permute(1,0)
        out = torch.sparse.mm(couplings, transports).permute(1,0)

        out = out.unsqueeze(0)
        out = self.projection(out).squeeze(1)
        return out
