import torch
import torch.nn as nn


# ============================================================
# Patch Embedding
# ============================================================

class PatchEmbedding(nn.Module):

    def __init__(
        self,
        image_size=32,
        patch_size=4,
        in_channels=1,
        embed_dim=128,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size

        # Number of patches:
        #
        # 32 / 4 = 8
        # 8 × 8 = 64 patches

        self.num_patches = (
            image_size // patch_size
        ) ** 2

        # Convert image patches into embeddings

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):

        # Input:
        # [B, C, H, W]

        x = self.projection(x)

        # [B, D, H/P, W/P]
        #
        # [4, 128, 8, 8]

        x = x.flatten(2)

        # [B, D, N]
        #
        # [4, 128, 64]

        x = x.transpose(1, 2)

        # [B, N, D]
        #
        # [4, 64, 128]

        return x


# ============================================================
# Test Patch Embedding
# ============================================================

if __name__ == "__main__":

    model = PatchEmbedding(
        image_size=32,
        patch_size=4,
        in_channels=1,
        embed_dim=128,
    )

    x = torch.randn(
        4,
        1,
        32,
        32,
    )

    output = model(x)

    print("=" * 60)
    print("PATCH EMBEDDING TEST")
    print("=" * 60)

    print("Input shape:", x.shape)

    print(
        "Output shape:",
        output.shape,
    )

    print(
        "Number of patches:",
        model.num_patches,
    )