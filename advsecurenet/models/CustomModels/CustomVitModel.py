"""
This is the a simple ViT (Vision Transformer) implementation in PyTorch. It is based on the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al. (https://arxiv.org/abs/2010.11929).
The model architecture is taken from great blog post by Brian Pulfer (https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c)

Further improvements have been made to the model, including vectorization of the multi-head self-attention mechanism, and patchification of the input images.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomVitModel(nn.Module):
    # TODO: Better align with the existing BaseModel class
    def __init__(self, chw=(3, 32, 32), n_patches=8, n_blocks=2, hidden_d=8, n_heads=2, num_classes=10, num_input_channels=3, **kwargs):
        """
        Args:
            chw (tuple): ( C , H , W ) shape of the input image
            n_patches (int): Number of patches to divide the image into
            n_blocks (int): Number of transformer blocks
            hidden_d (int): Hidden dimension
            n_heads (int): Number of heads in multi-head self-attention
            num_classes (int): Number of classes in the dataset
            num_input_channels (int): Number of input channels in the dataset

        Examples:
        >>> model = ModelFactory.create_model("CustomVitModel", num_classes=10, num_input_channels=1, chw=(1, 28, 28))
        """

        # Super constructor
        super(CustomVitModel, self).__init__()

        # Attributes
        self.chw = chw  # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        self.num_classes = num_classes
        self.num_input_channels = num_input_channels
        self.kwargs = kwargs

        
        # Input and patches sizes
        assert (
            chw[1] % n_patches == 0
        ), "Input shape not entirely divisible by number of patches"
        assert (
            chw[2] % n_patches == 0
        ), "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(self.num_input_channels *
                           self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3) Positional embedding
        self.register_buffer(
            "positional_embeddings",
            self.get_positional_embeddings(n_patches**2 + 1, hidden_d),
            persistent=False,
        )

        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList(
            [MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)]
        )

        # 5) Classification MLPk
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, self.num_classes), nn.Softmax(dim=-1))

    def forward(self, images):
        # Dividing images into patches
        n, c, h, w = images.shape
        patches = self.patchify(images, self.n_patches).to(
            self.positional_embeddings.device)

        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Getting the classification token only
        out = out[:, 0]

        # Map to output dimension, output category distribution
        return self.mlp(out)

    def patchify(self, images, n_patches):
        n, c, h, w = images.shape
        assert h == w, "Patchify method is implemented for square images only"

        # Calculate patch size
        patch_size = h // n_patches

        # Reshape and permute to bring patches into separate dimensions
        patches = images.view(n, c, n_patches, patch_size,
                              n_patches, patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5)

        # Flatten patches
        patches = patches.contiguous().view(n, n_patches**2, -1)

        return patches

    def get_positional_embeddings(self, sequence_length, d):
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                result[i][j] = (
                    np.sin(i / (10000 ** (j / d)))
                    if j % 2 == 0
                    else np.cos(i / (10000 ** ((j - 1) / d)))
                )
        return result


class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.k_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.v_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        N, seq_length, token_dim = sequences.shape

        # Reshape sequences to separate heads
        # New shape: (N, seq_length, n_heads, token_dim / n_heads)
        sequences = sequences.view(N, seq_length, self.n_heads, -1)

        # Transpose to bring heads next to N for easier processing
        # Shape: (N, n_heads, seq_length, token_dim / n_heads)
        sequences = sequences.transpose(1, 2)

        # Initialize lists to store results
        qs, ks, vs = [], [], []

        # Apply the mappings to each head in parallel
        for head in range(self.n_heads):
            q_mapping = self.q_mappings[head]
            k_mapping = self.k_mappings[head]
            v_mapping = self.v_mappings[head]

            qs.append(q_mapping(sequences[:, head]))
            ks.append(k_mapping(sequences[:, head]))
            vs.append(v_mapping(sequences[:, head]))

        # Stack results for query, key, and value
        qs, ks, vs = torch.stack(qs, dim=1), torch.stack(
            ks, dim=1), torch.stack(vs, dim=1)

        # Compute attention in parallel for all heads
        attention_scores = qs @ ks.transpose(-2, -1) / (self.d_head ** 0.5)
        attention_scores = F.softmax(attention_scores, dim=-1)

        # Apply attention to values
        # Shape: (N, n_heads, seq_length, token_dim / n_heads)
        seq_result = attention_scores @ vs

        # Reshape and concatenate heads
        seq_result = seq_result.transpose(
            1, 2).contiguous().view(N, seq_length, -1)

        return seq_result


class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d),
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out



