import torch

def precompute_theta_pos_frequencies(
    head_dim: int, seq_len: int, device: str, theta: float = 10000.0, 
):
    assert head_dim % 2 == 0, "Dimensions must be divisible by 2"

    # Buil the theta parameters
    # Formula: theta_i = 10000 ^ (-2(i-1)/dim) for i = [1, 2, ... dim / 2]
    
    # Shape: (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    
    # Shape: (head_dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    
    # Construct the positions (the "m" parameter)
    # Shape: (seq_len)
    m = torch.arange(seq_len, device=device)

    # Multiply each theta by each position using the outer product
    # Shape: (seq_len) outer_product * (head_dim / 2) -> (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float() 

    # We have to compute complex numbers in the polar form c = R * exp(1 * m * theta), where R = 1 as:
    # Shape: (seq_len, head_dim / 2) -> (seq_len, head_dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    
    return freqs_complex

def apply_rotary_embedings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (B, seq_len, H, head_dim) -> (B, seq_len, H, head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    
    # (seq_len, head_dim / 2) -> (1, seq_len, 1, head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # (B, seq_len, H, head_dim / 2) * (1, seq_len, 1, head_dim / 2) = (B, seq_len, H, head_dim / 2)
    x_rotated = x_complex * freqs_complex

    # (B, seq_len, H, head / 2) -> (B, seq_len, H, head_dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)

    # (B, seq_len, H, head_dim / 2, 2) -> (B, seq_len, H, head_dim)
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x).to(device)