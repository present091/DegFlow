import torch
import torch.nn.functional as F

def linear_interpolate(control_points, interp_factor, control_positions):
    """
    Perform piecewise linear interpolation and compute derivatives for a sequence of images.

    Args:
        control_points: Tensor of shape (N, B, C, H, W) representing control images.
        interp_factor: Tensor of shape (B,) with values in [0, 1] representing the interpolation factor.
        control_positions: Tensor of shape (N,) with increasing positions corresponding to control_points.

    Returns:
        interp:        Tensor of shape (B, C, H, W), the interpolated images.
        deriv:         Tensor of shape (B, C, H, W), the first derivative (piecewise constant).
        second_deriv:  Tensor of shape (B, C, H, W), the second derivative (zeros).
        third_deriv:   Tensor of shape (B, C, H, W), the third derivative (zeros).
    """
    N, B = control_points.shape[0], interp_factor.shape[0]
    device = control_points.device
    dtype = control_points.dtype

    # ensure control_positions on same device
    control_positions = control_positions.to(device)

    # find segment index for each batch
    i = torch.searchsorted(control_positions, interp_factor, right=True) - 1
    i = torch.clamp(i, 0, N - 2)

    # positions and segment length
    pos_i   = control_positions[i]       # (B,)
    pos_ip1 = control_positions[i+1]     # (B,)
    h       = pos_ip1 - pos_i            # (B,)

    # local interpolation weight
    alpha = (interp_factor - pos_i) / h  # (B,)

    # reshape for broadcasting
    h_b      = h.view(B, 1, 1, 1)
    alpha_b  = alpha.view(B, 1, 1, 1)

    # extract batch-wise endpoints
    batch_idx = torch.arange(B, device=device)
    P0 = control_points[i,       batch_idx]  # (B, C, H, W)
    P1 = control_points[i + 1,   batch_idx]  # (B, C, H, W)

    # piecewise linear interpolation
    interp = (1 - alpha_b) * P0 + alpha_b * P1

    # first derivative is constant on each interval
    deriv = (P1 - P0) / h_b

    # second and third derivatives are zero everywhere
    second_deriv = torch.zeros_like(interp)
    third_deriv  = torch.zeros_like(interp)

    return interp, deriv, second_deriv, third_deriv

def natural_cubic_spline_interpolate(control_points, interp_factor, control_positions):
    """
    Perform natural cubic spline interpolation and compute 1st, 2nd, 3rd derivatives.

    Returns:
        interp:        (B, C, H, W)
        deriv:         (B, C, H, W)
        second_deriv:  (B, C, H, W)
        third_deriv:   (B, C, H, W) - constant over each interval
    """
    N, B = control_points.shape[0], interp_factor.shape[0]
    device = control_points.device
    dtype  = control_points.dtype

    # ensure control_positions on same device
    control_positions = control_positions.to(device)

    # 구간 index 선택
    i = torch.searchsorted(control_positions, interp_factor, right=True) - 1
    i = torch.clamp(i, 0, N - 2)

    pos_i   = control_positions[i]       # (B,)
    pos_ip1 = control_positions[i+1]     # (B,)
    h       = pos_ip1 - pos_i            # (B,)
    alpha   = (interp_factor - pos_i) / h # (B,)

    # 2차 도함수 M 계산 (out-of-place)
    if N > 2:
        # compute tridiagonal coefficients
        h_all   = control_positions[1:] - control_positions[:-1]  # (N-1,)
        n       = N - 2
        a = h_all[:-1]                                            # (n,)
        b = 2 * (h_all[:-1] + h_all[1:])                           # (n,)
        c = h_all[1:]                                             # (n,)

        # right-hand side d_j
        d_list = []
        for j in range(n):
            idx = j + 1
            term = 6 * (
                (control_points[idx+1] - control_points[idx]) / h_all[idx] -
                (control_points[idx]   - control_points[idx-1]) / h_all[idx-1]
            )
            d_list.append(term)  # each term is shape (B,C,H,W)
        # solve for c_prime, d_prime via forward elimination
        c_prime_list = []
        d_prime_list = []
        # j=0
        c_prime_list.append(c[0] / b[0])       # scalar / scalar
        d_prime_list.append(d_list[0] / b[0])  # tensor / scalar
        # forward
        for j in range(1, n):
            prev_c = c_prime_list[j-1]
            prev_d = d_prime_list[j-1]
            denom  = b[j] - a[j] * prev_c        # scalar
            c_prime_list.append(c[j] / denom)
            d_prime_list.append((d_list[j] - a[j] * prev_d) / denom)
        # backward to get M_internal
        m_internal = [None] * n
        m_internal[-1] = d_prime_list[-1]
        for j in range(n-2, -1, -1):
            m_internal[j] = d_prime_list[j] - c_prime_list[j] * m_internal[j+1]
        m_internal_tensor = torch.stack(m_internal, dim=0)  # (n, B, C, H, W)

        # build full M: zeros at endpoints + internal
        zeros = torch.zeros((1, B, *control_points.shape[2:]), device=device, dtype=dtype)
        M = torch.cat([zeros, m_internal_tensor, zeros], dim=0)  # (N, B, C, H, W)
    else:
        # if only two points, all second derivs = 0
        M = torch.zeros_like(control_points)

    # 각 배치에 대한 P0, P1, M0, M1 추출
    batch_idx = torch.arange(B, device=device)
    P0 = control_points[i,         batch_idx]
    P1 = control_points[i + 1,     batch_idx]
    M0 = M[i,                       batch_idx]
    M1 = M[i + 1,                   batch_idx]

    # reshape for broadcast
    h_b = h.view(B, 1, 1, 1)
    A   = ((pos_ip1 - interp_factor) / h).view(B, 1, 1, 1)
    Bp  = alpha.view(B, 1, 1, 1)

    # 0차 보간 (위치)
    interp = (
        A * P0 +
        Bp * P1 +
        ((A**3 - A) * M0 + (Bp**3 - Bp) * M1) * (h_b**2) / 6.0
    )

    # 1차 도함수
    deriv = (
        (P1 - P0) / h_b +
        (h_b / 6.0) * ((-3 * A**2 + 1) * M0 + (3 * Bp**2 - 1) * M1)
    )

    # 2차 도함수
    second_deriv = A * M0 + Bp * M1

    # 3차 도함수 (상수)
    third_deriv = (M1 - M0) / h_b

    return interp, deriv, second_deriv, third_deriv

