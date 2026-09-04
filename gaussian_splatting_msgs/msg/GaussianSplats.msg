# This message holds a collection of N Gaussian splatting primitives, either
# 3D Gaussians (3DGS) or 2D oriented discs (2DGS), stored as parallel arrays.
#
# Positions, scales and orientations are expressed in the coordinate frame
# given by header.frame_id, following REP-103 (right-handed, meters).
#
# Parameters are stored in their definitional form, not in the parameterization
# used by the optimizers that produce them: scales are linear standard
# deviations, opacities are linear, and quaternions are normalized. PLY files
# written by the reference implementations store log-scales, logit-opacities
# and unnormalized quaternions, so the corresponding activations have to be
# applied when converting.

# Time of acquisition, and the coordinate frame ID.
std_msgs/Header header

# Primitive type constants
uint8 TYPE_3DGS=0                   # 3D Gaussian ellipsoid
uint8 TYPE_2DGS=1                   # 2D oriented disc (surfel)

uint8 type

# Rasterization mode constants. This describes how the stored opacities were
# produced and is therefore a property of the data, not a viewer preference.
# Data from Mip-Splatting should be converted from its save_fused_ply() export.
uint8 RASTERIZE_MODE_CLASSIC=0      # Screen-space dilation only
uint8 RASTERIZE_MODE_ANTIALIASED=1  # Dilation with opacity compensation,
                                    # rho = sqrt(det(S) / det(S + eps2d*I))

uint8 rasterize_mode

# Screen-space low-pass filter in px^2, added to the diagonal of the projected
# 2D covariance. 0.3 is the value used by the reference rasterizer.
float32 eps2d 0.3

# Degree of the spherical harmonics, giving K = (sh_degree+1)^2 coefficients per
# color channel. Color along view direction d is evaluated as
# sum_k SH_k(d)*c_k + 0.5, where the degree 0 basis SH_0 is 0.28209479177387814.
uint8 sh_degree

# The primitive count N is len(means)/3. The remaining arrays are sized
# relative to it:
#   scales     3*N, or 2*N when type is TYPE_2DGS
#   quats      4*N
#   opacities    N
#   sh_dc      3*N
#   sh_rest    3*N*(K-1)

# Gaussian centers, xyz interleaved.
float32[] means

# Linear standard deviations in meters. Column i of the rotation matrix built
# from quats is scaled by scales[i]. TYPE_2DGS stores two of them; the normal of
# the disc is column 2 of that rotation matrix.
float32[] scales

# Normalized orientation quaternions, in ROS component order (x, y, z, w).
# PLY files store rot_0..rot_3 as (w, x, y, z) and leave them unnormalized.
float32[] quats

# Linear opacity in the range [0, 1].
float32[] opacities

# Degree 0 (DC) spherical harmonics coefficient, indexed as sh_dc[i*3 + c].
# These are raw SH coefficients rather than RGB values.
float32[] sh_dc

# Higher degree spherical harmonics coefficients, coefficient-major, indexed as
# sh_rest[(i*(K-1) + k)*3 + c]. PLY files store f_rest_* channel-major, so the
# two inner dimensions have to be transposed. Empty when sh_degree is 0.
float32[] sh_rest
