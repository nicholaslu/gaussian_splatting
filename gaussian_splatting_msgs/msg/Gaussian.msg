# Gaussian.msg
geometry_msgs/Point32 position
std_msgs/ColorRGBA color

# 3D covariance matrix:
#  xx xy xz
#  xy yy yz
#  xz yz zz
float32 cov_xx
float32 cov_yy
float32 cov_zz
float32 cov_xy
float32 cov_xz
float32 cov_yz