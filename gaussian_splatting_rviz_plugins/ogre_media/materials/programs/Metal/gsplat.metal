// Metal counterpart of ../GLSL/gsplat.{vert,frag}. Ogre's Metal attribute
// mapping puts VES_POSITION at 0, VES_DIFFUSE at 3 and TEXCOORD n at 8 + n
// (see MetalProgram::getAttributeIndex).

struct GaussianSplatIn
{
    float3 position [[attribute(0)]];
    float4 color    [[attribute(3)]];
    float3 scale    [[attribute(8)]];   // linear sigma in metres
    float4 quat     [[attribute(9)]];   // unit quaternion, ROS order (x,y,z,w)
    float2 corner   [[attribute(10)]];  // quad corner in [-1, 1]
};

struct GaussianSplatOut
{
    float4 position  [[position]];
    float4 color;
    float2 localCoord;
};

// Ogre reflects this struct (MetalProgram::analyzeParameterBuffer) and binds
// each member by NAME at the offset the Metal compiler assigned, so the names
// here must match the param_named entries in
// ../../scriptsMetal/gaussiansplat.material. Declaration order does not have
// to match; a name that does not match is silently never written.
struct GaussianSplatUniforms
{
    metal::float4x4 projmatrix;
    metal::float4x4 viewmatrix;
    float4 vpsize;
    float fovy;
    float eps2d;
    float antialiased;
    // Extent of the rasterised quad, in standard deviations. Fill rate scales
    // with the square of this value, so lowering it is the cheapest
    // accuracy/speed trade available; 3.0 matches the reference rasteriser.
    float sigma_radius;
};

// Sigma = M * M^T with M = R * diag(s), i.e. column i of the rotation matrix
// scaled by s[i]. float3x3() takes columns.
static metal::float3x3 computeCov3D(float3 s, float4 q)
{
    const float x = q.x;
    const float y = q.y;
    const float z = q.z;
    const float w = q.w;

    const metal::float3x3 m = metal::float3x3(
        float3(1.0 - 2.0 * (y * y + z * z),
               2.0 * (x * y + w * z),
               2.0 * (x * z - w * y)) * s.x,
        float3(2.0 * (x * y - w * z),
               1.0 - 2.0 * (x * x + z * z),
               2.0 * (y * z + w * x)) * s.y,
        float3(2.0 * (x * z + w * y),
               2.0 * (y * z - w * x),
               1.0 - 2.0 * (x * x + y * y)) * s.z);

    return m * metal::transpose(m);
}

static float3 computeCov2D(
    float3 position, metal::float3x3 Vrk,
    constant GaussianSplatUniforms & u, thread float & compensation)
{
    const float tanFovy = metal::tan(u.fovy * 0.5);
    const float tanFovx = tanFovy * u.vpsize.x / u.vpsize.y;
    const float focalY = u.vpsize.y / (2.0 * tanFovy);
    const float focalX = u.vpsize.x / (2.0 * tanFovx);

    float4 t4 = u.viewmatrix * float4(position, 1.0);
    float3 t = t4.xyz;
    const float limx = 1.3 * tanFovx;
    const float limy = 1.3 * tanFovy;
    t.x = metal::clamp(t.x / t.z, -limx, limx) * t.z;
    t.y = metal::clamp(t.y / t.z, -limy, limy) * t.z;

    const metal::float3x3 J = metal::float3x3(
        float3(focalX / t.z, 0.0, -(focalX * t.x) / (t.z * t.z)),
        float3(0.0, focalY / t.z, -(focalY * t.y) / (t.z * t.z)),
        float3(0.0));
    const metal::float3x3 view3 = metal::float3x3(
        u.viewmatrix[0].xyz, u.viewmatrix[1].xyz, u.viewmatrix[2].xyz);
    const metal::float3x3 T = metal::transpose(view3) * J;
    metal::float3x3 cov = metal::transpose(T) * Vrk * T;

    const float detOrig = cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1];
    cov[0][0] += u.eps2d;
    cov[1][1] += u.eps2d;
    const float detBlur = cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1];
    compensation = metal::sqrt(metal::max(detOrig / detBlur, 0.0));
    return float3(cov[0][0], cov[0][1], cov[1][1]);
}

vertex GaussianSplatOut gsplat_vp(
    GaussianSplatIn in [[stage_in]],
    constant GaussianSplatUniforms & u [[buffer(CONST_SLOT_START)]])
{
    GaussianSplatOut out;
    const float4 clip = u.projmatrix * float4(in.position, 1.0);

    // Behind the camera the manual perspective divide below would wrap the
    // splat onto the opposite side of the screen, so discard it here instead.
    if (clip.w <= 0.0) {
        out.position = float4(0.0, 0.0, 2.0, 1.0);
        out.color = float4(0.0);
        out.localCoord = float2(0.0);
        return out;
    }
    const float3 projected = clip.xyz / clip.w;

    float compensation = 0.0;
    const float3 cov = computeCov2D(
        in.position, computeCov3D(in.scale, in.quat), u, compensation);

    out.color = in.color;
    // The compensation factor is only correct for opacities that were
    // optimised with it applied, so it follows the message's rasterize_mode.
    out.color.a *= metal::mix(1.0, compensation, u.antialiased);

    const float alphaCutoff = 1.0 / 255.0;
    if (out.color.a < alphaCutoff) {
        out.position = float4(0.0, 0.0, 2.0, 1.0);
        out.localCoord = float2(0.0);
        return out;
    }

    // alpha = opacity * exp(-r^2 / 2). Pixels beyond this radius would be
    // discarded by the fragment shader, so do not rasterise them in the first
    // place. This is exact with respect to the shader's alpha cutoff and can
    // shrink low-opacity splats substantially below the configured sigma cap.
    const float visibleRadius = metal::min(
        u.sigma_radius,
        metal::sqrt(2.0 * metal::log(out.color.a / alphaCutoff)));

    const float det = cov.x * cov.z - cov.y * cov.y;
    const float mid = 0.5 * (cov.x + cov.z);
    const float root = metal::sqrt(metal::max(0.0, mid * mid - det));
    const float lambda1 = metal::max(mid + root, 0.01);
    const float lambda2 = metal::max(mid - root, 0.01);

    float2 axis1;
    if (metal::abs(cov.y) > 1e-5) {
        axis1 = metal::normalize(float2(cov.y, lambda1 - cov.x));
    } else {
        axis1 = cov.x >= cov.z ? float2(1.0, 0.0) : float2(0.0, 1.0);
    }
    const float2 axis2 = float2(-axis1.y, axis1.x);

    const float2 pixelOffset =
        in.corner.x * axis1 * (visibleRadius * metal::sqrt(lambda1)) +
        in.corner.y * axis2 * (visibleRadius * metal::sqrt(lambda2));
    const float2 ndcOffset = pixelOffset * 2.0 / u.vpsize.xy;

    out.position = float4(projected.xy + ndcOffset, projected.z, 1.0);
    out.localCoord = in.corner * visibleRadius;
    return out;
}

fragment half4 gsplat_fp(GaussianSplatOut in [[stage_in]])
{
    const float power = -0.5 * metal::dot(in.localCoord, in.localCoord);
    const float alpha = metal::min(0.99, in.color.a * metal::exp(power));
    if (alpha < (1.0 / 255.0)) {
        metal::discard_fragment();
    }
    // Premultiplied alpha; see the GLSL fragment shader for why.
    return half4(half3(in.color.rgb * alpha), half(alpha));
}
