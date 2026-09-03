// Metal counterpart of ../GLSL/gsplat.{vert,frag}. The two covariance streams
// use TEXCOORD0/1, which Ogre's Metal mapping assigns to attributes 8 and 9.

struct GaussianSplatIn
{
    float3 position [[attribute(0)]];
    float4 color    [[attribute(3)]];
    float3 covDiag  [[attribute(8)]];
    float3 covUpper [[attribute(9)]];
};

struct GaussianSplatOut
{
    float4 position  [[position]];
    float  pointSize [[point_size]];
    float4 color;
    float3 conic;
    float2 radius;
};

struct GaussianSplatUniforms
{
    metal::float4x4 projmatrix;
    metal::float4x4 viewmatrix;
    float4 vpsize;
    float fovy;
};

static float3 computeCov2D(
    float3 position, float3 diag, float3 upper,
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
    const metal::float3x3 Vrk = metal::float3x3(
        float3(diag.x, upper.x, upper.y),
        float3(upper.x, diag.y, upper.z),
        float3(upper.y, upper.z, diag.z));
    metal::float3x3 cov = metal::transpose(T) * Vrk * T;

    const float detOrig = cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1];
    cov[0][0] += 0.3;
    cov[1][1] += 0.3;
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
    const float3 projected = clip.xyz / clip.w;

    float compensation = 0.0;
    const float3 cov = computeCov2D(in.position, in.covDiag, in.covUpper, u, compensation);
    out.color = in.color;
    out.color.a *= compensation;

    if (out.color.a < (1.0 / 255.0)) {
        out.position = float4(0.0, 0.0, 2.0, 1.0);
        out.pointSize = 1.0;
        out.conic = float3(0.0);
        out.radius = float2(0.0);
        return out;
    }

    const float det = cov.x * cov.z - cov.y * cov.y;
    const float mid = 0.5 * (cov.x + cov.z);
    const float root = metal::sqrt(metal::max(0.1, mid * mid - det));
    const float radius = metal::ceil(
        3.0 * metal::sqrt(metal::max(mid + root, mid - root)));

    out.position = float4(projected.xy, projected.z, 1.0);
    out.pointSize = 2.0 * radius;
    out.conic = float3(cov.z, -cov.y, cov.x) / det;
    out.radius = float2(radius);
    return out;
}

fragment half4 gsplat_fp(
    GaussianSplatOut in [[stage_in]], float2 pointCoord [[point_coord]])
{
    float2 d = (pointCoord * 2.0 - 1.0) * -in.radius;
    d.x *= -1.0;
    const float power =
        -0.5 * (in.conic.x * d.x * d.x + in.conic.z * d.y * d.y) -
        in.conic.y * d.x * d.y;
    if (power > 0.0) {
        metal::discard_fragment();
    }
    const float alpha = metal::min(0.99, in.color.a * metal::exp(power));
    return half4(half3(in.color.rgb), half(alpha));
}
