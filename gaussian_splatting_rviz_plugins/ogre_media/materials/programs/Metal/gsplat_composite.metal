// Composites the half-resolution splat target over the full-resolution scene.

struct CompositeIn
{
    float3 position [[attribute(0)]];
    float2 uv       [[attribute(8)]];
};

struct CompositeOut
{
    float4 position [[position]];
    float2 uv;
};

vertex CompositeOut gsplat_composite_vp(CompositeIn in [[stage_in]])
{
    CompositeOut out;
    // Rectangle2D vertices arrive in clip space already.
    out.position = float4(in.position.xy, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

fragment half4 gsplat_composite_fp(
    CompositeOut in [[stage_in]],
    metal::texture2d<half> splat_texture [[texture(0)]],
    metal::sampler splat_sampler [[sampler(0)]])
{
    // Already premultiplied; blended with "one one_minus_src_alpha".
    return splat_texture.sample(splat_sampler, in.uv);
}
