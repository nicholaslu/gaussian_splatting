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

// With MetalFX upscaling: the colour MetalFX upscaled, and the alpha of the
// target it upscaled from, filtered, since MetalFX writes an alpha of one.
fragment half4 gsplat_composite_upscaled_fp(
    CompositeOut in [[stage_in]],
    metal::texture2d<half> upscaled_texture [[texture(0)]],
    metal::texture2d<half> splat_texture [[texture(1)]],
    metal::sampler upscaled_sampler [[sampler(0)]],
    metal::sampler splat_sampler [[sampler(1)]])
{
    // Sharpening can undershoot below zero at an edge.
    const half3 colour = metal::max(upscaled_texture.sample(upscaled_sampler, in.uv).rgb, 0.0h);
    return half4(colour, splat_texture.sample(splat_sampler, in.uv).a);
}
