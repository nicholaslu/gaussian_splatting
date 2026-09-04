#version 120

uniform sampler2D splat_texture;

varying vec2 frag_uv;

void main()
{
  // The target already holds premultiplied alpha, so it is emitted as-is and
  // blended with "one one_minus_src_alpha".
  gl_FragColor = texture2D(splat_texture, frag_uv);
}
