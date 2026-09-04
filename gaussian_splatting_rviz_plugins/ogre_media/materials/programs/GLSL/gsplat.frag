#version 120

varying vec4 splat_colour;
varying vec2 splat_local_coord;

void main()
{
  float power = -0.5 * dot(splat_local_coord, splat_local_coord);
  float alpha = min(0.99, splat_colour.a * exp(power));
  if (alpha < 0.00392156862) {
    discard;
  }
  gl_FragColor = vec4(splat_colour.rgb, alpha);
}
