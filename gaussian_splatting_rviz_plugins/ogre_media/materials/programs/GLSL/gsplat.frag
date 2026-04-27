#version 120

varying vec4 splat_colour;
varying vec3 splat_conic;
varying vec2 splat_radius;

void main()
{
  vec2 d = (gl_PointCoord * 2.0 - 1.0) * -splat_radius;
  d.x *= -1.0;

  float power =
    -0.5 * (splat_conic.x * d.x * d.x + splat_conic.z * d.y * d.y) -
    splat_conic.y * d.x * d.y;

  if (power > 0.0) {
    discard;
  }

  float alpha = min(0.99, splat_colour.a * exp(power));
  gl_FragColor = vec4(splat_colour.rgb, alpha);
}
