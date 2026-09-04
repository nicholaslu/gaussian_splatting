#version 120

// Full-screen quad supplied as an Ogre::Rectangle2D, whose vertices are
// already in clip space.
attribute vec4 vertex;
attribute vec2 uv0;

varying vec2 frag_uv;

void main()
{
  frag_uv = uv0;
  gl_Position = vec4(vertex.xy, 0.0, 1.0);
}
