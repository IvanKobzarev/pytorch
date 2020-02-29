layout(binding=0) writeonly buffer outBuffer{
  float data[];
}uOutBuffer;

layout(binding=1) readonly buffer inBuffer{
  vec4 data[];
}uInBuffer;

layout(location=2) uniform ivec2 uInputSize;

layout (local_size_x = WORKGROUP_X, local_size_y = WORKGROUP_Y, local_size_z = WORKGROUP_Z) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    int W = uInputSize.x;
    int H = uInputSize.y;

    if (pos.x < W && pos.y < H)
    {
      vec4 v = uInBuffer.data[pos.z * W * H + pos.y * W + pos.x];
      int z = pos.z*4;
      uOutBuffer.data[pos.x + W*pos.y + (z+0)*W*H] = v.x;
      uOutBuffer.data[pos.x + W*pos.y + (z+1)*W*H] = v.y;
      uOutBuffer.data[pos.x + W*pos.y + (z+2)*W*H] = v.z;
      uOutBuffer.data[pos.x + W*pos.y + (z+3)*W*H] = v.w;
    }
}
