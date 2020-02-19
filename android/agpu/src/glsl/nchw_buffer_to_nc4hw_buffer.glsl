layout(binding=0) writeonly buffer dstBuffer{
  float data[];
}uOutBuffer;

layout(binding=1) readonly buffer srcBuffer{
  float data[];
}uInBuffer;

layout(location=2) uniform int uWidth;
layout(location=3) uniform int uHeight;

layout (local_size_x = WORKGROUP_X, local_size_y = WORKGROUP_Y, local_size_z = WORKGROUP_Z) in;

void main()
{
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  int W = uWidth;
  int H = uHeight;
  int WH = W * H;

  if (pos.x < W && pos.y < H)
  {
      int z4 = pos.z*4;
      int y4 = pos.y*4;
      int x4 = pos.x*4;

      uOutBuffer.data[z4*WH + y4*W + x4 + 0] = uInBuffer.data[(z4+0)*WH + W*pos.y + pos.x];
      uOutBuffer.data[z4*WH + y4*W + x4 + 1] = uInBuffer.data[(z4+1)*WH + W*pos.y + pos.x];
      uOutBuffer.data[z4*WH + y4*W + x4 + 2] = uInBuffer.data[(z4+2)*WH + W*pos.y + pos.x];
      uOutBuffer.data[z4*WH + y4*W + x4 + 3] = uInBuffer.data[(z4+3)*WH + W*pos.y + pos.x];
  }
}
