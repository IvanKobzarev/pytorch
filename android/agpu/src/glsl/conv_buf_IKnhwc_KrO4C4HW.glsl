layout(std430) buffer;
layout(binding=0) writeonly buffer outBuffer{
  float data[];
}uOutputBuffer;

layout(binding=1) readonly buffer inputBuffer{
  float data[];
}uInBuffer;

layout(binding=2) readonly buffer kernelBuffer{
  vec4 data[];
}uKBuffer;

layout(binding=3) readonly buffer bias{
  vec4 data[];
}uBias;

layout(location=4) uniform ivec2 uPad;
layout(location=5) uniform ivec2 uKernelSize;
layout(location=6) uniform ivec2 uStride;
layout(location=7) uniform ivec2 uDilate;

layout(location=8) uniform ivec4 uOutputSize;
layout(location=9) uniform ivec4 uInputSize;

#define UP_DIV(x, y) (((x)+(y)-1)/(y))

layout (local_size_x = WORKGROUP_X, local_size_y = WORKGROUP_Y, local_size_z = WORKGROUP_Z) in;

void main()
{
    if (all(lessThan(ivec3(gl_GlobalInvocationID), uOutputSize.xyz)))
    {
        ivec3 pos = ivec3(gl_GlobalInvocationID)*ivec3(4, 1, 1);
        int posx_4 = pos.x / 4;
        int KW = uKernelSize.x;
        int KH = uKernelSize.y;
        ivec4 inputSize = uInputSize;
        int W = uInputSize.x;
        int H = uInputSize.y;
        int C_4 = uInputSize.z;
        int C = uInputSize.w;

        int OW = uOutputSize.x;
        int OH = uOutputSize.y;
        int OC_4 = uOutputSize.z;
        int OC = uOutputSize.w;

        int OW_4 = UP_DIV(OW, 4);

        ivec2 s0 = pos.xy*uStride-uPad;
        int kxi, kyi, ic_4i;

        int oc_4i = pos.z;

        ivec2 sfxy = max(ivec2(0), (UP_DIV(-s0, uDilate)));
        ivec2 efxy = min(uKernelSize, UP_DIV(inputSize.xy-s0, uDilate));
        vec4 v[4];
        v[0] = uBias.data[pos.z];
        v[1] = v[0];
        v[2] = v[0];
        v[3] = v[0];
        ivec4 inBi, inBisx0, inBisx1, inBisx2, inBisx3;
        vec4 k[4];
        vec4 invsx[4];
        int sx0, sx1, sx2, sx3;

        for (kyi=sfxy.y; kyi<efxy.y; ++kyi)
        {
            int sy = kyi*uDilate.y + s0.y;
            for (kxi=0; kxi < KW; ++kxi)
            {
                sx0 = kxi*uDilate.x + s0.x;
                sx1 = sx0 + uStride.x;
                sx2 = sx1 + uStride.x;
                sx3 = sx2 + uStride.x;

                float m0 = sx0 >= 0 && sx0 < W ? 1.0 : 0.0;
                float m1 = sx1 >= 0 && sx1 < W ? 1.0 : 0.0;
                float m2 = sx2 >= 0 && sx2 < W ? 1.0 : 0.0;
                float m3 = sx3 >= 0 && sx3 < W ? 1.0 : 0.0;

                for (ic_4i=0; ic_4i < C_4; ++ic_4i)
                {
                    int kBi = oc_4i * (C_4 * 4 * KH * KW) + ic_4i * 4 * KH * KW + (kxi + kyi*KW) * 4;
                    vec4 k0 = uKBuffer.data[kBi+0];
                    vec4 k1 = uKBuffer.data[kBi+1];
                    vec4 k2 = uKBuffer.data[kBi+2];
                    vec4 k3 = uKBuffer.data[kBi+3];

                    for (int i = 0; i < 4; ++i) {
                      k[i] = vec4(0.0);
                      invsx[i] = vec4(0.0);
                    }

                    int ic4ie = min(4, C - 4*ic_4i);
                    for (int ic4i=0;ic4i<ic4ie; ++ic4i)
                    {
                      invsx[0][ic4i] = uInBuffer.data[C*(W*sy + sx0) + 4*ic_4i + ic4i];
                      invsx[1][ic4i] = uInBuffer.data[C*(W*sy + sx1) + 4*ic_4i + ic4i];
                      invsx[2][ic4i] = uInBuffer.data[C*(W*sy + sx2) + 4*ic_4i + ic4i];
                      invsx[3][ic4i] = uInBuffer.data[C*(W*sy + sx3) + 4*ic_4i + ic4i];
                    }

                    mat4 kmat = mat4(k0, k1, k2, k3);

                    v[0] += kmat * invsx[0] * m0;
                    v[1] += kmat * invsx[1] * m1;
                    v[2] += kmat * invsx[2] * m2;
                    v[3] += kmat * invsx[3] * m3;
                }
            }
        }

        int vxi = 0;
        int vxie = min(4, OW-pos.x);

        int vci = 0;
        int vcie = min(4, OC-4*oc_4i);
        for (;vxi<vxie;++vxi)
        {
          int outBi = OC*(OW*pos.y + pos.x + vxi) + 4*oc_4i;
          vec4 v = v[vxi];
          vci = 0;
          for(;vci<vcie;++vci)
          {
            uOutputBuffer.data[outBi + vci] = v[vci];
          }
        }
    }

}
