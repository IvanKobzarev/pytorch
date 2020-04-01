layout(FORMAT, binding=0) writeonly uniform PRECISION image3D uImage;

layout(binding=1) readonly buffer destBuffer{
    float data[];
} uInBuffer;

layout(location = 2) uniform ivec4 uInputSize;
layout(location = 3) uniform int uH;
layout(location = 4) uniform int uC;

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    int W = uInputSize.x;
    int H = uInputSize.y;
    if (pos.x < uInputSize.x && pos.y < uInputSize.y)
    {
        int C = uInputSize.z;
        int z = pos.z*4;
        int vie = min(4, C - z);
        vec4 v;
        int idx0 = C*(pos.y*W + pos.x) + pos.z;
        int vi = 0;
        for(;vi<vie;++vi){
          v[vi] = uInBuffer.data[idx0+vi];
        }
        imageStore(uImage, pos, v);
    }
}
