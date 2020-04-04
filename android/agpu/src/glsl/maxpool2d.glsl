layout(FORMAT, binding=0) writeonly uniform PRECISION image3D uOutput;
layout(FORMAT, binding=1) readonly uniform PRECISION image3D uInput;

layout(location = 2) uniform ivec2 uKernel;
layout(location = 3) uniform ivec2 uStride;
layout(location = 4) uniform ivec2 uPad;
layout(location = 5) uniform ivec2 uDilate;

layout(location=10) uniform ivec3 uOutputSize;
layout(location=11) uniform ivec3 uInputSize;

#define UP_DIV(x, y) (((x)+(y)-1)/(y))

layout (local_size_x = WORKGROUP_X, local_size_y = WORKGROUP_Y, local_size_z = WORKGROUP_Z) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    ivec3 outputSize = uOutputSize;
    ivec2 spos = pos.xy*uStride-uPad;

    if (all(lessThan(pos, outputSize)))
    {
        ivec3 inputSize = uInputSize;
        ivec2 sfxy = max(ivec2(0), UP_DIV(-spos, uDilate));
        ivec2 efxy = min(uKernel, UP_DIV(inputSize.xy-spos, uDilate));
        vec4 v = vec4(-100000.0);
        for (int fy=sfxy.y; fy<efxy.y; ++fy)
        {
            for (int fx=sfxy.x; fx<efxy.x; ++fx)
            {
                v = max(v, imageLoad(uInput, ivec3(spos.x+fx, spos.y+fy, pos.z)));
            }
        }
        imageStore(uOutput, pos, v);
    }
}
