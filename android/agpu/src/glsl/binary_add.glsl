
layout(FORMAT, binding=0) writeonly uniform PRECISION image3D uOutput;
layout(location=1) uniform mediump sampler3D uInput0;
layout(location=2) uniform mediump sampler3D uInput1;
layout(location=3) uniform ivec4 imgSize;

layout (local_size_x = COMP_GROUP_X, local_size_y = COMP_GROUP_Y, local_size_z = COMP_GROUP_Z) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    ivec3 inSize = imgSize.xyz;
    if(all(lessThan(pos, inSize)))
    {
        vec4 sum = texelFetch(uInput0, pos, 0) + texelFetch(uInput1, pos, 0);
        imageStore(uOutput, pos, sum);
    }
}
