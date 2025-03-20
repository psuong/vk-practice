struct PushConstants {
    float4 data1;
    float4 data2;
    float4 data3;
    float4 data4;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pushConstants;

[[vk::binding(0, 0)]]
RWTexture2D<float4> image;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 globalID : SV_DispatchThreadID) {
    int2 texel_coord = int2(globalID.xy);
    int2 size;
    image.GetDimensions(size.x, size.y);

    float4 top_color = pushConstants.data1;
    float4 bottom_color = pushConstants.data2;

    if (texel_coord.x < size.x && texel_coord.y < texel_coord.y) {
        float blend = float(texel_coord.y) / float(size.y);
        image[texel_coord] = lerp(top_color, bottom_color, blend);
    }
}
