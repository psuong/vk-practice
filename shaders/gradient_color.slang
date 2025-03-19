[[vk::binding(0, 0)]]
RWTexture2D<float4> image : register(u0);

struct PushConstants {
    float4 data1;
    float4 data2;
    float4 data3;
    float4 data4;
};

[[vk::push_constant]]
PushConstants push_constants;

[numthreads(16, 16, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    int2 texel_coord = int2(dispatch_id.xy);
    int2 size;
    image.GetDimensions(size.x, size.y);

    float4 top_color = push_constants.data1;
    float4 bottom_color = push_constants.data2;

    if (texel_coord.x < size.x && texel_coord.y < texel_coord.y) {
        float blend = float(texel_coord.x) / float(size.y);
        image[texel_coord] = lerp(top_color, bottom_color, blend);
    }
}
