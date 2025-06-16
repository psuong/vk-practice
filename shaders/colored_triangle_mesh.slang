struct Vertex {
    float3 position;
    float uv_x;
    float3 normal;
    float uv_y;
    float4 color;
};

[[vk::binding(0, 0)]]
StructuredBuffer<Vertex> vertexBuffer;

struct VSOutput {
    float4 position : SV_POSITION;
    float3 color : COLOR;
    float2 uv : TEXCOORD0;
};

struct PushConstants {
    float4x4 render_matrix;
};

[shader("vertex")]
VSOutput vert(uint vertexID : SV_VERTEXID, [[vk::push_constant]] PushConstants pc) {
    
}
