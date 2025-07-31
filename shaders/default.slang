import input_structures;

struct VSOutput {
    float4 position : SV_POSITION;
    float3 normal : TEXCOORD0;
    float3 color : TEXCOORD1;
    float2 uv : TEXCOORD2;
};

struct Vertex {
    float3 position;
    float uv_x;
    float3 normal;
    float uv_y;
    float3 color;
};

struct VertexBuffer {
    Vertex vertices[];
};

struct PushConstants {
    float4x4 render_matrix;
    uint64_t vertex_buffer_address;
};

inline Ptr<VertexBuffer> get_vertex_buffer(uint64_t address) {
    return (VertexBuffer*)address;
}

[shader("vertex")]
VSOutput vert(uint vertexID : SV_VertexID, [[vk::push_constant]] uniform PushConstants pc) {
    var output = (VSOutput)0;
    var buffer = get_vertex_buffer(pc.vertex_buffer_address);
    var v = buffer->vertices[vertexID];

    output.position = mul(pc.render_matrix, float4(v.position, 1));
    output.color = v.color * materialData.colorFactors.xyz;
    output.uv = float2(v.uv_x, v.uv_y);

    return output;
}

[shader("fragment")]
float4 frag(VSOutput input) : SV_Target {
    float lightValue = max(dot(input.normal, sceneData.sunlightDirection.xyz), 0.1f);
    float3 color = input.color * colorTexture.Sample(input.uv).xyz;
    // float3 color = input.color * colorTexture.Sample(colorSampler, input.uv).xyz;
    float3 ambient = color * sceneData.ambientColor.xyz;
    return float4(color * lightValue * sceneData.sunlightColor.w + ambient, 1.0f);
}
