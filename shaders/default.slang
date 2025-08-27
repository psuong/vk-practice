import input_structures;

struct VSOutput {
    float4 position : SV_POSITION;
    float3 normal : TEXCOORD0;
    float3 color : TEXCOORD1;
    float2 uv : TEXCOORD2;
};

struct Vertex {
    float3 position; // 12
    float  uv_x; // 4
    float3 normal; // 12
    float  uv_y; // 4
    float4 color; // 16
};

struct VertexBuffer {
    Vertex vertices[];
};

struct PushConstants {
    float4x4 render_matrix;
    uint64_t vertex_buffer_address;
};

inline Ptr<VertexBuffer> get_vertex_buffer(uint64_t address) {
    return (Ptr<VertexBuffer>)address;
}

[shader("vertex")]
VSOutput vert(uint vertexID : SV_VertexID, [[vk::push_constant]] uniform PushConstants pc) {
    var output = (VSOutput)0;
    var buffer = get_vertex_buffer(pc.vertex_buffer_address);
    var v = buffer->vertices[vertexID];

    output.position = mul(mul(sceneData.viewProj, pc.render_matrix), float4(v.position.xyz, 1.0));
    output.normal = mul(pc.render_matrix, float4(v.normal, 0.0f)).xyz;
    output.color = v.color.xyz * materialData.colorFactors.xyz;
    output.uv = float2(v.uv_x, v.uv_y);

    return output;
}

[shader("fragment")]
float4 frag(VSOutput input) : SV_Target {
    float lightValue = max(dot(input.normal, sceneData.sunlightDirection.xyz), 0.1f);
    float3 color = input.color * colorTexture.Sample(input.uv).xyz;
    float3 ambient = color * sceneData.ambientColor.xyz;
    return float4(color * lightValue * sceneData.sunlightColor.w + ambient, 1.0f);
}
