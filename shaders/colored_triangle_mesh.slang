struct VSOutput {
    float4 position : SV_POSITION;
    float3 color : COLOR;
    float2 uv : TEXCOORD0;
};

struct Vertex {
    float3 position;
    float uv_x;
    float3 normal;
    float uv_y;
    float4 color;
};

struct VertexBuffer {
    Vertex vertices[];
}

struct PushConstants {
    float4x4 render_matrix;
    uint64_t vertex_buffer_address;
};

inline Ptr<VertexBuffer> get_vertex_buffer(uint64_t address) {
    return (VertexBuffer*)address;
}

[shader("vertex")]
VSOutput vert(uint vertexID : SV_VERTEXID, [[vk::push_constant]] uniform PushConstants pc) {
    var output = (VSOutput)0;
    var buffer = get_vertex_buffer(pc.vertex_buffer_address);
    var v = buffer->vertices[vertexID];

    output.position = mul(pc.render_matrix, float4(v.position, 1.0));
    output.color = v.color.rgb;
    output.uv = float2(v.uv_x, v.uv_y);

    return output;
}

[shader("fragment")]
float4 frag(VSOutput input) : SV_Target {
    return float4(input.color, 1.0);
}
