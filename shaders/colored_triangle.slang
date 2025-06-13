struct VSOutput {
    float4 position : SV_Position;
    float3 color    : TEXCOORD0;
};

VSOutput vert(uint vertexID : SV_VertexID) {
    float3 positions[3] = {
        float3( 1.0f,  1.0f, 0.0f),
        float3(-1.0f,  1.0f, 0.0f),
        float3( 0.0f, -1.0f, 0.0f)
    };

    float3 colors[3] = {
        float3(1.0f, 0.0f, 0.0f),
        float3(0.0f, 1.0f, 0.0f),
        float3(0.0f, 0.0f, 1.0f)
    };

    VSOutput output;
    output.position = float4(positions[vertexID], 1.0f);
    output.color = colors[vertexID];
    return output;
}

float4 frag(float3 color : TEXCOORD0) : SV_Target {
    return float4(color, 1.0f);
}
