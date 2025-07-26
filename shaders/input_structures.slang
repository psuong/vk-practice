struct SceneData {
    float4x4 view;
    float4x4 proj;
    float4x4 viewProj;
    float4 ambientColor;
    float4 sunlightDirection;
    float4 sunlightColor;
};

struct GLTFMaterialData {
    float4 colorFactors;
    float4 metal_rough_factors;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SceneData> sceneData;

[[vk::binding(1, 0)]]
ConstantBuffer<GLTFMaterialData> materialData;

[[vk::binding(1, 1)]]
Texture2D colorTexture;
[[vk::binding(1, 2)]]
SamplerState colorSampler;

[[vk::binding(1, 3)]]
Texture2D metalRoughTex;
[[vk::binding(1, 4)]]
SamplerState metalRoughSampler;
