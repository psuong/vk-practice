[[vk::binding(0, 0)]]
RWTexture2D<float4> image : register(u0);

float Noise2d(float2 x) {
    float xhash = cos(x.x * 37.0);
    float yhash = cos(x.y * 57.0);
    return frac(415.92653 * (xhash + yhash));
}

float NoisyStarField(float2 vSamplePos, float fThreshhold) {
    float StarVal = Noise2d(vSamplePos);
    if (StarVal >= fThreshhold)
        StarVal = pow((StarVal - fThreshhold) / (1.0 - fThreshhold), 6.0);
    else
        StarVal = 0.0;
    return StarVal;
}

float StableStarField(float2 vSamplePos, float fThreshhold) {
    // Linear interpolation between four samples
    float fractX = frac(vSamplePos.x);
    float fractY = frac(vSamplePos.y);
    float2 floorSample = floor(vSamplePos);

    float v1 = NoisyStarField(floorSample, fThreshhold);
    float v2 = NoisyStarField(floorSample + float2(0.0, 1.0), fThreshhold);
    float v3 = NoisyStarField(floorSample + float2(1.0, 0.0), fThreshhold);
    float v4 = NoisyStarField(floorSample + float2(1.0, 1.0), fThreshhold);

    float StarVal = v1 * (1.0 - fractX) * (1.0 - fractY) +
                    v2 * (1.0 - fractX) * fractY +
                    v3 * fractX * (1.0 - fractY) +
                    v4 * fractX * fractY;
    return StarVal;
}

void mainImage(out float4 fragColor, float2 fragCoord) {
    int2 size;
    image.GetDimensions(size.x, size.y);
    float2 iResolution = float2(size.x, size.y);

    // Sky Background Color
    float3 vColor = float3(0.1, 0.2, 0.4) * (fragCoord.y / iResolution.y);

    // Choose fThreshhold in the range [0.99, 0.9999] for sparse stars
    float StarFieldThreshhold = 0.97;

    // Stars with a slow crawl
    float xRate = 0.2;
    float yRate = -0.06;
    float2 vSamplePos = fragCoord + float2(xRate * 1.0, yRate * 1.0);
    float StarVal = StableStarField(vSamplePos, StarFieldThreshhold);
    vColor += float3(StarVal);

    fragColor = float4(vColor, 1.0);
}

[numthreads(16, 16, 1)]
void computeMain(uint3 dispatchID : SV_DispatchThreadID) {
    int2 texelCoord = int2(dispatchID.xy);
    int2 size;
    image.GetDimensions(size.x, size.y);

    if (texelCoord.x < size.x && texelCoord.y < size.y)
    {
        float4 color;
        mainImage(color, float2(texelCoord));

        image[texelCoord] = color;
    }
}
