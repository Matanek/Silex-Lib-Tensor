StructuredBuffer<float> inputValues : register(t0, space0);
RWStructuredBuffer<float> outputValues : register(u0, space1);

cbuffer ScalarUniform : register(b0, space2) {
    float scalar;
};

[numthreads(1, 1, 1)]
void compute_main(uint3 id : SV_DispatchThreadID) {
    outputValues[id.x] = inputValues[id.x] * scalar;
}
