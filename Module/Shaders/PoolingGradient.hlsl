StructuredBuffer<float> gradientValues : register(t0, space0);
StructuredBuffer<float> inputValues : register(t1, space0);
RWStructuredBuffer<float> outputValues : register(u0, space1);

cbuffer Metadata : register(b0, space2) {
    uint4 header0;
    uint4 header1;
    uint4 gradientShape;
    uint4 gradientStrides;
    uint4 inputShape;
    uint4 inputStrides;
    uint4 outputShape;
};

uint gradient_index(uint batch, uint channel, uint y, uint x) {
    return header1.y + batch * gradientStrides.x + channel * gradientStrides.y + y * gradientStrides.z + x * gradientStrides.w;
}

uint input_index(uint batch, uint channel, uint y, uint x) {
    return header1.z + batch * inputStrides.x + channel * inputStrides.y + y * inputStrides.z + x * inputStrides.w;
}

[numthreads(1, 1, 1)]
void compute_main(uint3 id : SV_DispatchThreadID) {
    uint logical = id.x;
    if (logical >= header0.x) return;
    uint remaining = logical;
    uint inputX = remaining % inputShape.w;
    remaining /= inputShape.w;
    uint inputY = remaining % inputShape.z;
    remaining /= inputShape.z;
    uint channel = remaining % inputShape.y;
    uint batch = remaining / inputShape.y;
    float value = 0.0;

    for (uint outputY = 0; outputY < outputShape.z; ++outputY) {
        int startY = int(outputY * header0.z) - int(header0.w);
        for (uint outputX = 0; outputX < outputShape.w; ++outputX) {
            int startX = int(outputX * header0.z) - int(header0.w);
            if (int(inputY) < startY || int(inputY) >= startY + int(header0.y) ||
                int(inputX) < startX || int(inputX) >= startX + int(header0.y)) continue;
            float incoming = gradientValues[gradient_index(batch, channel, outputY, outputX)];
            if (header1.x == 1) {
                value += incoming / float(header0.y * header0.y);
                continue;
            }
            int winnerY = -1;
            int winnerX = -1;
            float winner = asfloat(0xff800000);
            for (uint kernelY = 0; kernelY < header0.y; ++kernelY) {
                int candidateY = startY + int(kernelY);
                for (uint kernelX = 0; kernelX < header0.y; ++kernelX) {
                    int candidateX = startX + int(kernelX);
                    if (candidateY >= 0 && candidateY < int(inputShape.z) && candidateX >= 0 && candidateX < int(inputShape.w)) {
                        float next = inputValues[input_index(batch, channel, uint(candidateY), uint(candidateX))];
                        if (winnerY < 0 || isnan(next) || next > winner) {
                            winner = next;
                            winnerY = candidateY;
                            winnerX = candidateX;
                        }
                    }
                }
            }
            if (winnerY == int(inputY) && winnerX == int(inputX)) value += incoming;
        }
    }
    outputValues[logical] = value;
}
