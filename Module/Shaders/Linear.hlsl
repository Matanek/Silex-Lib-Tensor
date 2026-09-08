StructuredBuffer<float> leftValues : register(t0, space0);
StructuredBuffer<float> rightValues : register(t1, space0);
RWStructuredBuffer<float> outputValues : register(u0, space1);

cbuffer Metadata : register(b0, space2) {
    uint4 dimensions;
    uint4 ranks;
    uint4 offsets;
    uint4 batchShape0;
    uint4 batchShape1;
    uint4 leftShape0;
    uint4 leftShape1;
    uint4 leftStrides0;
    uint4 leftStrides1;
    uint4 rightShape0;
    uint4 rightShape1;
    uint4 rightStrides0;
    uint4 rightStrides1;
};

uint component(uint4 value, uint index) {
    if (index == 0) return value.x;
    if (index == 1) return value.y;
    if (index == 2) return value.z;
    return value.w;
}

uint pair_component(uint4 first, uint4 second, uint index) {
    if (index < 4) return component(first, index);
    return component(second, index - 4);
}

uint batch_offset(
    uint batch,
    uint rank,
    uint offset,
    uint4 shape0,
    uint4 shape1,
    uint4 strides0,
    uint4 strides1
) {
    uint result = offset;
    int inputBatchRank = int(rank) - 2;
    int missing = int(ranks.x) - inputBatchRank;
    uint remaining = batch;
    for (int outputAxis = int(ranks.x) - 1; outputAxis >= 0; --outputAxis) {
        uint dimension = pair_component(batchShape0, batchShape1, uint(outputAxis));
        uint coordinate = dimension == 0 ? 0 : remaining % dimension;
        if (dimension > 0) remaining /= dimension;
        int inputAxis = outputAxis - missing;
        if (inputAxis >= 0 && pair_component(shape0, shape1, uint(inputAxis)) != 1) {
            result += coordinate * pair_component(strides0, strides1, uint(inputAxis));
        }
    }
    return result;
}

[numthreads(1, 1, 1)]
void compute_main(uint3 id : SV_DispatchThreadID) {
    uint outputIndex = id.x;
    if (outputIndex >= dimensions.w) return;
    uint matrixSize = dimensions.x * dimensions.y;
    uint batch = ranks.w == 1 ? 0 : outputIndex / matrixSize;
    uint matrixIndex = ranks.w == 1 ? 0 : outputIndex % matrixSize;
    uint row = ranks.w == 1 ? 0 : matrixIndex / dimensions.y;
    uint column = ranks.w == 1 ? 0 : matrixIndex % dimensions.y;
    uint leftBase = batch_offset(batch, ranks.y, offsets.x, leftShape0, leftShape1, leftStrides0, leftStrides1);
    uint rightBase = batch_offset(batch, ranks.z, offsets.y, rightShape0, rightShape1, rightStrides0, rightStrides1);
    uint leftRowStride = ranks.w == 1 ? 0 : pair_component(leftStrides0, leftStrides1, ranks.y - 2);
    uint leftInnerStride = pair_component(leftStrides0, leftStrides1, ranks.y - 1);
    uint rightInnerStride = ranks.w == 1 ?
        pair_component(rightStrides0, rightStrides1, ranks.z - 1) :
        pair_component(rightStrides0, rightStrides1, ranks.z - 2);
    uint rightColumnStride = ranks.w == 1 ? 0 : pair_component(rightStrides0, rightStrides1, ranks.z - 1);
    float value = 0.0;
    for (uint inner = 0; inner < dimensions.z; ++inner) {
        uint leftIndex = leftBase + row * leftRowStride + inner * leftInnerStride;
        uint rightIndex = rightBase + inner * rightInnerStride + column * rightColumnStride;
        value += leftValues[leftIndex] * rightValues[rightIndex];
    }
    outputValues[outputIndex] = value;
}
