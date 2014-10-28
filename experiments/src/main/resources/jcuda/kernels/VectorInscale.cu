extern "C"
__global__ void fSigmoid(float *original, int size) {
  const int X = gridDim.x;
  const int index = gridDim.y * X * threadIdx.x + X * blockIdx.y + blockIdx.x;

  if(index < size) {
    original[index] = 1.f / (1.f + expf(-original[index]));
  }
}

extern "C"
__global__ void fExp(float *original, int size) {
  const int X = gridDim.x;
  const int index = gridDim.y * X * threadIdx.x + X * blockIdx.y + blockIdx.x;

  if(index < size) {
    original[index] = expf(original[index]);
  }
}
