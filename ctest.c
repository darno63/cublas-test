#include <Python.h>

#include <cblas.h>
#include <cublas.h>
#include <cuda_runtime.h>


static PyObject *matrix_multiply_cublas(PyObject *self, PyObject *args) {

  PyObject *a, *b, *r;
  if (!PyArg_ParseTuple(args, "OOO", &a, &b, &r))
    return nullptr;

  Py_buffer A, B, R;
  if (PyObject_GetBuffer(a, &A, PyBUF_CONTIG))
    return nullptr;
  if (PyObject_GetBuffer(b, &B, PyBUF_CONTIG))
    return nullptr;
  if (PyObject_GetBuffer(r, &R, PyBUF_CONTIG))
    return nullptr;
  if (A.itemsize != sizeof(float) || A.ndim != 2 ||
      B.itemsize != sizeof(float) || B.ndim != 2 ||
      R.itemsize != sizeof(float) || R.ndim != 2)
    return nullptr;

  cublasHandle_t handle; // CUBLAS context

  float *d_A;
  float *d_B;
  float *d_R;

  auto memsize = [](Py_buffer &A) {
    return static_cast<size_t>(A.shape[0] * A.shape[1] * A.itemsize);
  };

  cublasCreate_v2(&handle);

  cudaMalloc(reinterpret_cast<void **>(&d_A), memsize(A));
  cudaMalloc(reinterpret_cast<void **>(&d_B), memsize(B));
  cudaMemcpy(d_A, A.buf, memsize(A), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.buf, memsize(B), cudaMemcpyHostToDevice);
  cudaMalloc(reinterpret_cast<void **>(&d_R), memsize(R));

  auto n = static_cast<int>(A.shape[0]);
  const float alpha = 1.0;
  const float beta = 0.0;
  cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                 static_cast<const float *>(d_A), n,
                 static_cast<const float *>(d_B), n, &beta, d_R, n);

  cudaMemcpy(R.buf, d_R, memsize(R), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_R);

  PyBuffer_Release(&A);
  PyBuffer_Release(&B);
  PyBuffer_Release(&R);

  Py_IncRef(Py_None);
  return Py_None;
}

static PyMethodDef Methods[] = {
    {"mul_cublas", matrix_multiply_cublas, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,
                                    "ctest",
                                    nullptr,
                                    -1,
                                    Methods,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr};

PyMODINIT_FUNC PyInit_ctest(void) {
  return PyModule_Create(&module);
}
