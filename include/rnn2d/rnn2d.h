#ifndef RNN2D_RNN2D_H_
#define RNN2D_RNN2D_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CUstream_st *cudaStream_t;

struct rnn2dContext;
typedef struct rnn2dContext *rnn2dHandle_t;

struct rnn2dLstmStruct;
typedef struct rnn2dLstmStruct *rnn2dLstmDescriptor_t;

typedef enum {
  RNN2D_STATUS_SUCCESS         = 0,
  RNN2D_STATUS_NOT_INITIALIZED = 1,
  RNN2D_STATUS_BAD_PARAM       = 2,
  RNN2D_STATUS_NOT_IMPLEMENTED = 3
} rnn2dStatus_t;

rnn2dStatus_t rnn2dCreate(rnn2dHandle_t *handle);
rnn2dStatus_t rnn2dDestroy(rnn2dHandle_t handle);
rnn2dStatus_t rnn2dSetDevice(rnn2dHandle_t handle, int device);
rnn2dStatus_t rnn2dGetDevice(rnn2dHandle_t handle, int* device);
rnn2dStatus_t rnn2dSetStream(rnn2dHandle_t handle, cudaStream_t streamId);
rnn2dStatus_t rnn2dGetStream(rnn2dHandle_t handle, cudaStream_t *streamId);

#ifdef __cplusplus
}
#endif

#endif  // RNN2D_RNN2D_H_
