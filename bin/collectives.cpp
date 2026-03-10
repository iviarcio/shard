// ABI for Collectives:
//
// The current Hexagon runtime ABI appends execution IDs at the tail of the
// entry-point signature. Right-to-left (last -> first), the relevant slots look
// like:
//   [..., ntpc, num_cores, <reserved>, tid, cid, <reserved>]
//
// Therefore, with last = args[N-1]:
//
//   coreId             = args[N-2]
//   threadId           = args[N-3]
//   numCores           = args[N-5]
//   numThreadsPerCore  = args[N-6]

// Compute linear process id:
//   linearIdx = coreId * numThreadsPerCore + threadId
// Compute num of workers:
//   numWorkers = numCores * numThreadsPerCore
//

// Element type encoding chosen by compiler and runtime.
enum NSPElementType {
  NSP_ET_F16 = 0,
  NSP_ET_F32 = 1,
  NSP_ET_I32 = 2,
  NSP_ET_I64 = 3
};

// Reduction kind encoding chosen by compiler and runtime.
enum NSPReductionKind {
  NSP_RED_ADD = 0,
  NSP_RED_MAX = 1,
  NSP_RED_MIN = 2,
  NSP_RED_MUL = 3
};

int32_t __nsp_all_gather_1d(
    void *src,
    void *dst,
    int64_t localElemCount,
    int32_t elemType,
    int32_t numWorkers,
    int32_t linearIdx);

int32_t __nsp_all_reduce_1d(
    void *src,
    void *dst,
    int64_t elemCount,
    int32_t elemType,
    int32_t reductionKind,
    int32_t numWorkers,
    int32_t linearIdx);

// Restrictions:

//   Initial support only for 1-D contiguous memrefs.
//   Runtime wrapper exposed as `func.func private @__nsp_all_gather_1d(...) -> i3
//   The operation already appears in memref form.
//   Valid only when the collective is performed across the entire grid (1D).
//
// If we have something equivalent to a subgroup — for example, a collective only
// between threads on the same core, or only between cores while keeping the 
// threadId fixed — then:
//   numWorkers is no longer numCores * numThreadsPerCore.
//   linearIdx is no longer the correct local group ID.
//
// In this case, we will need a group-local linear index. For example:
//
// Collective between threads of the same core:
//   groupSize = numThreadsPerCore
//   groupLinearIdx = threadId
//
// Collective between cores with a fixed threadId:
//   groupSize = numCores
//   groupLinearIdx = coreId
