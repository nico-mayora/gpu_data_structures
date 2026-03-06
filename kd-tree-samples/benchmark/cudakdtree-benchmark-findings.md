# cudaKDTree Benchmark: Findings on Usage Differences

## Context

Our benchmark compares our KD-tree implementation against Ingo Wald's cudaKDTree library.
During a review, we compared how our benchmark was using cudaKDTree versus how the older
photon-mapping project (`photon-mapping`) used it.

## Key Finding: Different Candidate List Types

The benchmark was using `cukd::FlexHeapCandidateList` — a **runtime-K** variant that stores
its entries in **externally allocated global memory** — while the photon-mapping project
uses `cukd::HeapCandidateList<K>` — a **compile-time-K** variant that stores entries
**on the stack** (registers / local memory).

### What the benchmark was doing (before)

```cuda
// Kernel: non-templated, runtime k
__global__ void knn_query_cukd(
    const float3 *tree, int num_points,
    const float *query_positions, int num_queries,
    uint64_t *candidate_mem,    // pre-allocated global memory buffer
    int k)                      // runtime k
{
    uint64_t *my_mem = candidate_mem + (size_t)tid * k;
    cukd::FlexHeapCandidateList closest(my_mem, k, 1e30f);
    cukd::stackBased::knn(closest, qp, tree, num_points);
}
```

- `FlexHeapCandidateList` takes a pointer to **externally allocated** `uint64_t` memory.
- K is a **runtime** parameter — no template dispatch needed.
- Each thread's candidate entries live in **global memory**, accessed through the pointer.
- This was a workaround for a reported NVCC C++20 two-phase lookup issue with
  the templated `HeapCandidateList<K>`.

### What the photon-mapping project does

```cuda
// From shading.h — called per-ray on the GPU
cukd::HeapCandidateList<K_NEAREST_NEIGHBOURS> closest(K_MAX_DISTANCE);
cukd::stackBased::knn<
    cukd::HeapCandidateList<K_NEAREST_NEIGHBOURS>, Photon, Photon_traits
>(closest, queryPoint, photons, numPoints);
```

- `HeapCandidateList<K>` is a **compile-time templated** struct.
- Its `uint64_t entry[K]` array lives **inside the struct itself** — on the stack,
  in registers or local memory per thread.
- No external memory allocation per query.
- K is known at compile time (K=50 in the photon mapper).

### Why this matters

| Aspect | `FlexHeapCandidateList` (old benchmark) | `HeapCandidateList<K>` (photon-mapping) |
|---|---|---|
| K resolution | Runtime | Compile-time |
| Entry storage | External global memory (cudaMalloc) | Stack / registers / local memory |
| Memory traffic | Every candidate access goes to global mem | Small K can stay in registers |
| Per-query allocation | Required (`uint64_t * num_queries * k`) | None |
| Template instantiation | Single kernel for all K values | One kernel instantiation per K |

For **small K values** (8, 16, 32), the `HeapCandidateList<K>` approach can keep entries in
registers, avoiding global memory round-trips entirely. This gives cudaKDTree a structural
advantage that the benchmark was previously hiding by forcing all accesses through global memory.

For **large K values** (256, 512, 1024), entries spill to local memory regardless, so the
difference is smaller — but local memory still benefits from L1/L2 caching with better
access patterns than scattered global memory reads.

### Resolution

Updated the benchmark to use `HeapCandidateList<K>` with compile-time K template dispatch,
matching the photon-mapping project's usage pattern. This ensures a fairer comparison between
our KD-tree and cudaKDTree at each K value.
