/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#define TVM_LOG_CUSTOMIZE 1

#include "hexagon_buffer.h"

#include <tvm/runtime/module.h>

#include "hexagon_common.h"

#if defined(__hexagon__)
#include "HAP_compute_res.h"
#endif

#include <string>
#include <utility>

namespace tvm {
namespace runtime {
namespace hexagon {

struct Allocation {
  Allocation(size_t nbytes, size_t alignment) : nbytes_(nbytes), alignment_(alignment) {}
  virtual ~Allocation() {}
  Allocation(const Allocation&) = delete;
  Allocation& operator=(const Allocation&) = delete;
  Allocation(Allocation&&) = delete;
  Allocation& operator=(Allocation&&) = delete;

  void* data_{nullptr};
  size_t nbytes_;
  size_t alignment_;
};

struct DDRAllocation : public Allocation {
  DDRAllocation(size_t nbytes, size_t alignment) : Allocation(nbytes, alignment) {
#ifdef _WIN32
    data_ = _aligned_malloc(nbytes, alignment);
    CHECK(data_ != nullptr);
#else
    int ret = posix_memalign(&data_, alignment, nbytes);
    CHECK_EQ(ret, 0);
#endif
  }
  ~DDRAllocation() {
#ifdef _WIN32
    _aligned_free(data_);
#else
    free(data_);
#endif
  }
};

#if defined(__hexagon__)
struct VTCMAllocation : public Allocation {
  VTCMAllocation(size_t nbytes, size_t alignment) : Allocation(nbytes, alignment) {
    compute_res_attr_t res_info;
    HEXAGON_SAFE_CALL(HAP_compute_res_attr_init(&res_info));

    // allocate nbytes of vtcm on a single page
    HEXAGON_SAFE_CALL(HAP_compute_res_attr_set_vtcm_param(&res_info, /*vtcm_size = */ nbytes,
                                                          /*b_single_page = */ 1));
    context_id_ = HAP_compute_res_acquire(&res_info, /*timeout = */ 10000);

    if (context_id_) {
      data_ = HAP_compute_res_attr_get_vtcm_ptr(&res_info);
      if (!data_) {
        HEXAGON_PRINT(ERROR, "ERROR: Allocated VTCM ptr is null.");
        HEXAGON_SAFE_CALL(HAP_compute_res_release(context_id_));
        return;
      }
    } else {
      HEXAGON_PRINT(ERROR, "ERROR: Unable to acquire requeisted resource.");
      return;
    }
    // HEXAGON_PRINT(ALWAYS, "VTCMAllocation() - Context ID: %u, VTCM ptr: %p", context_id_, data_);
  }
  ~VTCMAllocation() {
    // HEXAGON_PRINT(ALWAYS, "~VTCMAllocation() - Context ID: %u, VTCM ptr: %p", context_id_,
    // data_);
    HEXAGON_SAFE_CALL(HAP_compute_res_release(context_id_));
    data_ = nullptr;
  }
  unsigned int context_id_{0};
};
#else
struct VTCMAllocation : public DDRAllocation {
  VTCMAllocation(size_t nbytes, size_t alignment) : DDRAllocation(nbytes, alignment) {}
};
#endif

template <HexagonBuffer::StorageScope S>
std::unique_ptr<Allocation> Allocator(size_t nbytes, size_t alignment);

template <>
std::unique_ptr<Allocation> Allocator<HexagonBuffer::StorageScope::kDDR>(size_t nbytes,
                                                                         size_t alignment) {
  return std::make_unique<DDRAllocation>(nbytes, alignment);
}

template <>
std::unique_ptr<Allocation> Allocator<HexagonBuffer::StorageScope::kVTCM>(size_t nbytes,
                                                                          size_t alignment) {
  return std::make_unique<VTCMAllocation>(nbytes, alignment);
}

HexagonBuffer::HexagonBuffer(size_t nbytes, size_t alignment, Optional<String> scope)
    : nallocs_(1), nbytes_(nbytes) {
  SetStorageScope(scope);

  std::unique_ptr<Allocation> alloca = nullptr;
  if (GetStorageScope() == StorageScope::kDDR) {
    alloca = Allocator<StorageScope::kDDR>(nbytes, alignment);
  } else if (GetStorageScope() == StorageScope::kVTCM) {
    alloca = Allocator<StorageScope::kVTCM>(nbytes, alignment);
  }
  CHECK(alloca != nullptr);
  allocations_.push_back(alloca->data_);
  managed_allocations_.push_back(std::move(alloca));
}

HexagonBuffer::HexagonBuffer(size_t nallocs, size_t nbytes, size_t alignment,
                             Optional<String> scope)
    : nallocs_(nallocs), nbytes_(nallocs * nbytes) {
  SetStorageScope(scope);
  for (size_t i = 0; i < nallocs; ++i) {
    std::unique_ptr<Allocation> alloca = nullptr;
    if (GetStorageScope() == StorageScope::kDDR) {
      alloca = Allocator<StorageScope::kDDR>(nbytes, alignment);
    } else if (GetStorageScope() == StorageScope::kVTCM) {
      alloca = Allocator<StorageScope::kVTCM>(nbytes, alignment);
    }
    CHECK(alloca != nullptr);
    allocations_.push_back(alloca->data_);
    managed_allocations_.push_back(std::move(alloca));
  }
}

HexagonBuffer::HexagonBuffer(void* data, size_t nbytes, Optional<String> scope)
    : nallocs_(1), nbytes_(nbytes) {
  SetStorageScope(scope);
  // disallow external VTCM allocations
  CHECK(GetStorageScope() != HexagonBuffer::StorageScope::kVTCM);
  allocations_.push_back(data);
}

HexagonBuffer::~HexagonBuffer() { managed_allocations_.clear(); }

void** HexagonBuffer::GetPointer() {
  if (!allocations_.size()) {
    return nullptr;
  }
  return allocations_.data();
}

HexagonBuffer::StorageScope HexagonBuffer::GetStorageScope() const { return storage_scope_; }

void HexagonBuffer::SetStorageScope(Optional<String> scope) {
  if (!scope.defined()) {
    storage_scope_ = StorageScope::kDDR;
  } else {
    if (scope.value() == "global") {
      storage_scope_ = StorageScope::kDDR;
    } else if (scope.value() == "global.vtcm") {
      storage_scope_ = StorageScope::kVTCM;
    } else {
      CHECK(false) << "Encountered unknown HexagonBuffer storage scope: "
                   << std::string(scope.value());
    }
  }
}

void HexagonBuffer::CopyTo(void* data, size_t nbytes) {
  CHECK(nbytes_ == nbytes);
  size_t offset = 0;
  for (size_t i = 0; i < nallocs_; ++i) {
    CHECK(nbytes / nallocs_ == managed_allocations_[i]->nbytes_);

    memcpy(static_cast<char*>(data) + offset,
           static_cast<const char*>(managed_allocations_[i]->data_),
           managed_allocations_[i]->nbytes_);

    offset += managed_allocations_[i]->nbytes_;
  }
}

void HexagonBuffer::CopyFrom(void* data, size_t nbytes) {
  CHECK(nbytes_ == nbytes);
  size_t offset = 0;
  for (size_t i = 0; i < nallocs_; ++i) {
    CHECK(nbytes / nallocs_ == managed_allocations_[i]->nbytes_);

    memcpy(static_cast<char*>(managed_allocations_[i]->data_),
           static_cast<const char*>(data) + offset, managed_allocations_[i]->nbytes_);

    offset += managed_allocations_[i]->nbytes_;
  }
}

void HexagonBuffer::CopyFrom(const HexagonBuffer& other) {
  CHECK(nbytes_ == other.nbytes_);

  if (nallocs_ == other.nallocs_) {
    for (size_t i = 0; i < nallocs_; ++i) {
      CHECK(managed_allocations_[i]->nbytes_ == other.managed_allocations_[i]->nbytes_);

      memcpy(static_cast<char*>(managed_allocations_[i]->data_),
             static_cast<const char*>(other.managed_allocations_[i]->data_),
             managed_allocations_[i]->nbytes_);
    }
  } else if (nallocs_ == 1) {
    size_t offset = 0;
    for (size_t i = 0; i < other.nallocs_; ++i) {
      CHECK(nbytes_ / other.nallocs_ == other.managed_allocations_[i]->nbytes_);

      memcpy(static_cast<char*>(managed_allocations_[0]->data_) + offset,
             static_cast<const char*>(other.managed_allocations_[i]->data_),
             other.managed_allocations_[i]->nbytes_);

      offset += other.managed_allocations_[i]->nbytes_;
    }
  } else if (other.nallocs_ == 1) {
    size_t offset = 0;
    for (size_t i = 0; i < nallocs_; ++i) {
      CHECK(other.nbytes_ / nallocs_ == managed_allocations_[i]->nbytes_);

      memcpy(static_cast<char*>(managed_allocations_[i]->data_),
             static_cast<const char*>(other.managed_allocations_[0]->data_) + offset,
             managed_allocations_[i]->nbytes_);

      offset += managed_allocations_[i]->nbytes_;
    }
  } else {
    CHECK(false) << "To copy between Hexagon Buffers they must either have the same number of "
                    "dimensions or one of the Hexagon Buffers must have a single dimension.";
  }
}

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
