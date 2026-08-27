// SPDX-License-Identifier: Apache-2.0
// Mutable shared Metal arena for the bounded Qwen4 n-gram row cache.

#include <Python.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

#include <nanobind/nanobind.h>

#include "mlx/allocator.h"
#include "mlx/array.h"

#include <Metal/Metal.hpp>

namespace nb = nanobind;
namespace mx = mlx::core;
using namespace nb::literals;

namespace {

nb::object allocate_metal_u8_2d(
    std::uint64_t rows,
    std::uint64_t columns) {
  if (rows == 0 || columns == 0) {
    throw std::invalid_argument("Metal allocation dimensions must be non-zero");
  }
  const auto shape_max = static_cast<std::uint64_t>(
      std::numeric_limits<mx::ShapeElem>::max());
  if (rows > shape_max || columns > shape_max) {
    throw std::overflow_error(
        "Metal allocation dimension exceeds MLX shape limits");
  }
  if (rows > std::numeric_limits<std::uint64_t>::max() / columns) {
    throw std::overflow_error("Metal allocation byte size overflows uint64");
  }
  const std::uint64_t length = rows * columns;
  if (length > static_cast<std::uint64_t>(
                   std::numeric_limits<NS::UInteger>::max())) {
    throw std::overflow_error("Metal allocation exceeds platform byte limits");
  }

  constexpr auto options = static_cast<MTL::ResourceOptions>(
      MTL::ResourceStorageModeShared |
      MTL::ResourceHazardTrackingModeUntracked);
  static MTL::Device* device = MTL::CreateSystemDefaultDevice();
  if (device == nullptr) {
    throw std::runtime_error("no default Metal device is available");
  }
  MTL::Buffer* buffer = nullptr;
  {
    nb::gil_scoped_release release;
    buffer = device->newBuffer(static_cast<NS::UInteger>(length), options);
  }
  if (buffer == nullptr || buffer->contents() == nullptr) {
    if (buffer != nullptr) {
      buffer->release();
    }
    throw std::runtime_error("Metal refused the mutable shared allocation");
  }

  mx::array result(
      mx::allocator::Buffer{static_cast<void*>(buffer)},
      mx::Shape{
          static_cast<mx::ShapeElem>(rows),
          static_cast<mx::ShapeElem>(columns),
      },
      mx::uint8,
      [](mx::allocator::Buffer allocation) {
        auto* metal_buffer = static_cast<MTL::Buffer*>(allocation.ptr());
        if (metal_buffer != nullptr) {
          metal_buffer->release();
        }
      });
  nb::object array_class = nb::module_::import_("mlx.core").attr("array");
  nb::object output = array_class(nb::int_(0));
  nb::inst_ptr<mx::array>(output)->overwrite_descriptor(result);
  return output;
}

}  // namespace

NB_MODULE(_ngram_cache_mlx, module) {
  module.doc() = "Fixed mutable Metal arena for exact n-gram row caching";
  module.def(
      "allocate_metal_u8_2d",
      &allocate_metal_u8_2d,
      "rows"_a,
      "columns"_a,
      "Allocate a shape-bounded mutable shared Metal byte matrix.");
}
