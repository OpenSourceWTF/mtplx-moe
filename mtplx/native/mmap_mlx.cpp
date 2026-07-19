// SPDX-License-Identifier: Apache-2.0
// File-backed, no-copy MLX arrays for Apple unified memory.

#include <Python.h>

#include <cerrno>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "mlx/array.h"
#include "mlx/allocator.h"

#include <Metal/Metal.hpp>

namespace nb = nanobind;
namespace mx = mlx::core;
using namespace nb::literals;

namespace {

std::size_t prefetch_arrays(const nb::list& values) {
  struct Range {
    void* pointer;
    std::size_t length;
  };
  std::vector<Range> ranges;
  ranges.reserve(nb::len(values));
  // nb::inst_ptr is an unchecked cast: handed a Python object that is not an
  // mlx.core.array it reinterprets arbitrary object storage as an mx::array
  // and segfaults. MLX's Python module and this extension carry separate
  // hidden nanobind RTTI, so nb::isinstance<mx::array> cannot be trusted to
  // recognise MLX's own type; compare against the real class object instead.
  static const nb::object array_class =
      nb::module_::import_("mlx.core").attr("array");
  for (nb::handle value : values) {
    if (!nb::isinstance(value, array_class)) {
      throw std::invalid_argument("prefetch inputs must be mlx.core.array");
    }
    const auto* array = nb::inst_ptr<mx::array>(value);
    // Every MLX array on the Metal backend holds an MTL::Buffer* in
    // allocator::Buffer, whichever mmap_u32 flavor produced it: the wired
    // flavor's raw-pointer constructor routes through
    // MetalAllocator::make_buffer (newBufferWithBytesNoCopy:), and the
    // unwired flavor builds the MTLBuffer here. MLX's own Buffer::raw_ptr()
    // performs this identical contents() message send.
    // MLX >= 0.31 returns const void* from Buffer::ptr(); madvise needs the
    // mutable pointer to the same shared storage.
    auto* buffer =
        static_cast<MTL::Buffer*>(const_cast<void*>(array->buffer().ptr()));
    if (buffer == nullptr || buffer->contents() == nullptr) {
      throw std::invalid_argument("prefetch input has no shared Metal storage");
    }
    ranges.push_back(
        Range{static_cast<char*>(buffer->contents()) + array->offset(),
              array->nbytes()});
  }
  std::size_t failures = 0;
  {
    nb::gil_scoped_release release;
    for (const auto& range : ranges) {
      if (::madvise(range.pointer, range.length, MADV_WILLNEED) != 0) {
        failures++;
      }
    }
  }
  return failures;
}

nb::object mmap_u32(
    const std::string& path,
    std::uint64_t offset,
    std::uint64_t length,
    bool wired) {
  if (length == 0 || (length % sizeof(std::uint32_t)) != 0) {
    throw std::invalid_argument("mmap length must be a non-zero multiple of 4");
  }
  const long page_size_value = ::sysconf(_SC_PAGESIZE);
  if (page_size_value <= 0) {
    throw std::runtime_error("could not determine the virtual-memory page size");
  }
  const auto page_size = static_cast<std::uint64_t>(page_size_value);
  if ((offset % page_size) != 0 || (length % page_size) != 0) {
    throw std::invalid_argument("mmap offset and length must be page aligned");
  }
  const auto elements = length / sizeof(std::uint32_t);
  if (elements > static_cast<std::uint64_t>(
                     std::numeric_limits<mx::ShapeElem>::max())) {
    throw std::overflow_error("file-backed array exceeds MLX shape limits");
  }
  if (offset > static_cast<std::uint64_t>(
                   std::numeric_limits<off_t>::max()) ||
      length > static_cast<std::uint64_t>(
                   std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error("file-backed array exceeds platform limits");
  }

  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) {
    throw std::system_error(errno, std::generic_category(), "open failed");
  }
  struct stat info {};
  if (::fstat(fd, &info) != 0) {
    const int error_number = errno;
    ::close(fd);
    throw std::system_error(
        error_number, std::generic_category(), "fstat failed");
  }
  if (!S_ISREG(info.st_mode)) {
    ::close(fd);
    throw std::invalid_argument("file-backed MLX storage must be a regular file");
  }
  const auto file_size = static_cast<std::uint64_t>(info.st_size);
  if (offset > file_size || length > file_size - offset) {
    ::close(fd);
    throw std::out_of_range("mmap range exceeds the source file");
  }

  void* data = ::mmap(
      nullptr,
      static_cast<std::size_t>(length),
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE,
      fd,
      static_cast<off_t>(offset));
  const int mmap_error = errno;
  ::close(fd);
  if (data == MAP_FAILED) {
    throw std::system_error(
        mmap_error, std::generic_category(), "mmap failed");
  }
  (void)::madvise(data, static_cast<std::size_t>(length), MADV_NORMAL);

  mx::array result = [&]() {
    if (wired) {
      return mx::array(
          data,
          mx::Shape{static_cast<mx::ShapeElem>(elements)},
          mx::uint32,
          [length](void* pointer) {
            if (pointer != nullptr) {
              (void)::munmap(pointer, static_cast<std::size_t>(length));
            }
          });
    }

    // MLX's allocator registers wrapped buffers in one process-wide wired
    // residency set. That is correct for ordinary model weights but disastrous
    // for a sidecar larger than RAM: submitting any command can make Metal
    // page in every registered allocation. Construct the MTLBuffer directly
    // so only commands that bind this expert range make it resident.
    constexpr auto options = static_cast<MTL::ResourceOptions>(
        MTL::ResourceStorageModeShared |
        MTL::ResourceHazardTrackingModeUntracked);
    // Device discovery is surprisingly expensive (~80 ms on the target M5
    // Max). Cache the process-wide Metal device instead of repeating it for
    // every 10 MiB expert record.
    static MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (device == nullptr) {
      (void)::munmap(data, static_cast<std::size_t>(length));
      throw std::runtime_error("no default Metal device is available");
    }
    MTL::Buffer* buffer = nullptr;
    {
      // Allow parallel startup mapping. Metal's no-copy buffer creation has
      // a high fixed cost, but independent records can be registered
      // concurrently when Python worker threads are not blocked by the GIL.
      nb::gil_scoped_release release;
      buffer = device->newBuffer(
          data,
          static_cast<NS::UInteger>(length),
          options,
          nullptr);
    }
    if (buffer == nullptr) {
      (void)::munmap(data, static_cast<std::size_t>(length));
      throw std::runtime_error("Metal refused the external file-backed buffer");
    }
    return mx::array(
        mx::allocator::Buffer{static_cast<void*>(buffer)},
        mx::Shape{static_cast<mx::ShapeElem>(elements)},
        mx::uint32,
        [data, length](mx::allocator::Buffer allocation) {
          auto* metal_buffer = static_cast<MTL::Buffer*>(allocation.ptr());
          if (metal_buffer != nullptr) {
            metal_buffer->release();
          }
          (void)::munmap(data, static_cast<std::size_t>(length));
        });
  }();

  // MLX's Python module and third-party extensions use separate hidden
  // nanobind RTTI. Mint a Python-side array and replace its descriptor, the
  // same bridge used by MTPLX's paged-attention extension.
  nb::object array_class = nb::module_::import_("mlx.core").attr("array");
  nb::object output = array_class(nb::int_(0));
  nb::inst_ptr<mx::array>(output)->overwrite_descriptor(result);
  return output;
}

}  // namespace

NB_MODULE(_mmap_mlx, module) {
  module.doc() = "No-copy file-backed MLX arrays for streamed MoE weights";
  module.def(
      "mmap_u32",
      &mmap_u32,
      "path"_a,
      "offset"_a,
      "length"_a,
      "wired"_a = false,
      "Map a page-aligned file range as a no-copy MLX uint32 array.");
  module.def(
      "prefetch",
      &prefetch_arrays,
      "arrays"_a,
      "Advise macOS that file-backed MLX record pages will be needed soon.");
}
