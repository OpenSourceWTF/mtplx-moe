#include <Python.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

#include <nanobind/nanobind.h>

#include <unistd.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

class BufferLease {
 public:
  explicit BufferLease(nb::handle value) {
    if (PyObject_GetBuffer(
            value.ptr(),
            &view_,
            PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) != 0) {
      throw nb::python_error();
    }
    active_ = true;
  }

  BufferLease(const BufferLease&) = delete;
  BufferLease& operator=(const BufferLease&) = delete;

  ~BufferLease() {
    if (active_) {
      PyBuffer_Release(&view_);
    }
  }

  void* data() const { return view_.buf; }
  Py_ssize_t size() const { return view_.len; }

 private:
  Py_buffer view_{};
  bool active_{false};
};

std::size_t pread_exact_into(
    int fd,
    std::uint64_t offset,
    nb::handle destination) {
  if (fd < 0) {
    throw std::invalid_argument("fd must be non-negative");
  }
  if (offset > static_cast<std::uint64_t>(
                   std::numeric_limits<off_t>::max())) {
    throw std::overflow_error("file offset exceeds off_t");
  }
  BufferLease lease(destination);
  if (lease.size() < 0) {
    throw std::invalid_argument("destination has a negative size");
  }
  auto* bytes = static_cast<std::uint8_t*>(lease.data());
  const auto requested = static_cast<std::size_t>(lease.size());
  std::size_t total = 0;
  int error_number = 0;

  {
    nb::gil_scoped_release release;
    while (total < requested) {
      const auto max_offset = static_cast<std::uint64_t>(
          std::numeric_limits<off_t>::max());
      if (total > max_offset || offset > max_offset - total) {
        error_number = EOVERFLOW;
        break;
      }
      const std::uint64_t current_offset = offset + total;
      const std::size_t requested_count = std::min(
          requested - total,
          static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
      const ssize_t read_count = ::pread(
          fd,
          bytes + total,
          requested_count,
          static_cast<off_t>(current_offset));
      if (read_count > 0) {
        total += static_cast<std::size_t>(read_count);
        continue;
      }
      if (read_count == 0) {
        break;
      }
      if (errno == EINTR) {
        continue;
      }
      error_number = errno;
      break;
    }
  }

  if (error_number != 0) {
    throw std::system_error(
        error_number,
        std::generic_category(),
        "pread failed");
  }
  return total;
}

}  // namespace

NB_MODULE(_ext, module) {
  module.doc() = "Bounded GIL-free positional reads for MTPLX expert slots";
  module.def(
      "pread_exact_into",
      &pread_exact_into,
      "fd"_a,
      "offset"_a,
      "destination"_a,
      "Read from an existing descriptor into a writable contiguous buffer.");
}
