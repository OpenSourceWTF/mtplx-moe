## 2026-07-13 06:47 [saved]
Goal: Make benchmark bottleneck diagnosis resource-based and agent-readable.
Decisions:
- Correlate uncached reader throughput with queue, worker, byte, fence, CPU, and GPU evidence on one clock.
- Keep telemetry opt-in and label enabled token rates diagnostic until reproduced without instrumentation.
- Report absent GPU and DRAM measurements as unavailable; emit candidates only when their required evidence exists.
- Treat cached reader bytes as logical demand; require F_NOCACHE for SSD-ceiling attribution.
Rejected:
- Infer serialization from elapsed time or low SSD use alone.
- Treat pending Metal fences as GPU utilization or routed bytes as DRAM traffic.
Open: Capture authorized process GPU samples when attribution remains incomplete.
