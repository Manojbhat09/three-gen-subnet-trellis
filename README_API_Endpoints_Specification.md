# API Endpoints Specification - Distributed RL System

## 🎯 Overview

Complete API specification for the Distributed RL System backend that supports the dashboard frontend and job management.

## 🌐 Base Configuration

- **Base URL**: `http://localhost:8090`
- **WebSocket Base**: `ws://localhost:8090`
- **Content-Type**: `application/json`
- **Authentication**: Bearer token (future implementation)

## 📡 REST API Endpoints

### System Management

#### GET /api/system/status
**Purpose**: Get overall system status and health
**Response**:
```json
{
  "status": "running",
  "uptime_seconds": 3600,
  "coordinator_health": "healthy",
  "total_gpus": 8,
  "healthy_gpus": 7,
  "busy_gpus": 5,
  "idle_gpus": 2,
  "error_gpus": 1,
  "memory_usage": {
    "coordinator_memory_mb": 512,
    "total_gpu_memory_gb": 192,
    "used_gpu_memory_gb": 145
  },
  "performance_stats": {
    "avg_score": 0.842,
    "best_score": 0.901,
    "worst_score": 0.623,
    "prompts_per_hour": 450
  }
}
```

#### GET /api/system/config
**Purpose**: Get current system configuration
**Response**:
```json
{
  "num_gpus": 8,
  "base_port": 8096,
  "coordinator_port": 8090,
  "batch_size_strategy": "performance_based",
  "strategy_sync_frequency": 300,
  "memory_sync_interval": 60,
  "max_concurrent_jobs": 3
}
```

#### POST /api/system/shutdown
**Purpose**: Graceful system shutdown
**Body**: `{ "force": false }`
**Response**: `{ "status": "shutting_down", "eta_seconds": 30 }`

#### POST /api/system/emergency_stop
**Purpose**: Immediate emergency stop
**Response**: `{ "status": "stopped", "stopped_at": "2024-01-01T12:00:00Z" }`

### GPU Management

#### GET /api/gpus/status
**Purpose**: Get status of all GPUs
**Response**:
```json
{
  "gpus": {
    "0": {
      "gpu_id": 0,
      "port": 8096,
      "status": "busy",
      "health": "healthy",
      "current_job_id": "job_123",
      "current_prompt": "a vintage red bicycle with chrome details",
      "current_round": 5,
      "max_rounds": 12,
      "current_best_score": 0.847,
      "memory_used_gb": 18.5,
      "memory_total_gb": 24.0,
      "utilization_percent": 95,
      "temperature_celsius": 67,
      "last_heartbeat": "2024-01-01T12:00:00Z",
      "processing_time_seconds": 145,
      "prompts_completed": 23,
      "errors_count": 0
    }
  }
}
```

#### GET /api/gpus/{gpu_id}/status
**Purpose**: Get detailed status of specific GPU
**Response**: Same structure as single GPU in above response

#### POST /api/gpus/{gpu_id}/restart
**Purpose**: Restart specific GPU server
**Response**: `{ "status": "restarting", "eta_seconds": 60 }`

#### POST /api/gpus/{gpu_id}/pause
**Purpose**: Pause processing on specific GPU
**Response**: `{ "status": "paused", "paused_at": "2024-01-01T12:00:00Z" }`

#### POST /api/gpus/{gpu_id}/resume
**Purpose**: Resume processing on specific GPU
**Response**: `{ "status": "resumed", "resumed_at": "2024-01-01T12:00:00Z" }`

#### POST /api/gpus/restart_all
**Purpose**: Restart all GPU servers
**Body**: `{ "exclude_gpus": [1, 3] }`
**Response**: `{ "status": "restarting_all", "excluded": [1, 3], "eta_seconds": 120 }`

#### POST /api/gpus/cleanup
**Purpose**: Clean up GPU memory
**Response**: `{ "status": "cleaned", "memory_freed_gb": 5.2 }`

### Job Management

#### GET /api/jobs/queue
**Purpose**: Get current job queue
**Response**:
```json
{
  "queue": [
    {
      "job_id": "job_124",
      "status": "queued",
      "priority": 1,
      "prompts_count": 80,
      "target_score": 0.85,
      "episodes": 10,
      "estimated_duration_minutes": 120,
      "submitted_at": "2024-01-01T11:30:00Z",
      "position_in_queue": 1
    }
  ],
  "total_queued": 1,
  "max_concurrent": 3
}
```

#### POST /api/jobs/submit
**Purpose**: Submit new RL optimization job
**Body**:
```json
{
  "prompts": ["a red car", "a blue house"],
  "target_score": 0.85,
  "max_episodes": 10,
  "max_rounds_per_episode": 12,
  "improvement_threshold": 0.03,
  "priority": 1,
  "job_name": "Car and House Optimization"
}
```
**Response**:
```json
{
  "job_id": "job_125",
  "status": "queued",
  "estimated_start": "2024-01-01T12:15:00Z",
  "estimated_duration_minutes": 60
}
```

#### GET /api/jobs/{job_id}/status
**Purpose**: Get detailed job status
**Response**:
```json
{
  "job_id": "job_123",
  "status": "running",
  "progress": {
    "total_prompts": 80,
    "completed_prompts": 32,
    "percentage": 40.0,
    "current_episode": 3,
    "total_episodes": 10
  },
  "performance": {
    "best_score": 0.887,
    "average_score": 0.742,
    "prompts_per_minute": 4.2,
    "eta_minutes": 45
  },
  "gpu_assignments": {
    "0": ["prompt1", "prompt2"],
    "1": ["prompt3", "prompt4"]
  },
  "started_at": "2024-01-01T11:00:00Z"
}
```

#### POST /api/jobs/{job_id}/pause
**Purpose**: Pause specific job
**Response**: `{ "status": "paused", "can_resume": true }`

#### POST /api/jobs/{job_id}/resume
**Purpose**: Resume paused job
**Response**: `{ "status": "resumed", "estimated_completion": "2024-01-01T13:00:00Z" }`

#### POST /api/jobs/{job_id}/cancel
**Purpose**: Cancel job
**Body**: `{ "save_partial_results": true }`
**Response**: `{ "status": "cancelled", "partial_results_saved": true }`

#### POST /api/jobs/{job_id}/priority
**Purpose**: Change job priority
**Body**: `{ "priority": 2 }`
**Response**: `{ "status": "priority_updated", "new_position": 3 }`

### Results & Analytics

#### GET /api/jobs/{job_id}/results
**Purpose**: Get job results
**Response**:
```json
{
  "job_id": "job_123",
  "status": "completed",
  "results": {
    "total_prompts": 80,
    "successful_optimizations": 78,
    "failed_optimizations": 2,
    "average_improvement": 0.145,
    "best_results": [
      {
        "original_prompt": "a red car",
        "optimized_prompt": "a sleek crimson sports car with chrome details",
        "score_improvement": 0.234,
        "final_score": 0.891,
        "gpu_id": 3,
        "optimization_time_seconds": 180
      }
    ]
  },
  "performance_stats": {
    "total_duration_minutes": 67,
    "average_gpu_utilization": 94.2,
    "throughput_prompts_per_hour": 71.6
  },
  "completed_at": "2024-01-01T12:07:00Z"
}
```

#### GET /api/jobs/{job_id}/results/export
**Purpose**: Export results in various formats
**Query Params**: `?format=json|csv|xlsx`
**Response**: File download or JSON data

#### GET /api/analytics/performance_report
**Purpose**: System performance analytics
**Query Params**: `?timeframe=1h|24h|7d&gpus=0,1,2`
**Response**:
```json
{
  "timeframe": "24h",
  "metrics": {
    "total_jobs": 15,
    "successful_jobs": 14,
    "average_job_duration_minutes": 45,
    "total_prompts_processed": 1200,
    "average_score_improvement": 0.123,
    "gpu_utilization_average": 87.3
  },
  "gpu_performance": {
    "0": { "utilization": 92.1, "prompts_completed": 156, "average_score": 0.834 }
  },
  "strategy_effectiveness": {
    "creative_expansion": { "success_rate": 0.78, "average_improvement": 0.145 }
  }
}
```

### Memory & Strategy Management

#### GET /api/memory/episodic/{prompt_hash}
**Purpose**: Get episodic memory for specific prompt
**Response**:
```json
{
  "prompt": "a red car",
  "prompt_hash": "abc123",
  "best_score": 0.891,
  "best_prompt": "a sleek crimson sports car",
  "episodes_run": 5,
  "total_attempts": 67,
  "curriculum_level": 2,
  "mastery_achieved": true,
  "strategy_performance": {
    "creative_expansion": { "attempts": 23, "avg_score": 0.756 }
  }
}
```

#### POST /api/memory/sync
**Purpose**: Force memory synchronization across GPUs
**Response**: `{ "status": "synced", "entries_updated": 145, "conflicts_resolved": 3 }`

#### GET /api/strategies/effectiveness
**Purpose**: Get strategy effectiveness across all GPUs
**Response**:
```json
{
  "strategies": {
    "creative_expansion": {
      "total_uses": 456,
      "success_rate": 0.78,
      "average_improvement": 0.145,
      "best_gpu": 3,
      "worst_gpu": 7
    }
  },
  "cross_gpu_insights": [
    {
      "insight": "creative_expansion works better on complex prompts",
      "confidence": 0.89,
      "supporting_gpus": [0, 2, 4]
    }
  ]
}
```

## 🔌 WebSocket Endpoints

### /ws/real_time_updates
**Purpose**: Real-time system status updates
**Message Format**:
```json
{
  "type": "system_update",
  "timestamp": "2024-01-01T12:00:00Z",
  "data": {
    "gpu_status": { /* same as REST API */ },
    "job_progress": { /* current progress */ },
    "system_health": { /* health metrics */ }
  }
}
```

### /ws/score_updates
**Purpose**: Real-time score improvements
**Message Format**:
```json
{
  "type": "score_update",
  "timestamp": "2024-01-01T12:00:00Z",
  "data": {
    "job_id": "job_123",
    "gpu_id": 3,
    "prompt": "a red car",
    "new_score": 0.887,
    "improvement": 0.045,
    "round": 7
  }
}
```

### /ws/system_alerts
**Purpose**: System alerts and notifications
**Message Format**:
```json
{
  "type": "alert",
  "level": "error|warning|info",
  "timestamp": "2024-01-01T12:00:00Z",
  "data": {
    "message": "GPU 5 encountered memory error",
    "component": "gpu",
    "component_id": "5",
    "action_required": true,
    "suggested_action": "restart_gpu"
  }
}
```

### /ws/job_events
**Purpose**: Job lifecycle events
**Message Format**:
```json
{
  "type": "job_event",
  "event": "started|completed|failed|paused|resumed",
  "timestamp": "2024-01-01T12:00:00Z",
  "data": {
    "job_id": "job_123",
    "status": "completed",
    "duration_minutes": 67,
    "results_summary": { /* brief results */ }
  }
}
```

## 🛡️ Error Responses

### Standard Error Format
```json
{
  "error": {
    "code": "GPU_NOT_FOUND",
    "message": "GPU with ID 9 not found",
    "details": {
      "available_gpus": [0, 1, 2, 3, 4, 5, 6, 7],
      "requested_gpu": 9
    },
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

### Common Error Codes
- `GPU_NOT_FOUND` (404): Requested GPU doesn't exist
- `GPU_BUSY` (409): GPU is busy and cannot perform action
- `JOB_NOT_FOUND` (404): Job ID doesn't exist
- `INVALID_PARAMETERS` (400): Request parameters are invalid
- `SYSTEM_OVERLOADED` (503): System at capacity
- `MEMORY_EXHAUSTED` (507): GPU memory exhausted
- `COORDINATOR_OFFLINE` (503): Coordinator unavailable

## 🔐 Authentication (Future)

### Bearer Token Format
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Token Scope
- `read:system` - Read system status
- `read:jobs` - Read job information  
- `write:jobs` - Create and manage jobs
- `admin:gpus` - GPU management operations
- `admin:system` - System administration

## 📊 Rate Limiting

### Limits per IP
- **GET requests**: 1000/hour
- **POST requests**: 200/hour
- **WebSocket connections**: 10 concurrent
- **Emergency endpoints**: 10/minute

### Headers
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1609459200
```

## 🧪 Testing Endpoints

### GET /api/test/ping
**Purpose**: Basic connectivity test
**Response**: `{ "message": "pong", "timestamp": "2024-01-01T12:00:00Z" }`

### POST /api/test/mock_job
**Purpose**: Create mock job for testing
**Body**: `{ "duration_seconds": 30, "gpu_count": 2 }`
**Response**: `{ "job_id": "test_job_1", "status": "running" }`

---

This API specification provides comprehensive endpoints for monitoring, controlling, and managing the distributed RL system while ensuring the frontend can provide rich real-time feedback without impacting system performance.




