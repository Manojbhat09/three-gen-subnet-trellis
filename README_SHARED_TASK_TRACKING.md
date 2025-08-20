# Shared Task Tracking for TRELLIS Mining

This document describes the new shared task tracking functionality that prevents duplicate task processing across multiple mining instances.

## Overview

The shared task tracking system prevents multiple mining instances from processing the same task simultaneously, which was causing "no feedback" errors and wasted computational resources.

## How It Works

1. **Task Lock Acquisition**: When a mining instance pulls a task, it acquires a database lock
2. **Duplicate Prevention**: Other instances see the lock and skip that task
3. **Automatic Cleanup**: Expired locks (2+ minutes old) are automatically cleaned up
4. **Load Balancing**: Instances automatically distribute work across available validators

## Database Schema

The system adds a new table `shared_task_tracking` with the following structure:

```sql
CREATE TABLE shared_task_tracking (
    task_id TEXT PRIMARY KEY,
    validator_uid INTEGER NOT NULL,
    status TEXT DEFAULT 'in_progress',
    instance_id TEXT NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    timeout_at TIMESTAMP,
    completed_at TIMESTAMP,
    instance_hostname TEXT,
    instance_pid INTEGER
);
```

## Usage

### Enable Task Tracking (Default)
```bash
# Task tracking is enabled by default
./run_trellis_mining.sh --continuous --harvest --submit --vllm --vllm-url http://localhost:9004 --vllm-model llama-3-2-3b-it --generation-server http://localhost:8100 --trellis-server-port 8100
```

### Explicitly Enable Task Tracking
```bash
./run_trellis_mining.sh --continuous --harvest --submit --vllm --vllm-url http://localhost:9004 --vllm-model llama-3-2-3b-it --generation-server http://localhost:8100 --trellis-server-port 8100 --enable-task-tracking
```

### Disable Task Tracking
```bash
./run_trellis_mining.sh --continuous --harvest --submit --vllm --vllm-url http://localhost:9004 --vllm-model llama-3-2-3b-it --generation-server http://localhost:8100 --trellis-server-port 8100 --disable-task-tracking
```

## Running Multiple Instances

### Instance 1 (Port 8100)
```bash
CUDA_VISIBLE_DEVICES=0 ./run_trellis_mining.sh --continuous --harvest --submit --vllm --vllm-url http://localhost:9004 --vllm-model llama-3-2-3b-it --generation-server http://localhost:8100 --trellis-server-port 8100
```

### Instance 2 (Port 8097)
```bash
CUDA_VISIBLE_DEVICES=4 ./run_trellis_mining.sh --continuous --harvest --submit --vllm --vllm-url http://localhost:9001 --vllm-model llama-3-2-3b-it --generation-server http://localhost:8097 --trellis-server-port 8097
```

## Benefits

1. **No More Duplicate Tasks**: Each task is processed by only one instance
2. **Automatic Load Balancing**: Work is distributed across validators
3. **Fault Tolerance**: Expired locks allow recovery from crashed instances
4. **Real-time Monitoring**: See task distribution across all instances
5. **Backward Compatible**: Existing instances continue to work normally

## Monitoring

The system provides real-time statistics showing:

- Total tracked tasks
- Active tasks by instance
- Task distribution by validator
- Lock cleanup operations

Example output:
```
🔄 Shared Task Tracking:
  Total tracked tasks: 15
  Active tasks: 3
  Completed tasks: 12
  Task distribution by instance:
    d8dfb2328451_12345_abc12345: 2 tasks (this instance)
    d8dfb2328451_67890_def67890: 1 tasks
  Active tasks by validator:
    UID 142: 1 active tasks
    UID 212: 2 active tasks
```

## Configuration

Task tracking can be configured via:

- **Command line**: `--enable-task-tracking` or `--disable-task-tracking`
- **Default**: Enabled by default
- **Timeout**: 2 minutes (configurable in code)

## Troubleshooting

### Task Locks Not Being Released
- Check if instances are crashing unexpectedly
- Locks automatically expire after 2 minutes
- Manual cleanup occurs every hour

### Performance Issues
- Task tracking adds minimal database overhead
- Disable with `--disable-task-tracking` if needed
- Monitor database performance with large numbers of instances

### Database Corruption
- The system uses SQLite with proper transaction handling
- Backup your database regularly
- New tables are created with `CREATE TABLE IF NOT EXISTS`

## Migration

- **Existing instances**: Continue working normally
- **New instances**: Automatically benefit from task tracking
- **Database**: New tables are created automatically
- **No data loss**: All existing data is preserved

## Technical Details

- **Lock Timeout**: 2 minutes (configurable)
- **Cleanup Interval**: Every hour
- **Instance ID**: Generated as `hostname_pid_uuid8`
- **Database**: SQLite with proper indexing
- **Thread Safety**: SQLite handles concurrent access
