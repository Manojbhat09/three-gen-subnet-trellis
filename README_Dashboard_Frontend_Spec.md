# RL Dashboard Frontend Specification

## 🎯 Overview

A real-time monitoring dashboard for the Distributed RL System that provides live visibility into 8-GPU parallel processing without impacting system performance.

## 🎨 Dashboard Layout & Components

### Main Dashboard Layout (Single Page Application)

```
┌─────────────────────────────────────────────────────────────────┐
│                    RL Optimization Dashboard                    │
├─────────────────────────────────────────────────────────────────┤
│ [System Status] [Job Controls] [Export] [Settings] [Emergency]  │
├──────────────────┬──────────────────────────────────────────────┤
│                  │                                              │
│   System Panel   │             GPU Monitoring Grid             │
│                  │                                              │
│  ┌─────────────┐ │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐        │
│  │ Total Jobs  │ │  │GPU 0 │ │GPU 1 │ │GPU 2 │ │GPU 3 │        │
│  │ Queued: 3   │ │  │ BUSY │ │ IDLE │ │ BUSY │ │ BUSY │        │
│  │ Running: 8  │ │  └──────┘ └──────┘ └──────┘ └──────┘        │
│  │ Done: 42    │ │                                              │
│  └─────────────┘ │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐        │
│                  │  │GPU 4 │ │GPU 5 │ │GPU 6 │ │GPU 7 │        │
│  ┌─────────────┐ │  │ BUSY │ │ERROR │ │ BUSY │ │ IDLE │        │
│  │ Performance │ │  └──────┘ └──────┘ └──────┘ └──────┘        │
│  │ Avg: 0.842  │ │                                              │
│  │ Best: 0.901 │ │                                              │
│  │ Worst:0.623 │ │                                              │
│  └─────────────┘ │                                              │
├──────────────────┼──────────────────────────────────────────────┤
│                  │                                              │
│ Progress Panel   │            Detailed Metrics                 │
│                  │                                              │
│ [████████░░] 80% │  [Real-time Charts & Analytics]             │
│ ETA: 45 min      │                                              │
│                  │                                              │
└──────────────────┴──────────────────────────────────────────────┘
```

## 📋 Component Specifications

### 1. System Status Panel (Top Left)

#### Job Queue Status
```jsx
<JobQueueStatus>
  <Metric label="Queued" value={queuedJobs} color="orange" />
  <Metric label="Running" value={runningJobs} color="green" />
  <Metric label="Completed" value={completedJobs} color="blue" />
  <Metric label="Failed" value={failedJobs} color="red" />
</JobQueueStatus>
```

#### Performance Summary
```jsx
<PerformanceSummary>
  <Metric label="Average Score" value={avgScore} format="0.000" />
  <Metric label="Best Score" value={bestScore} format="0.000" />
  <Metric label="Worst Score" value={worstScore} format="0.000" />
  <TrendIndicator trend={scoreTrend} />
</PerformanceSummary>
```

#### System Health
```jsx
<SystemHealth>
  <HealthIndicator label="Coordinator" status={coordinatorStatus} />
  <HealthIndicator label="GPUs" status={`${healthyGPUs}/8`} />
  <HealthIndicator label="Memory" status={memoryStatus} />
  <HealthIndicator label="Network" status={networkStatus} />
</SystemHealth>
```

### 2. GPU Monitoring Grid (Top Right)

#### Individual GPU Card
```jsx
<GPUCard gpuId={id}>
  <GPUHeader>
    <GPUId>GPU {id}</GPUId>
    <StatusBadge status={gpu.status} /> {/* BUSY/IDLE/ERROR */}
    <PortLabel>:{gpu.port}</PortLabel>
  </GPUHeader>
  
  <GPUMetrics>
    <MemoryBar used={gpu.memoryUsed} total={gpu.memoryTotal} />
    <UtilizationBar percentage={gpu.utilization} />
    <Temperature value={gpu.temperature} />
  </GPUMetrics>
  
  <CurrentTask>
    {gpu.currentPrompt && (
      <>
        <PromptPreview>{gpu.currentPrompt.slice(0, 30)}...</PromptPreview>
        <ProgressBar 
          current={gpu.currentRound} 
          total={gpu.maxRounds} 
        />
        <Score current={gpu.currentBestScore} />
      </>
    )}
  </CurrentTask>
  
  <GPUActions>
    <IconButton icon="restart" onClick={() => restartGPU(id)} />
    <IconButton icon="details" onClick={() => showGPUDetails(id)} />
  </GPUActions>
</GPUCard>
```

#### Status Color Coding
- **BUSY** (Green): Currently processing prompts
- **IDLE** (Blue): Available for work
- **ERROR** (Red): Requires attention/restart
- **RECOVERY** (Orange): Recovering from error

### 3. Progress Panel (Bottom Left)

#### Overall Progress
```jsx
<OverallProgress>
  <ProgressBar 
    percentage={overallProgress} 
    label={`${completedPrompts}/${totalPrompts} prompts`}
  />
  <ETADisplay eta={estimatedTimeRemaining} />
  <ThroughputDisplay 
    promptsPerMinute={currentThroughput}
    trend={throughputTrend}
  />
</OverallProgress>
```

#### Active Jobs List
```jsx
<ActiveJobsList>
  {activeJobs.map(job => (
    <JobItem key={job.id}>
      <JobId>{job.id}</JobId>
      <JobProgress percentage={job.progress} />
      <JobControls>
        <Button onClick={() => pauseJob(job.id)}>Pause</Button>
        <Button onClick={() => prioritizeJob(job.id)}>Priority</Button>
      </JobControls>
    </JobItem>
  ))}
</ActiveJobsList>
```

### 4. Detailed Metrics Panel (Bottom Right)

#### Real-time Charts
```jsx
<MetricsCharts>
  <ScoreProgressionChart 
    data={scoreHistory}
    title="Score Progression Over Time"
    realTime={true}
  />
  
  <GPUUtilizationChart 
    data={gpuUtilizationHistory}
    title="GPU Utilization"
    realTime={true}
  />
  
  <ThroughputChart 
    data={throughputHistory}
    title="Prompts Processed/Hour"
    realTime={true}
  />
</MetricsCharts>
```

#### Strategy Effectiveness Heatmap
```jsx
<StrategyHeatmap>
  <HeatmapGrid 
    strategies={strategies}
    gpus={gpus}
    effectiveness={strategyEffectiveness}
    colorScale="greenToRed"
  />
  <HeatmapLegend />
</StrategyHeatmap>
```

## 🔗 Complete API Integration Guide

### **Base Configuration**
```javascript
const API_BASE_URL = 'http://localhost:8090';
const WS_BASE_URL = 'ws://localhost:8090';

// API client with error handling
const apiClient = {
  get: async (endpoint) => {
    const response = await fetch(`${API_BASE_URL}${endpoint}`);
    if (!response.ok) throw new Error(`API Error: ${response.status}`);
    return response.json();
  },
  
  post: async (endpoint, data) => {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    if (!response.ok) throw new Error(`API Error: ${response.status}`);
    return response.json();
  }
};
```

### **Core API Endpoints (Available Now)**

#### **1. System Status & Monitoring**
```javascript
// GET /api/system/status - Get complete system status
const getSystemStatus = async () => {
  return await apiClient.get('/api/system/status');
};

/* Response Format:
{
  "status": "running",
  "timestamp": "2024-01-15T10:30:00Z",
  "jobs": {
    "queued": 3,
    "active": 2, 
    "completed": 15
  },
  "gpus": {
    "0": { "status": "busy", "current_job": "job_abc123", "performance_score": 1.2, "error_count": 0 },
    "1": { "status": "idle", "current_job": null, "performance_score": 0.9, "error_count": 0 },
    "2": { "status": "error", "current_job": null, "performance_score": 0.5, "error_count": 3 }
  },
  "cross_gpu_insights": 45,
  "redis_available": true
}
*/
```

#### **2. Job Management**
```javascript
// POST /api/jobs/submit - Submit new job
const submitJob = async (jobData) => {
  return await apiClient.post('/api/jobs/submit', jobData);
};

/* Request Format:
{
  "prompts": ["a red sports car", "a blue house"],
  "target_score": 0.85,          // optional, default: 0.85
  "max_episodes": 5,             // optional, default: 5
  "max_rounds_per_episode": 8,   // optional, default: 8
  "priority": 2                  // optional, default: 1
}

Response Format:
{
  "status": "success",
  "job_id": "job_abc123",
  "prompts_count": 2,
  "queue_position": 1
}
*/

// GET /api/jobs/{job_id} - Get job status
const getJobStatus = async (jobId) => {
  return await apiClient.get(`/api/jobs/${jobId}`);
};

/* Response Formats:
// Queued Job:
{
  "job_id": "job_abc123",
  "status": "queued", 
  "prompts_count": 2,
  "submitted_at": "2024-01-15T10:30:00Z"
}

// Active Job:
{
  "job_id": "job_abc123",
  "status": "active",
  "prompts_count": 2,
  "submitted_at": "2024-01-15T10:30:00Z"
}

// Completed Job:
{
  "job_id": "job_abc123", 
  "status": "completed",
  "completed_at": "2024-01-15T10:35:00Z",
  "results": {
    "average_score": 0.87,
    "best_score": 0.91,
    "processing_time_minutes": 4.2,
    "prompts_processed": 2
  },
  "job": {
    "prompts_count": 2,
    "target_score": 0.85,
    "submitted_at": "2024-01-15T10:30:00Z"
  }
}
*/
```

#### **3. Cross-GPU Insights**
```javascript
// GET /api/insights - Get cross-GPU learning insights
const getCrossGPUInsights = async () => {
  return await apiClient.get('/api/insights');
};

/* Response Format:
{
  "insights": [
    {
      "gpu_id": 0,
      "prompt": "a red sports car",
      "strategy": "creative_expansion",
      "score_achieved": 0.91,
      "improvement_delta": 0.15,
      "timestamp": "2024-01-15T10:32:00Z",
      "received_at": "2024-01-15T10:32:01Z"
    }
  ],
  "strategy_performance": {
    "creative_expansion": {
      "total_uses": 15,
      "total_score": 12.3,
      "success_count": 12,
      "avg_score": 0.82,
      "success_rate": 0.8
    },
    "detail_enhancement": {
      "total_uses": 8,
      "total_score": 6.1,
      "success_count": 5,
      "avg_score": 0.76,
      "success_rate": 0.625
    }
  },
  "total_insights": 45
}
*/
```

#### **4. Individual GPU Status**
```javascript
// GET to individual GPU agents for detailed status
const getGPUStatus = async (gpuId) => {
  const port = 8096 + gpuId;
  return await fetch(`http://localhost:${port}/status`).then(res => res.json());
};

/* Response Format:
{
  "gpu_id": 0,
  "status": "busy",  // "idle", "busy"
  "current_job": "job_abc123",
  "stats": {
    "prompts_processed": 45,
    "total_episodes": 180,
    "average_score": 0.83,
    "strategies_shared": 12,
    "insights_received": 8,
    "memory_cache_size": 23,
    "insights_buffer_size": 15
  }
}
*/
```

### **WebSocket Integration (Real-time Updates)**

```javascript
// WebSocket connection with reconnection logic
class DashboardWebSocket {
  constructor(endpoint, onMessage) {
    this.endpoint = `${WS_BASE_URL}${endpoint}`;
    this.onMessage = onMessage;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 10;
    this.connect();
  }
  
  connect() {
    this.ws = new WebSocket(this.endpoint);
    
    this.ws.onopen = () => {
      console.log(`WebSocket connected: ${this.endpoint}`);
      this.reconnectAttempts = 0;
    };
    
    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.onMessage(data);
      } catch (error) {
        console.error('WebSocket message parse error:', error);
      }
    };
    
    this.ws.onclose = () => {
      this.reconnect();
    };
    
    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }
  
  reconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.min(1000 * this.reconnectAttempts, 30000);
      setTimeout(() => this.connect(), delay);
    }
  }
  
  close() {
    this.ws?.close();
  }
}

// Usage in React component
const useDashboardWebSocket = () => {
  const [systemStatus, setSystemStatus] = useState(null);
  
  useEffect(() => {
    const ws = new DashboardWebSocket('/ws/updates', (data) => {
      // Handle different update types
      switch (data.type) {
        case 'system_status':
          setSystemStatus(data.status);
          break;
        case 'job_progress':
          updateJobProgress(data.job_id, data.progress);
          break;
        case 'strategy_insight':
          addNewInsight(data.insight);
          break;
        case 'job_completed':
          handleJobCompletion(data.job_id, data.results);
          break;
      }
    });
    
    return () => ws.close();
  }, []);
  
  return systemStatus;
};
```

### **Dashboard Data Fetching Patterns**

#### **Initial Load Pattern**
```javascript
const DashboardDataProvider = ({ children }) => {
  const [systemData, setSystemData] = useState({
    systemStatus: null,
    jobStatuses: {},
    insights: null,
    gpuStatuses: {}
  });
  
  // Initial data load
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        const [systemStatus, insights] = await Promise.all([
          getSystemStatus(),
          getCrossGPUInsights()
        ]);
        
        setSystemData(prev => ({
          ...prev,
          systemStatus,
          insights
        }));
        
        // Load individual GPU statuses
        const gpuStatuses = {};
        for (let gpuId = 0; gpuId < 8; gpuId++) {
          try {
            gpuStatuses[gpuId] = await getGPUStatus(gpuId);
          } catch (error) {
            gpuStatuses[gpuId] = { status: 'error', error: error.message };
          }
        }
        
        setSystemData(prev => ({ ...prev, gpuStatuses }));
        
      } catch (error) {
        console.error('Failed to load initial data:', error);
      }
    };
    
    loadInitialData();
  }, []);
  
  return (
    <DataContext.Provider value={systemData}>
      {children}
    </DataContext.Provider>
  );
};
```

#### **Polling Pattern for Non-WebSocket Data**
```javascript
const usePolling = (fetchFunction, interval = 5000) => {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    const poll = async () => {
      try {
        const result = await fetchFunction();
        setData(result);
        setError(null);
      } catch (err) {
        setError(err.message);
      }
    };
    
    poll(); // Initial fetch
    const intervalId = setInterval(poll, interval);
    
    return () => clearInterval(intervalId);
  }, [fetchFunction, interval]);
  
  return { data, error };
};

// Usage
const SystemStatusPanel = () => {
  const { data: systemStatus, error } = usePolling(getSystemStatus, 2000);
  
  if (error) return <ErrorDisplay error={error} />;
  if (!systemStatus) return <LoadingSpinner />;
  
  return <SystemStatusDisplay status={systemStatus} />;
};
```

### **Error Handling & Fallback Strategies**

```javascript
const withErrorBoundary = (WrappedComponent) => {
  return class extends React.Component {
    constructor(props) {
      super(props);
      this.state = { hasError: false, error: null };
    }
    
    static getDerivedStateFromError(error) {
      return { hasError: true, error };
    }
    
    componentDidCatch(error, errorInfo) {
      console.error('Dashboard component error:', error, errorInfo);
    }
    
    render() {
      if (this.state.hasError) {
        return (
          <div className="error-fallback">
            <h2>Dashboard Component Error</h2>
            <p>The system is still running, but this component failed to render.</p>
            <button onClick={() => window.location.reload()}>
              Refresh Dashboard
            </button>
          </div>
        );
      }
      
      return <WrappedComponent {...this.props} />;
    }
  };
};
```

### **Testing API Integration**

```javascript
// API integration tests
const testAPIIntegration = async () => {
  console.log('Testing API integration...');
  
  try {
    // Test system status
    const systemStatus = await getSystemStatus();
    console.log('✅ System status:', systemStatus);
    
    // Test job submission
    const jobResponse = await submitJob({
      prompts: ['test prompt'],
      target_score: 0.8
    });
    console.log('✅ Job submitted:', jobResponse);
    
    // Test job status
    const jobStatus = await getJobStatus(jobResponse.job_id);
    console.log('✅ Job status:', jobStatus);
    
    // Test insights
    const insights = await getCrossGPUInsights();
    console.log('✅ Insights:', insights);
    
    console.log('🎉 All API tests passed!');
    
  } catch (error) {
    console.error('❌ API test failed:', error);
  }
};

// Run in browser console for testing
// testAPIIntegration();
```

## 🎛️ Control Features

### Job Management Controls
```jsx
<JobControls>
  <ControlGroup label="Job Management">
    <Button 
      variant="primary" 
      onClick={pauseAllJobs}
      disabled={!hasRunningJobs}
    >
      Pause All
    </Button>
    <Button 
      variant="primary" 
      onClick={resumeAllJobs}
      disabled={!hasPausedJobs}
    >
      Resume All
    </Button>
    <Button 
      variant="secondary" 
      onClick={showJobQueue}
    >
      Queue Manager
    </Button>
  </ControlGroup>
  
  <ControlGroup label="GPU Management">
    <Button 
      variant="warning" 
      onClick={restartAllGPUs}
      confirmMessage="Restart all GPUs? This will interrupt current work."
    >
      Restart All GPUs
    </Button>
    <Button 
      variant="secondary" 
      onClick={clearGPUMemory}
    >
      Clear GPU Memory
    </Button>
  </ControlGroup>
  
  <ControlGroup label="Emergency">
    <Button 
      variant="danger" 
      onClick={emergencyStop}
      confirmMessage="Emergency stop will halt all processing immediately!"
    >
      Emergency Stop
    </Button>
  </ControlGroup>
</JobControls>
```

### Settings Panel
```jsx
<SettingsPanel>
  <Setting label="Update Frequency">
    <Select 
      value={updateFrequency} 
      onChange={setUpdateFrequency}
      options={[
        { value: 1000, label: '1 second' },
        { value: 5000, label: '5 seconds' },
        { value: 10000, label: '10 seconds' }
      ]}
    />
  </Setting>
  
  <Setting label="Chart History Length">
    <Select 
      value={chartHistoryLength} 
      onChange={setChartHistoryLength}
      options={[
        { value: 100, label: '100 points' },
        { value: 500, label: '500 points' },
        { value: 1000, label: '1000 points' }
      ]}
    />
  </Setting>
  
  <Setting label="Notifications">
    <Switch 
      checked={notificationsEnabled} 
      onChange={setNotificationsEnabled}
      label="Enable browser notifications"
    />
  </Setting>
</SettingsPanel>
```

## 📊 Data Update Strategy

### Update Frequencies
- **System Status**: 2 seconds
- **GPU Metrics**: 5 seconds  
- **Score Updates**: Real-time (WebSocket)
- **Charts**: 5-second rolling updates
- **Job Progress**: 10 seconds

### Performance Optimization
```javascript
// Throttle updates to prevent UI blocking
const throttledUpdate = useCallback(
  throttle((data) => {
    updateDashboard(data);
  }, 1000),
  []
);

// Batch chart updates
const batchChartUpdates = useBatch(() => {
  updateAllCharts();
}, 5000);

// Virtualize large lists
const VirtualizedJobList = memo(({ jobs }) => (
  <FixedSizeList
    height={400}
    itemCount={jobs.length}
    itemSize={60}
  >
    {JobItem}
  </FixedSizeList>
));
```

## 🎨 UI/UX Requirements

### Visual Design
- **Dark Theme**: Easy on eyes for monitoring
- **Color-coded Status**: Intuitive status understanding
- **Responsive Layout**: Works on different screen sizes
- **Accessibility**: Screen reader compatible

### User Experience
- **Real-time Updates**: No manual refresh needed
- **Non-blocking**: Never interferes with backend processing
- **Intuitive Controls**: Clear action buttons
- **Error Handling**: Graceful degradation on connection issues

### Notification System
```jsx
<NotificationSystem>
  <Notification 
    type="error" 
    message="GPU 5 encountered an error"
    action={() => restartGPU(5)}
    actionLabel="Restart GPU 5"
  />
  <Notification 
    type="success" 
    message="Job completed with best score: 0.901"
    dismissible={true}
  />
  <Notification 
    type="warning" 
    message="Memory usage high on GPU 2"
    persistent={true}
  />
</NotificationSystem>
```

## 🔧 Technical Stack Recommendations

### Frontend Framework
- **React** with TypeScript for type safety
- **Material-UI** or **Chakra UI** for components
- **Recharts** or **Chart.js** for real-time charts
- **React Query** for API state management

### Real-time Communication
- **Socket.IO** for WebSocket management
- **React-use-websocket** for React integration

### State Management
- **Zustand** for simple global state
- **React Query** for server state

### Build & Development
- **Vite** for fast development and building
- **ESLint + Prettier** for code quality

## 🚀 Development Priority

### Phase 1: Core Monitoring
1. Basic system status display
2. GPU grid with real-time status
3. WebSocket integration for live updates

### Phase 2: Job Management
1. Job control buttons (pause/resume)
2. Progress tracking
3. Basic charts

### Phase 3: Advanced Features
1. Strategy effectiveness heatmap
2. Advanced analytics
3. Settings and configuration

### Phase 4: Polish & Optimization
1. Performance optimization
2. Error handling
3. Accessibility improvements

---

This dashboard specification provides comprehensive monitoring capabilities while ensuring the frontend never impacts the performance of the distributed RL processing system.
