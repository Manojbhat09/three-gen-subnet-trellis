# Dashboard Frontend - Detailed Implementation Guide

## 🎨 Complete React Component Architecture

### Project Structure
```
dashboard/
├── src/
│   ├── components/
│   │   ├── SystemPanel/
│   │   │   ├── JobQueueStatus.tsx
│   │   │   ├── PerformanceSummary.tsx
│   │   │   └── SystemHealth.tsx
│   │   ├── GPUGrid/
│   │   │   ├── GPUCard.tsx
│   │   │   ├── GPUMetrics.tsx
│   │   │   └── GPUGrid.tsx
│   │   ├── ProgressPanel/
│   │   │   ├── OverallProgress.tsx
│   │   │   ├── ActiveJobsList.tsx
│   │   │   └── JobControls.tsx
│   │   ├── MetricsPanel/
│   │   │   ├── ScoreChart.tsx
│   │   │   ├── UtilizationChart.tsx
│   │   │   ├── StrategyHeatmap.tsx
│   │   │   └── MetricsPanel.tsx
│   │   └── Controls/
│   │       ├── JobManagement.tsx
│   │       ├── GPUManagement.tsx
│   │       └── EmergencyControls.tsx
│   ├── hooks/
│   │   ├── useWebSocket.ts
│   │   ├── useGPUStatus.ts
│   │   ├── useJobProgress.ts
│   │   └── useMetrics.ts
│   ├── services/
│   │   ├── api.ts
│   │   ├── websocket.ts
│   │   └── metrics.ts
│   ├── store/
│   │   ├── systemStore.ts
│   │   ├── gpuStore.ts
│   │   └── jobStore.ts
│   ├── types/
│   │   ├── gpu.types.ts
│   │   ├── job.types.ts
│   │   └── metrics.types.ts
│   └── App.tsx
```

## 📦 Core Component Implementations

### 1. Main Dashboard Component (`App.tsx`)

```tsx
import React, { useEffect, useState } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box, Grid, Paper, Container } from '@mui/material';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Toaster } from 'react-hot-toast';

// Components
import SystemPanel from './components/SystemPanel';
import GPUGrid from './components/GPUGrid';
import ProgressPanel from './components/ProgressPanel';
import MetricsPanel from './components/MetricsPanel';
import ControlBar from './components/Controls/ControlBar';

// Hooks
import { useWebSocketConnection } from './hooks/useWebSocket';
import { useSystemStatus } from './hooks/useSystemStatus';

// Theme
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#2196f3',
    },
    secondary: {
      main: '#4caf50',
    },
    error: {
      main: '#f44336',
    },
    warning: {
      main: '#ff9800',
    },
    background: {
      default: '#0a0e27',
      paper: '#1a1f3a',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
  },
});

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchInterval: 5000, // Refetch every 5 seconds
      staleTime: 3000,
    },
  },
});

function App() {
  const [selectedJob, setSelectedJob] = useState<string | null>(null);
  const { wsStatus, wsData } = useWebSocketConnection();
  const { systemStatus, isLoading } = useSystemStatus();

  useEffect(() => {
    // Initialize dashboard
    console.log('Dashboard initialized with WebSocket status:', wsStatus);
  }, [wsStatus]);

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={darkTheme}>
        <CssBaseline />
        <Toaster 
          position="top-right"
          toastOptions={{
            style: {
              background: '#1a1f3a',
              color: '#fff',
            },
          }}
        />
        
        <Box sx={{ flexGrow: 1, minHeight: '100vh', bgcolor: 'background.default' }}>
          {/* Control Bar */}
          <ControlBar />
          
          <Container maxWidth={false} sx={{ py: 3 }}>
            <Grid container spacing={3}>
              {/* Left Panel - System Status */}
              <Grid item xs={12} md={3}>
                <SystemPanel 
                  systemStatus={systemStatus}
                  isLoading={isLoading}
                />
              </Grid>
              
              {/* Center - GPU Grid */}
              <Grid item xs={12} md={6}>
                <GPUGrid 
                  wsData={wsData}
                  onGPUSelect={(gpuId) => console.log('Selected GPU:', gpuId)}
                />
              </Grid>
              
              {/* Right Panel - Job Progress */}
              <Grid item xs={12} md={3}>
                <ProgressPanel 
                  selectedJob={selectedJob}
                  onJobSelect={setSelectedJob}
                />
              </Grid>
              
              {/* Bottom - Metrics */}
              <Grid item xs={12}>
                <MetricsPanel 
                  selectedJob={selectedJob}
                  wsData={wsData}
                />
              </Grid>
            </Grid>
          </Container>
        </Box>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
```

### 2. GPU Card Component (`GPUCard.tsx`)

```tsx
import React, { useMemo } from 'react';
import {
  Card,
  CardContent,
  CardActions,
  Typography,
  LinearProgress,
  Box,
  Chip,
  IconButton,
  Tooltip,
  styled,
} from '@mui/material';
import {
  RestartAlt,
  Info,
  Memory,
  Thermostat,
  Speed,
} from '@mui/icons-material';
import { motion } from 'framer-motion';

interface GPUCardProps {
  gpuId: number;
  status: 'idle' | 'busy' | 'error' | 'recovery';
  port: number;
  currentJob?: string;
  currentPrompt?: string;
  currentRound?: number;
  maxRounds?: number;
  currentBestScore?: number;
  memoryUsed: number;
  memoryTotal: number;
  utilization: number;
  temperature: number;
  errorCount: number;
  onRestart: (gpuId: number) => void;
  onShowDetails: (gpuId: number) => void;
}

const StyledCard = styled(Card)<{ status: string }>(({ theme, status }) => ({
  position: 'relative',
  background: theme.palette.background.paper,
  border: `2px solid`,
  borderColor: 
    status === 'busy' ? theme.palette.success.main :
    status === 'idle' ? theme.palette.info.main :
    status === 'error' ? theme.palette.error.main :
    theme.palette.warning.main,
  transition: 'all 0.3s ease',
  '&:hover': {
    transform: 'translateY(-4px)',
    boxShadow: theme.shadows[8],
  },
}));

const StatusBadge = styled(Chip)<{ status: string }>(({ theme, status }) => ({
  position: 'absolute',
  top: 8,
  right: 8,
  fontWeight: 'bold',
  backgroundColor:
    status === 'busy' ? theme.palette.success.main :
    status === 'idle' ? theme.palette.info.main :
    status === 'error' ? theme.palette.error.main :
    theme.palette.warning.main,
}));

const MetricBox = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(1),
  marginBottom: theme.spacing(1),
}));

export const GPUCard: React.FC<GPUCardProps> = ({
  gpuId,
  status,
  port,
  currentJob,
  currentPrompt,
  currentRound,
  maxRounds,
  currentBestScore,
  memoryUsed,
  memoryTotal,
  utilization,
  temperature,
  errorCount,
  onRestart,
  onShowDetails,
}) => {
  const memoryPercentage = useMemo(
    () => (memoryUsed / memoryTotal) * 100,
    [memoryUsed, memoryTotal]
  );

  const progressPercentage = useMemo(
    () => currentRound && maxRounds ? (currentRound / maxRounds) * 100 : 0,
    [currentRound, maxRounds]
  );

  const temperatureColor = useMemo(() => {
    if (temperature < 60) return 'success';
    if (temperature < 75) return 'warning';
    return 'error';
  }, [temperature]);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      <StyledCard status={status}>
        <StatusBadge
          label={status.toUpperCase()}
          status={status}
          size="small"
        />
        
        <CardContent>
          <Typography variant="h6" gutterBottom>
            GPU {gpuId}
            <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
              Port: {port}
            </Typography>
          </Typography>

          {/* Metrics */}
          <MetricBox>
            <Memory fontSize="small" />
            <Box sx={{ flex: 1 }}>
              <Typography variant="caption" color="text.secondary">
                Memory: {memoryUsed.toFixed(1)}GB / {memoryTotal}GB
              </Typography>
              <LinearProgress
                variant="determinate"
                value={memoryPercentage}
                color={memoryPercentage > 90 ? 'error' : 'primary'}
                sx={{ height: 6, borderRadius: 1 }}
              />
            </Box>
          </MetricBox>

          <MetricBox>
            <Speed fontSize="small" />
            <Box sx={{ flex: 1 }}>
              <Typography variant="caption" color="text.secondary">
                Utilization: {utilization}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={utilization}
                color="secondary"
                sx={{ height: 6, borderRadius: 1 }}
              />
            </Box>
          </MetricBox>

          <MetricBox>
            <Thermostat fontSize="small" />
            <Typography variant="caption" color="text.secondary">
              Temperature: 
            </Typography>
            <Chip
              label={`${temperature}°C`}
              size="small"
              color={temperatureColor}
              variant="outlined"
            />
          </MetricBox>

          {/* Current Task */}
          {status === 'busy' && currentJob && (
            <Box sx={{ mt: 2, p: 1, bgcolor: 'background.default', borderRadius: 1 }}>
              <Typography variant="caption" color="text.secondary">
                Current Task
              </Typography>
              <Typography variant="body2" noWrap>
                {currentPrompt?.substring(0, 30)}...
              </Typography>
              
              {currentRound && maxRounds && (
                <>
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                    Round {currentRound} / {maxRounds}
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={progressPercentage}
                    sx={{ height: 8, borderRadius: 1, mt: 0.5 }}
                  />
                </>
              )}
              
              {currentBestScore !== undefined && (
                <Typography variant="body2" sx={{ mt: 1 }}>
                  Best Score: <strong>{currentBestScore.toFixed(4)}</strong>
                </Typography>
              )}
            </Box>
          )}

          {/* Error Count */}
          {errorCount > 0 && (
            <Chip
              label={`Errors: ${errorCount}`}
              size="small"
              color="error"
              variant="outlined"
              sx={{ mt: 1 }}
            />
          )}
        </CardContent>

        <CardActions>
          <Tooltip title="Restart GPU">
            <IconButton
              size="small"
              onClick={() => onRestart(gpuId)}
              disabled={status === 'busy'}
            >
              <RestartAlt />
            </IconButton>
          </Tooltip>
          <Tooltip title="View Details">
            <IconButton size="small" onClick={() => onShowDetails(gpuId)}>
              <Info />
            </IconButton>
          </Tooltip>
        </CardActions>
      </StyledCard>
    </motion.div>
  );
};
```

### 3. Real-time Score Chart Component (`ScoreChart.tsx`)

```tsx
import React, { useEffect, useRef, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { Paper, Typography, Box, ToggleButton, ToggleButtonGroup } from '@mui/material';
import { useTheme } from '@mui/material/styles';

interface ScoreDataPoint {
  timestamp: string;
  gpu0?: number;
  gpu1?: number;
  gpu2?: number;
  gpu3?: number;
  gpu4?: number;
  gpu5?: number;
  gpu6?: number;
  gpu7?: number;
  average?: number;
}

interface ScoreChartProps {
  data: ScoreDataPoint[];
  targetScore: number;
  timeRange: '1m' | '5m' | '15m' | '1h';
  onTimeRangeChange: (range: string) => void;
}

export const ScoreChart: React.FC<ScoreChartProps> = ({
  data,
  targetScore,
  timeRange,
  onTimeRangeChange,
}) => {
  const theme = useTheme();
  const [animationKey, setAnimationKey] = useState(0);
  
  // GPU colors
  const gpuColors = [
    '#FF6B6B', // GPU 0 - Red
    '#4ECDC4', // GPU 1 - Teal
    '#45B7D1', // GPU 2 - Blue
    '#96CEB4', // GPU 3 - Green
    '#FFEAA7', // GPU 4 - Yellow
    '#DDA0DD', // GPU 5 - Plum
    '#98D8C8', // GPU 6 - Mint
    '#F7DC6F', // GPU 7 - Gold
  ];

  // Filter data based on time range
  const filteredData = React.useMemo(() => {
    const now = Date.now();
    const ranges = {
      '1m': 60 * 1000,
      '5m': 5 * 60 * 1000,
      '15m': 15 * 60 * 1000,
      '1h': 60 * 60 * 1000,
    };
    
    const cutoff = now - ranges[timeRange];
    
    return data.filter(point => {
      const pointTime = new Date(point.timestamp).getTime();
      return pointTime >= cutoff;
    });
  }, [data, timeRange]);

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <Paper sx={{ p: 1, bgcolor: 'background.paper', border: 1, borderColor: 'divider' }}>
          <Typography variant="caption" color="text.secondary">
            {new Date(label).toLocaleTimeString()}
          </Typography>
          {payload.map((entry: any, index: number) => (
            <Typography
              key={index}
              variant="body2"
              style={{ color: entry.color }}
            >
              {entry.name}: {entry.value?.toFixed(4)}
            </Typography>
          ))}
        </Paper>
      );
    }
    return null;
  };

  // Animate on data change
  useEffect(() => {
    setAnimationKey(prev => prev + 1);
  }, [data.length]);

  return (
    <Paper sx={{ p: 2, height: '100%' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Score Progression</Typography>
        <ToggleButtonGroup
          value={timeRange}
          exclusive
          onChange={(e, value) => value && onTimeRangeChange(value)}
          size="small"
        >
          <ToggleButton value="1m">1m</ToggleButton>
          <ToggleButton value="5m">5m</ToggleButton>
          <ToggleButton value="15m">15m</ToggleButton>
          <ToggleButton value="1h">1h</ToggleButton>
        </ToggleButtonGroup>
      </Box>
      
      <ResponsiveContainer width="100%" height={300}>
        <LineChart
          data={filteredData}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          key={animationKey}
        >
          <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
          <XAxis
            dataKey="timestamp"
            tickFormatter={(value) => new Date(value).toLocaleTimeString()}
            stroke={theme.palette.text.secondary}
          />
          <YAxis
            domain={[0, 1]}
            stroke={theme.palette.text.secondary}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          
          {/* Target score reference line */}
          <ReferenceLine
            y={targetScore}
            stroke={theme.palette.warning.main}
            strokeDasharray="5 5"
            label="Target"
          />
          
          {/* GPU lines */}
          {[0, 1, 2, 3, 4, 5, 6, 7].map(gpuId => (
            <Line
              key={`gpu${gpuId}`}
              type="monotone"
              dataKey={`gpu${gpuId}`}
              stroke={gpuColors[gpuId]}
              strokeWidth={2}
              dot={false}
              animationDuration={500}
              name={`GPU ${gpuId}`}
            />
          ))}
          
          {/* Average line */}
          <Line
            type="monotone"
            dataKey="average"
            stroke={theme.palette.primary.main}
            strokeWidth={3}
            dot={false}
            strokeDasharray="5 5"
            name="Average"
          />
        </LineChart>
      </ResponsiveContainer>
    </Paper>
  );
};
```

### 4. WebSocket Hook (`useWebSocket.ts`)

```typescript
import { useEffect, useRef, useState, useCallback } from 'react';
import { toast } from 'react-hot-toast';

export interface WebSocketMessage {
  type: 'system_update' | 'score_update' | 'alert' | 'job_event';
  timestamp: string;
  data: any;
}

interface UseWebSocketReturn {
  wsStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  wsData: WebSocketMessage | null;
  sendMessage: (message: any) => void;
  reconnect: () => void;
}

export const useWebSocketConnection = (
  url: string = 'ws://localhost:8090/ws/real_time_updates'
): UseWebSocketReturn => {
  const [wsStatus, setWsStatus] = useState<UseWebSocketReturn['wsStatus']>('connecting');
  const [wsData, setWsData] = useState<WebSocketMessage | null>(null);
  const ws = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);

  const connect = useCallback(() => {
    try {
      ws.current = new WebSocket(url);
      
      ws.current.onopen = () => {
        setWsStatus('connected');
        reconnectAttempts.current = 0;
        toast.success('Connected to server');
        console.log('WebSocket connected');
      };

      ws.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          setWsData(message);
          
          // Handle different message types
          switch (message.type) {
            case 'alert':
              handleAlert(message.data);
              break;
            case 'job_event':
              handleJobEvent(message.data);
              break;
            case 'score_update':
              handleScoreUpdate(message.data);
              break;
            default:
              // Regular updates
              break;
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.current.onerror = (error) => {
        setWsStatus('error');
        console.error('WebSocket error:', error);
        toast.error('Connection error');
      };

      ws.current.onclose = () => {
        setWsStatus('disconnected');
        console.log('WebSocket disconnected');
        
        // Attempt reconnection
        if (reconnectAttempts.current < 5) {
          reconnectAttempts.current++;
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
          
          toast.loading(`Reconnecting... (Attempt ${reconnectAttempts.current}/5)`);
          
          reconnectTimeout.current = setTimeout(() => {
            connect();
          }, delay);
        } else {
          toast.error('Failed to connect to server');
        }
      };
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      setWsStatus('error');
    }
  }, [url]);

  const sendMessage = useCallback((message: any) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected');
      toast.error('Cannot send message: Not connected');
    }
  }, []);

  const reconnect = useCallback(() => {
    if (ws.current) {
      ws.current.close();
    }
    reconnectAttempts.current = 0;
    connect();
  }, [connect]);

  // Handle specific message types
  const handleAlert = (alert: any) => {
    const { level, message, action_required } = alert;
    
    switch (level) {
      case 'error':
        toast.error(message, { duration: 10000 });
        break;
      case 'warning':
        toast(message, { icon: '⚠️', duration: 5000 });
        break;
      case 'info':
        toast(message, { icon: 'ℹ️' });
        break;
    }
    
    if (action_required) {
      // Could trigger a modal or special UI element here
      console.log('Action required:', alert.suggested_action);
    }
  };

  const handleJobEvent = (event: any) => {
    const { event: eventType, job_id, status } = event;
    
    switch (eventType) {
      case 'started':
        toast.success(`Job ${job_id} started`);
        break;
      case 'completed':
        toast.success(`Job ${job_id} completed successfully`, { duration: 10000 });
        break;
      case 'failed':
        toast.error(`Job ${job_id} failed`);
        break;
    }
  };

  const handleScoreUpdate = (update: any) => {
    const { gpu_id, new_score, improvement } = update;
    
    if (improvement > 0.1) {
      toast.success(`GPU ${gpu_id}: Major improvement! +${improvement.toFixed(3)}`);
    }
  };

  useEffect(() => {
    connect();
    
    return () => {
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
      }
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [connect]);

  return {
    wsStatus,
    wsData,
    sendMessage,
    reconnect,
  };
};
```

### 5. Strategy Heatmap Component (`StrategyHeatmap.tsx`)

```tsx
import React, { useMemo } from 'react';
import { Paper, Typography, Box } from '@mui/material';
import { scaleSequential } from 'd3-scale';
import { interpolateRdYlGn } from 'd3-scale-chromatic';
import { Tooltip } from '@mui/material';

interface StrategyData {
  strategies: string[];
  gpus: number[];
  effectiveness: { [key: string]: { [key: number]: number } };
}

interface StrategyHeatmapProps {
  data: StrategyData;
}

export const StrategyHeatmap: React.FC<StrategyHeatmapProps> = ({ data }) => {
  const { strategies, gpus, effectiveness } = data;
  
  // Color scale
  const colorScale = useMemo(
    () => scaleSequential(interpolateRdYlGn).domain([0, 1]),
    []
  );
  
  // Calculate cell size
  const cellSize = 40;
  const labelWidth = 150;
  const labelHeight = 30;
  
  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Strategy Effectiveness Heatmap
      </Typography>
      
      <Box sx={{ overflowX: 'auto' }}>
        <svg
          width={labelWidth + gpus.length * cellSize + 20}
          height={labelHeight + strategies.length * cellSize + 20}
        >
          {/* GPU labels */}
          {gpus.map((gpu, i) => (
            <text
              key={`gpu-label-${gpu}`}
              x={labelWidth + i * cellSize + cellSize / 2}
              y={labelHeight - 5}
              textAnchor="middle"
              fontSize="12"
              fill="#999"
            >
              GPU {gpu}
            </text>
          ))}
          
          {/* Strategy labels and cells */}
          {strategies.map((strategy, strategyIdx) => (
            <g key={`strategy-${strategy}`}>
              <text
                x={labelWidth - 10}
                y={labelHeight + strategyIdx * cellSize + cellSize / 2 + 5}
                textAnchor="end"
                fontSize="12"
                fill="#999"
              >
                {strategy}
              </text>
              
              {/* Effectiveness cells */}
              {gpus.map((gpu, gpuIdx) => {
                const value = effectiveness[strategy]?.[gpu] || 0;
                const color = colorScale(value);
                
                return (
                  <Tooltip
                    key={`cell-${strategy}-${gpu}`}
                    title={`${strategy} on GPU ${gpu}: ${(value * 100).toFixed(1)}%`}
                    arrow
                  >
                    <rect
                      x={labelWidth + gpuIdx * cellSize}
                      y={labelHeight + strategyIdx * cellSize}
                      width={cellSize - 2}
                      height={cellSize - 2}
                      fill={color}
                      stroke="#1a1f3a"
                      strokeWidth="1"
                      style={{ cursor: 'pointer' }}
                      onMouseEnter={(e) => {
                        e.currentTarget.setAttribute('stroke', '#fff');
                        e.currentTarget.setAttribute('stroke-width', '2');
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.setAttribute('stroke', '#1a1f3a');
                        e.currentTarget.setAttribute('stroke-width', '1');
                      }}
                    />
                  </Tooltip>
                );
              })}
            </g>
          ))}
          
          {/* Legend */}
          <defs>
            <linearGradient id="legend-gradient">
              {Array.from({ length: 11 }, (_, i) => i / 10).map(value => (
                <stop
                  key={value}
                  offset={`${value * 100}%`}
                  stopColor={colorScale(value)}
                />
              ))}
            </linearGradient>
          </defs>
          
          <rect
            x={labelWidth}
            y={labelHeight + strategies.length * cellSize + 10}
            width={gpus.length * cellSize}
            height={15}
            fill="url(#legend-gradient)"
          />
          
          <text
            x={labelWidth}
            y={labelHeight + strategies.length * cellSize + 40}
            fontSize="10"
            fill="#999"
          >
            0%
          </text>
          
          <text
            x={labelWidth + gpus.length * cellSize}
            y={labelHeight + strategies.length * cellSize + 40}
            textAnchor="end"
            fontSize="10"
            fill="#999"
          >
            100%
          </text>
        </svg>
      </Box>
      
      <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
        Effectiveness shows success rate of each strategy on each GPU
      </Typography>
    </Paper>
  );
};
```

### 6. Job Management Controls (`JobManagement.tsx`)

```tsx
import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Box,
  Typography,
  Slider,
  Chip,
  Stack,
} from '@mui/material';
import { useJobSubmission } from '../hooks/useJobSubmission';
import { toast } from 'react-hot-toast';

interface JobSubmissionDialogProps {
  open: boolean;
  onClose: () => void;
}

export const JobSubmissionDialog: React.FC<JobSubmissionDialogProps> = ({
  open,
  onClose,
}) => {
  const { submitJob, isSubmitting } = useJobSubmission();
  
  const [jobData, setJobData] = useState({
    prompts: '',
    targetScore: 0.85,
    maxEpisodes: 10,
    maxRounds: 12,
    improvementThreshold: 0.03,
    priority: 1,
    jobName: '',
  });
  
  const [promptCount, setPromptCount] = useState(0);
  
  const handlePromptsChange = (value: string) => {
    setJobData({ ...jobData, prompts: value });
    const lines = value.trim().split('\n').filter(line => line.trim());
    setPromptCount(lines.length);
  };
  
  const handleSubmit = async () => {
    if (promptCount === 0) {
      toast.error('Please enter at least one prompt');
      return;
    }
    
    const prompts = jobData.prompts
      .trim()
      .split('\n')
      .filter(line => line.trim())
      .map(line => line.trim());
    
    try {
      const result = await submitJob({
        ...jobData,
        prompts,
      });
      
      toast.success(`Job ${result.job_id} submitted successfully`);
      onClose();
    } catch (error) {
      toast.error('Failed to submit job');
    }
  };
  
  const estimatedTime = useMemo(() => {
    // Rough estimation: 8 minutes per prompt per episode / 8 GPUs
    const minutesPerPrompt = 8;
    const totalMinutes = (promptCount * jobData.maxEpisodes * minutesPerPrompt) / 8;
    
    if (totalMinutes < 60) {
      return `${Math.round(totalMinutes)} minutes`;
    } else {
      const hours = Math.floor(totalMinutes / 60);
      const minutes = Math.round(totalMinutes % 60);
      return `${hours}h ${minutes}m`;
    }
  }, [promptCount, jobData.maxEpisodes]);
  
  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Submit New Optimization Job</DialogTitle>
      
      <DialogContent>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
          {/* Job Name */}
          <TextField
            label="Job Name"
            value={jobData.jobName}
            onChange={(e) => setJobData({ ...jobData, jobName: e.target.value })}
            fullWidth
            placeholder="e.g., Vehicle Optimization Batch 1"
          />
          
          {/* Prompts */}
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Prompts (one per line)
            </Typography>
            <TextField
              multiline
              rows={6}
              value={jobData.prompts}
              onChange={(e) => handlePromptsChange(e.target.value)}
              fullWidth
              placeholder="a red sports car&#10;a blue ceramic vase&#10;a wooden table"
            />
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
              <Chip label={`${promptCount} prompts`} size="small" />
              <Chip label={`Est. time: ${estimatedTime}`} size="small" color="info" />
            </Box>
          </Box>
          
          {/* Target Score */}
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Target Score: {jobData.targetScore.toFixed(2)}
            </Typography>
            <Slider
              value={jobData.targetScore}
              onChange={(e, value) => setJobData({ ...jobData, targetScore: value as number })}
              min={0.5}
              max={1.0}
              step={0.05}
              marks={[
                { value: 0.5, label: '0.5' },
                { value: 0.75, label: '0.75' },
                { value: 1.0, label: '1.0' },
              ]}
            />
          </Box>
          
          {/* Episodes */}
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Max Episodes: {jobData.maxEpisodes}
            </Typography>
            <Slider
              value={jobData.maxEpisodes}
              onChange={(e, value) => setJobData({ ...jobData, maxEpisodes: value as number })}
              min={1}
              max={50}
              step={1}
              marks={[
                { value: 1, label: '1' },
                { value: 10, label: '10' },
                { value: 25, label: '25' },
                { value: 50, label: '50' },
              ]}
            />
          </Box>
          
          {/* Rounds per Episode */}
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Max Rounds per Episode: {jobData.maxRounds}
            </Typography>
            <Slider
              value={jobData.maxRounds}
              onChange={(e, value) => setJobData({ ...jobData, maxRounds: value as number })}
              min={1}
              max={20}
              step={1}
              marks={[
                { value: 1, label: '1' },
                { value: 5, label: '5' },
                { value: 12, label: '12' },
                { value: 20, label: '20' },
              ]}
            />
          </Box>
          
          {/* Priority */}
          <FormControl>
            <InputLabel>Priority</InputLabel>
            <Select
              value={jobData.priority}
              onChange={(e) => setJobData({ ...jobData, priority: e.target.value as number })}
              label="Priority"
            >
              <MenuItem value={1}>Low</MenuItem>
              <MenuItem value={2}>Normal</MenuItem>
              <MenuItem value={3}>High</MenuItem>
              <MenuItem value={4}>Critical</MenuItem>
            </Select>
          </FormControl>
          
          {/* Summary */}
          <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
            <Typography variant="subtitle2" gutterBottom>
              Job Summary
            </Typography>
            <Stack direction="row" spacing={1} flexWrap="wrap">
              <Chip label={`${promptCount} prompts`} size="small" />
              <Chip label={`${jobData.maxEpisodes} episodes`} size="small" />
              <Chip label={`${jobData.maxRounds} rounds/episode`} size="small" />
              <Chip label={`Target: ${jobData.targetScore}`} size="small" color="primary" />
              <Chip label={`Est: ${estimatedTime}`} size="small" color="info" />
            </Stack>
          </Paper>
        </Box>
      </DialogContent>
      
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          onClick={handleSubmit}
          variant="contained"
          disabled={isSubmitting || promptCount === 0}
        >
          {isSubmitting ? 'Submitting...' : 'Submit Job'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};
```

## 🎨 Advanced UI Features

### Dark Mode Theme Configuration

```typescript
// theme/darkTheme.ts
export const createDarkTheme = () => createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#2196f3',
      light: '#64b5f6',
      dark: '#1976d2',
    },
    secondary: {
      main: '#4caf50',
      light: '#81c784',
      dark: '#388e3c',
    },
    error: {
      main: '#f44336',
      light: '#ef5350',
      dark: '#d32f2f',
    },
    warning: {
      main: '#ff9800',
      light: '#ffb74d',
      dark: '#f57c00',
    },
    info: {
      main: '#00bcd4',
      light: '#4dd0e1',
      dark: '#0097a7',
    },
    success: {
      main: '#4caf50',
      light: '#81c784',
      dark: '#388e3c',
    },
    background: {
      default: '#0a0e27',
      paper: '#1a1f3a',
    },
    text: {
      primary: '#ffffff',
      secondary: '#b0b0b0',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: { fontWeight: 700 },
    h2: { fontWeight: 600 },
    h3: { fontWeight: 600 },
    h4: { fontWeight: 600 },
    h5: { fontWeight: 500 },
    h6: { fontWeight: 500 },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          borderRadius: 12,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 500,
        },
      },
    },
  },
});
```

## 📱 Responsive Design

The dashboard automatically adapts to different screen sizes:
- **Desktop (>1200px)**: Full 3-column layout with all panels visible
- **Tablet (768-1200px)**: 2-column layout with collapsible side panels
- **Mobile (<768px)**: Single column with tab navigation

## 🚀 Performance Optimizations

1. **Virtual Scrolling**: Large lists use react-window for virtualization
2. **Memoization**: Heavy computations cached with useMemo
3. **Throttled Updates**: WebSocket updates throttled to prevent UI blocking
4. **Lazy Loading**: Charts and heavy components loaded on demand
5. **Request Deduplication**: API calls deduplicated with React Query

This comprehensive dashboard implementation provides a production-ready monitoring interface for the distributed RL system with real-time updates, advanced visualizations, and intuitive controls.




