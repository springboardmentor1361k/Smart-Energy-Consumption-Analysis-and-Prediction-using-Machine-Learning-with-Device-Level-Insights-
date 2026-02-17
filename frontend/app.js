// Smart Energy Dashboard - Main Application Script

// Configuration
const CONFIG = {
    API_BASE_URL: '/api',
    UPDATE_INTERVAL: 5000, // 5 seconds
    CHART_COLORS: {
        primary: '#00d4ff',
        secondary: '#ff00ff',
        accent: '#ffeb3b',
        success: '#00ff88',
        warning: '#ff9500',
        danger: '#ff3366'
    }
};

// Global state
let currentView = 'dashboard';
let charts = {};
let updateIntervals = {};

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    initializeNavigation();
    initializeCharts();
    initializeDevices();
    initializeHourlyPredictions();
    initializeHeatmap();
    startRealTimeUpdates();
    
    // Hide loading overlay
    setTimeout(() => {
        document.getElementById('loadingOverlay').classList.remove('active');
    }, 1500);
});

// Navigation
function initializeNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const view = btn.dataset.view;
            switchView(view);
            
            // Update active state
            navButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });
}

function switchView(view) {
    // Hide all views
    document.querySelectorAll('.view-container').forEach(container => {
        container.classList.remove('active');
    });
    
    // Show selected view
    document.getElementById(`${view}-view`).classList.add('active');
    currentView = view;
    
    // Refresh charts in the new view
    setTimeout(() => {
        Object.values(charts).forEach(chart => chart.resize && chart.resize());
    }, 100);
}

// Chart Initialization
function initializeCharts() {
    initializeRealtimeChart();
    initializeDevicePieChart();
    initializeTrendsChart();
    initializeDeviceComparisonChart();
    initializeForecastChart();
    initializeModelPerformanceChart();
    initializeCostProjectionChart();
}

// Real-time Energy Flow Chart
function initializeRealtimeChart() {
    const ctx = document.getElementById('realtimeChart');
    if (!ctx) return;
    
    const gradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(0, 212, 255, 0.4)');
    gradient.addColorStop(1, 'rgba(0, 212, 255, 0.0)');
    
    const now = new Date();
    const labels = [];
    for (let i = 24; i >= 0; i--) {
        const time = new Date(now - i * 3600000);
        labels.push(time.getHours() + ':00');
    }
    
    charts.realtime = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Power Consumption (kW)',
                data: generateRealisticData(25, 1.5, 3.5),
                borderColor: CONFIG.CHART_COLORS.primary,
                backgroundColor: gradient,
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 6,
                pointHoverBackgroundColor: CONFIG.CHART_COLORS.primary,
                pointHoverBorderColor: '#fff',
                pointHoverBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(26, 31, 53, 0.95)',
                    titleColor: '#00d4ff',
                    bodyColor: '#fff',
                    borderColor: '#00d4ff',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false,
                    callbacks: {
                        label: function(context) {
                            return `${context.parsed.y.toFixed(2)} kW`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(0, 212, 255, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#6b7589',
                        font: {
                            family: 'Orbitron'
                        }
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(0, 212, 255, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#6b7589',
                        font: {
                            family: 'Orbitron'
                        },
                        callback: function(value) {
                            return value + ' kW';
                        }
                    }
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            }
        }
    });
}

// Device Distribution Pie Chart
function initializeDevicePieChart() {
    const ctx = document.getElementById('devicePieChart');
    if (!ctx) return;
    
    charts.devicePie = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['HVAC', 'Water Heater', 'Refrigerator', 'Lighting', 'Others'],
            datasets: [{
                data: [32, 24, 18, 15, 11],
                backgroundColor: [
                    CONFIG.CHART_COLORS.primary,
                    CONFIG.CHART_COLORS.secondary,
                    CONFIG.CHART_COLORS.accent,
                    CONFIG.CHART_COLORS.success,
                    CONFIG.CHART_COLORS.warning
                ],
                borderColor: '#0a0e1a',
                borderWidth: 3,
                hoverOffset: 10
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(26, 31, 53, 0.95)',
                    titleColor: '#00d4ff',
                    bodyColor: '#fff',
                    borderColor: '#00d4ff',
                    borderWidth: 1,
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.parsed}%`;
                        }
                    }
                }
            },
            cutout: '70%'
        }
    });
}

// Consumption Trends Chart
function initializeTrendsChart() {
    const ctx = document.getElementById('trendsChart');
    if (!ctx) return;
    
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'];
    
    charts.trends = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: months,
            datasets: [
                {
                    label: 'Actual Consumption',
                    data: [720, 680, 750, 800, 820, 790],
                    backgroundColor: 'rgba(0, 212, 255, 0.6)',
                    borderColor: CONFIG.CHART_COLORS.primary,
                    borderWidth: 2,
                    borderRadius: 6
                },
                {
                    label: 'Previous Year',
                    data: [780, 740, 820, 850, 870, 840],
                    backgroundColor: 'rgba(255, 0, 255, 0.3)',
                    borderColor: CONFIG.CHART_COLORS.secondary,
                    borderWidth: 2,
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#a0aec0',
                        font: {
                            family: 'Urbanist',
                            size: 12
                        },
                        padding: 15,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(26, 31, 53, 0.95)',
                    titleColor: '#00d4ff',
                    bodyColor: '#fff',
                    borderColor: '#00d4ff',
                    borderWidth: 1,
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.parsed.y} kWh`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#6b7589',
                        font: {
                            family: 'Urbanist'
                        }
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(0, 212, 255, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#6b7589',
                        font: {
                            family: 'Orbitron'
                        },
                        callback: function(value) {
                            return value + ' kWh';
                        }
                    }
                }
            }
        }
    });
}

// Device Comparison Chart
function initializeDeviceComparisonChart() {
    const ctx = document.getElementById('deviceComparisonChart');
    if (!ctx) return;
    
    charts.deviceComparison = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['HVAC', 'Water Heater', 'Refrigerator', 'Washer', 'Dryer', 'Dishwasher', 'Lighting', 'TV'],
            datasets: [{
                label: 'Daily Average (kWh)',
                data: [9.2, 6.8, 5.1, 2.3, 3.1, 1.8, 4.2, 1.5],
                backgroundColor: function(context) {
                    const value = context.parsed.y;
                    if (value > 7) return 'rgba(255, 51, 102, 0.7)';
                    if (value > 4) return 'rgba(255, 149, 0, 0.7)';
                    return 'rgba(0, 255, 136, 0.7)';
                },
                borderColor: function(context) {
                    const value = context.parsed.y;
                    if (value > 7) return CONFIG.CHART_COLORS.danger;
                    if (value > 4) return CONFIG.CHART_COLORS.warning;
                    return CONFIG.CHART_COLORS.success;
                },
                borderWidth: 2,
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(26, 31, 53, 0.95)',
                    titleColor: '#00d4ff',
                    bodyColor: '#fff',
                    borderColor: '#00d4ff',
                    borderWidth: 1,
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            return `${context.parsed.x.toFixed(1)} kWh/day`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(0, 212, 255, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#6b7589',
                        font: {
                            family: 'Orbitron'
                        }
                    }
                },
                y: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#a0aec0',
                        font: {
                            family: 'Urbanist',
                            size: 11
                        }
                    }
                }
            }
        }
    });
}

// Forecast Chart
function initializeForecastChart() {
    const ctx = document.getElementById('forecastChart');
    if (!ctx) return;
    
    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    const actual = [28.5, 26.8, 29.2, 27.5, 30.1, 35.2, 33.8];
    const predicted = [28.2, 27.1, 28.9, 27.8, 29.8, 34.9, 33.5];
    
    charts.forecast = new Chart(ctx, {
        type: 'line',
        data: {
            labels: days,
            datasets: [
                {
                    label: 'Actual',
                    data: actual,
                    borderColor: CONFIG.CHART_COLORS.primary,
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    borderWidth: 3,
                    tension: 0.4,
                    pointRadius: 5,
                    pointBackgroundColor: CONFIG.CHART_COLORS.primary,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2
                },
                {
                    label: 'LSTM Prediction',
                    data: predicted,
                    borderColor: CONFIG.CHART_COLORS.success,
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    borderWidth: 3,
                    borderDash: [5, 5],
                    tension: 0.4,
                    pointRadius: 5,
                    pointBackgroundColor: CONFIG.CHART_COLORS.success,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2
                },
                {
                    label: 'Confidence Interval',
                    data: predicted.map(v => v + 2),
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    fill: '+1',
                    pointRadius: 0
                },
                {
                    label: 'Confidence Interval Lower',
                    data: predicted.map(v => v - 2),
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    fill: false,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#a0aec0',
                        font: {
                            family: 'Urbanist'
                        },
                        filter: function(item) {
                            return !item.text.includes('Confidence Interval Lower');
                        },
                        usePointStyle: true
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(26, 31, 53, 0.95)',
                    titleColor: '#00d4ff',
                    bodyColor: '#fff',
                    borderColor: '#00d4ff',
                    borderWidth: 1,
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            if (context.dataset.label.includes('Confidence')) return null;
                            return `${context.dataset.label}: ${context.parsed.y.toFixed(1)} kWh`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(0, 212, 255, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#6b7589',
                        font: {
                            family: 'Urbanist'
                        }
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(0, 212, 255, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#6b7589',
                        font: {
                            family: 'Orbitron'
                        },
                        callback: function(value) {
                            return value + ' kWh';
                        }
                    }
                }
            }
        }
    });
}

// Model Performance Chart
function initializeModelPerformanceChart() {
    const ctx = document.getElementById('modelPerformanceChart');
    if (!ctx) return;
    
    const epochs = Array.from({length: 50}, (_, i) => i + 1);
    
    charts.modelPerformance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                {
                    label: 'Training Loss',
                    data: generateLossData(50, 0.8, 0.05),
                    borderColor: CONFIG.CHART_COLORS.primary,
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 0
                },
                {
                    label: 'Validation Loss',
                    data: generateLossData(50, 0.85, 0.08),
                    borderColor: CONFIG.CHART_COLORS.secondary,
                    backgroundColor: 'rgba(255, 0, 255, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#a0aec0',
                        font: {
                            family: 'Urbanist'
                        },
                        usePointStyle: true
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(26, 31, 53, 0.95)',
                    titleColor: '#00d4ff',
                    bodyColor: '#fff',
                    borderColor: '#00d4ff',
                    borderWidth: 1,
                    padding: 12
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(0, 212, 255, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#6b7589',
                        font: {
                            family: 'Orbitron'
                        }
                    },
                    title: {
                        display: true,
                        text: 'Epoch',
                        color: '#a0aec0'
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(0, 212, 255, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#6b7589',
                        font: {
                            family: 'Orbitron'
                        }
                    },
                    title: {
                        display: true,
                        text: 'Loss',
                        color: '#a0aec0'
                    }
                }
            }
        }
    });
}

// Cost Projection Chart
function initializeCostProjectionChart() {
    const ctx = document.getElementById('costProjectionChart');
    if (!ctx) return;
    
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    
    charts.costProjection = new Chart(ctx, {
        type: 'line',
        data: {
            labels: months,
            datasets: [
                {
                    label: 'Current Path',
                    data: [142, 138, 156, 168, 172, 165, 178, 182, 159, 148, 152, 145],
                    borderColor: CONFIG.CHART_COLORS.danger,
                    backgroundColor: 'rgba(255, 51, 102, 0.1)',
                    borderWidth: 3,
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'With Optimizations',
                    data: [82, 78, 91, 95, 98, 89, 102, 105, 87, 79, 84, 81],
                    borderColor: CONFIG.CHART_COLORS.success,
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    borderWidth: 3,
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#a0aec0',
                        font: {
                            family: 'Urbanist'
                        },
                        usePointStyle: true
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(26, 31, 53, 0.95)',
                    titleColor: '#00d4ff',
                    bodyColor: '#fff',
                    borderColor: '#00d4ff',
                    borderWidth: 1,
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: $${context.parsed.y}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(0, 212, 255, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#6b7589',
                        font: {
                            family: 'Urbanist'
                        }
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(0, 212, 255, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#6b7589',
                        font: {
                            family: 'Orbitron'
                        },
                        callback: function(value) {
                            return '$' + value;
                        }
                    }
                }
            }
        }
    });
}

// Device Grid Initialization
function initializeDevices() {
    const devices = [
        { icon: '❄️', name: 'HVAC', power: '2.85 kW', status: 'Active', active: true },
        { icon: '💧', name: 'Water Heater', power: '1.45 kW', status: 'Active', active: true },
        { icon: '🧊', name: 'Refrigerator', power: '0.68 kW', status: 'Active', active: true },
        { icon: '💡', name: 'Lighting', power: '0.42 kW', status: 'Active', active: true },
        { icon: '📺', name: 'TV', power: '0.15 kW', status: 'Active', active: true },
        { icon: '🔌', name: 'Washer', power: '0.00 kW', status: 'Standby', active: false },
        { icon: '🌀', name: 'Dryer', power: '0.00 kW', status: 'Standby', active: false },
        { icon: '🍽️', name: 'Dishwasher', power: '0.00 kW', status: 'Standby', active: false }
    ];
    
    const grid = document.getElementById('devicesGrid');
    if (!grid) return;
    
    grid.innerHTML = devices.map(device => `
        <div class="device-item ${device.active ? 'active' : ''}">
            <div class="device-icon">${device.icon}</div>
            <div class="device-name">${device.name}</div>
            <div class="device-power">${device.power}</div>
            <div class="device-status">${device.status}</div>
        </div>
    `).join('');
    
    // Initialize detailed devices view
    initializeDetailedDevices(devices);
}

function initializeDetailedDevices(devices) {
    const detailedDevices = [
        { icon: '❄️', name: 'Central HVAC', location: 'Whole House', status: 'on', current: '2.85 kW', today: '32.4 kWh', efficiency: '82%' },
        { icon: '💧', name: 'Water Heater', location: 'Utility Room', status: 'on', current: '1.45 kW', today: '18.2 kWh', efficiency: '89%' },
        { icon: '🧊', name: 'Refrigerator', location: 'Kitchen', status: 'on', current: '0.68 kW', today: '12.8 kWh', efficiency: '91%' },
        { icon: '💡', name: 'Smart Lights', location: 'Living Room', status: 'on', current: '0.28 kW', today: '3.2 kWh', efficiency: '95%' },
        { icon: '📺', name: 'Smart TV', location: 'Living Room', status: 'on', current: '0.15 kW', today: '1.8 kWh', efficiency: '88%' },
        { icon: '🔌', name: 'Washing Machine', location: 'Laundry Room', status: 'off', current: '0.00 kW', today: '0.8 kWh', efficiency: '93%' },
        { icon: '🌀', name: 'Dryer', location: 'Laundry Room', status: 'off', current: '0.00 kW', today: '0.0 kWh', efficiency: '85%' },
        { icon: '🍽️', name: 'Dishwasher', location: 'Kitchen', status: 'off', current: '0.00 kW', today: '1.2 kWh', efficiency: '90%' },
        { icon: '💻', name: 'Home Office', location: 'Office', status: 'on', current: '0.32 kW', today: '4.5 kWh', efficiency: '87%' },
        { icon: '🎮', name: 'Gaming Console', location: 'Bedroom', status: 'off', current: '0.00 kW', today: '0.0 kWh', efficiency: '78%' }
    ];
    
    const grid = document.getElementById('devicesDetailed');
    if (!grid) return;
    
    grid.innerHTML = detailedDevices.map(device => `
        <div class="device-detailed-card">
            <div class="device-detailed-header">
                <div class="device-detailed-info">
                    <div class="device-icon" style="font-size: 1.5rem; margin-bottom: 0.5rem;">${device.icon}</div>
                    <h4>${device.name}</h4>
                    <p class="device-location">${device.location}</p>
                </div>
                <span class="device-status-badge ${device.status}">${device.status.toUpperCase()}</span>
            </div>
            <div class="device-stats">
                <div class="device-stat">
                    <p class="device-stat-label">Current</p>
                    <p class="device-stat-value">${device.current}</p>
                </div>
                <div class="device-stat">
                    <p class="device-stat-label">Today</p>
                    <p class="device-stat-value">${device.today}</p>
                </div>
                <div class="device-stat">
                    <p class="device-stat-label">Efficiency</p>
                    <p class="device-stat-value">${device.efficiency}</p>
                </div>
            </div>
        </div>
    `).join('');
}

// Hourly Predictions
function initializeHourlyPredictions() {
    const container = document.getElementById('hourlyPredictions');
    if (!container) return;
    
    const now = new Date();
    const predictions = [];
    
    for (let i = 1; i <= 24; i++) {
        const time = new Date(now.getTime() + i * 3600000);
        const hour = time.getHours();
        const value = 1.2 + Math.sin(hour / 12 * Math.PI) * 1.5 + Math.random() * 0.3;
        
        predictions.push({
            time: `${hour}:00`,
            value: value.toFixed(2)
        });
    }
    
    container.innerHTML = predictions.map(pred => `
        <div class="hourly-item">
            <span class="hourly-time">${pred.time}</span>
            <span class="hourly-value">${pred.value} kW</span>
        </div>
    `).join('');
}

// Heatmap Generation
function initializeHeatmap() {
    const container = document.getElementById('heatmapContainer');
    if (!container) return;
    
    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    let heatmapHTML = '';
    
    days.forEach(day => {
        heatmapHTML += `<div style="grid-column: 1 / -1; font-size: 0.75rem; color: #6b7589; padding: 0.25rem 0;">${day}</div>`;
        
        for (let hour = 0; hour < 24; hour++) {
            const value = Math.random();
            const color = getHeatmapColor(value);
            const title = `${day} ${hour}:00 - ${(value * 5).toFixed(2)} kW`;
            
            heatmapHTML += `<div class="heatmap-cell" style="background: ${color};" title="${title}"></div>`;
        }
    });
    
    container.innerHTML = heatmapHTML;
}

function getHeatmapColor(value) {
    const colors = [
        'rgba(0, 212, 255, 0.1)',
        'rgba(0, 212, 255, 0.3)',
        'rgba(0, 212, 255, 0.5)',
        'rgba(0, 212, 255, 0.7)',
        'rgba(0, 212, 255, 0.9)'
    ];
    
    const index = Math.floor(value * colors.length);
    return colors[Math.min(index, colors.length - 1)];
}

// Real-time Updates
function startRealTimeUpdates() {
    // Update current power every 5 seconds
    updateIntervals.power = setInterval(() => {
        updateCurrentPower();
        updateRealtimeChart();
    }, CONFIG.UPDATE_INTERVAL);
    
    // Update stats every 10 seconds
    updateIntervals.stats = setInterval(() => {
        updateStats();
    }, 10000);
}

function updateCurrentPower() {
    const powerEl = document.getElementById('currentPower');
    if (!powerEl) return;
    
    const current = parseFloat(powerEl.textContent);
    const newValue = current + (Math.random() - 0.5) * 0.3;
    powerEl.textContent = Math.max(0.5, newValue).toFixed(2) + ' kW';
}

function updateRealtimeChart() {
    if (!charts.realtime) return;
    
    const data = charts.realtime.data.datasets[0].data;
    data.shift();
    data.push(2 + Math.random() * 2);
    
    const labels = charts.realtime.data.labels;
    labels.shift();
    const now = new Date();
    labels.push(now.getHours() + ':' + now.getMinutes());
    
    charts.realtime.update('none');
}

function updateStats() {
    // Simulate stat updates
    const stats = [
        { id: 'todayUsage', min: 20, max: 30, unit: ' kWh' },
        { id: 'monthlyCost', min: 140, max: 160, unit: '' },
        { id: 'accuracy', min: 93, max: 96, unit: '%' }
    ];
    
    stats.forEach(stat => {
        const el = document.getElementById(stat.id);
        if (el) {
            const value = stat.min + Math.random() * (stat.max - stat.min);
            el.textContent = (stat.id === 'monthlyCost' ? '$' : '') + 
                            value.toFixed(stat.id === 'accuracy' ? 1 : 2) + 
                            stat.unit;
        }
    });
}

// Utility Functions
function generateRealisticData(count, min, max) {
    const data = [];
    let current = (min + max) / 2;
    
    for (let i = 0; i < count; i++) {
        const change = (Math.random() - 0.5) * 0.5;
        current = Math.max(min, Math.min(max, current + change));
        
        // Add time-of-day pattern
        const hourEffect = Math.sin(i / count * Math.PI * 2) * 0.8;
        data.push(current + hourEffect);
    }
    
    return data;
}

function generateLossData(count, start, end) {
    const data = [];
    const decay = Math.pow(end / start, 1 / count);
    
    for (let i = 0; i < count; i++) {
        const value = start * Math.pow(decay, i);
        const noise = (Math.random() - 0.5) * 0.02;
        data.push(Math.max(end, value + noise));
    }
    
    return data;
}

// Event Listeners
document.getElementById('timeRangeSelect')?.addEventListener('change', (e) => {
    // Update chart based on time range
    console.log('Time range changed to:', e.target.value);
});

document.getElementById('analyticsPeriod')?.addEventListener('change', (e) => {
    // Update analytics based on period
    console.log('Analytics period changed to:', e.target.value);
});

document.querySelectorAll('.toggle-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        const parent = this.parentElement;
        parent.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        
        const chartType = this.dataset.chart;
        if (chartType && charts.trends) {
            charts.trends.config.type = chartType === 'area' ? 'line' : chartType;
            if (chartType === 'area') {
                charts.trends.data.datasets.forEach(ds => ds.fill = true);
            } else {
                charts.trends.data.datasets.forEach(ds => ds.fill = false);
            }
            charts.trends.update();
        }
    });
});

// Filter buttons for devices view
document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        
        const room = this.dataset.room;
        // Filter logic would go here
        console.log('Filtering devices for room:', room);
    });
});

// Model card selection
document.querySelectorAll('.model-card').forEach(card => {
    card.addEventListener('click', function() {
        document.querySelectorAll('.model-card').forEach(c => c.classList.remove('active'));
        this.classList.add('active');
    });
});

// API Integration Functions (to be connected to Flask backend)
async function fetchEnergyData(endpoint) {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/${endpoint}`);
        if (!response.ok) throw new Error('Network response was not ok');
        return await response.json();
    } catch (error) {
        console.error('Error fetching data:', error);
        return null;
    }
}

async function sendPredictionRequest(data) {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) throw new Error('Prediction request failed');
        return await response.json();
    } catch (error) {
        console.error('Error sending prediction request:', error);
        return null;
    }
}

// Export functions for external use
window.EnergyDashboard = {
    switchView,
    updateCurrentPower,
    fetchEnergyData,
    sendPredictionRequest,
    charts
};

console.log('Smart Energy Dashboard initialized successfully! 🚀');
