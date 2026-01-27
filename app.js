        let dataset = [];
        let anomalies = []; // Array of anomaly objects
        let granularity = 'minute'; // 'minute' or 'hour'
        let minTimeRangePercent = 0;
        let maxTimeRangePercent = 100;
        let chart = null;
        let timeRangeSlider = null;

        // Chart.register(CrosshairPlugin); // Temporarily disabled due to loading issues

        // Drag and drop for dataset
        const datasetDropZone = document.getElementById('datasetDropZone');
        const datasetFileInput = document.getElementById('datasetFile');
        datasetDropZone.addEventListener('click', () => datasetFileInput.click());
        datasetDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            datasetDropZone.classList.add('dragover');
        });
        datasetDropZone.addEventListener('dragleave', () => {
            datasetDropZone.classList.remove('dragover');
        });
        datasetDropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            datasetDropZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                datasetFileInput.files = files;
                handleDatasetUpload({ target: datasetFileInput });
            }
        });
        datasetFileInput.addEventListener('change', handleDatasetUpload);

        // Drag and drop for metadata
        const metadataDropZone = document.getElementById('metadataDropZone');
        const metadataFileInput = document.getElementById('metadataFile');
        metadataDropZone.addEventListener('click', () => metadataFileInput.click());
        metadataDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            metadataDropZone.classList.add('dragover');
        });
        metadataDropZone.addEventListener('dragleave', () => {
            metadataDropZone.classList.remove('dragover');
        });
        metadataDropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            metadataDropZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                metadataFileInput.files = files;
                handleMetadataUpload({ target: metadataFileInput });
            }
        });
        metadataFileInput.addEventListener('change', handleMetadataUpload);

        document.getElementById('minuteBtn').addEventListener('click', () => setGranularity('minute'));
        document.getElementById('hourBtn').addEventListener('click', () => setGranularity('hour'));
        document.getElementById('trainBtn').addEventListener('click', trainModel);
        document.getElementById('detectBtn').addEventListener('click', detectAnomalies);
        document.getElementById('clearBtn').addEventListener('click', clearMetadata);

        function handleDatasetUpload(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    let content = e.target.result;
                    content = content.replace(/^\uFEFF/, ''); // Remove BOM
                    if (file.name.endsWith('.jsonl')) {
                        try {
                            dataset = content.trim().split('\n').filter(line => line.trim()).map(line => JSON.parse(line));
                            dataset.sort((a, b) => a.timestamp - b.timestamp);
                            Toastify({
                                text: `Loaded ${dataset.length} data points from ${file.name}`,
                                duration: 3000,
                                gravity: "top",
                                position: "right",
                                backgroundColor: "linear-gradient(to right, #4CAF50, #388E3C)",
                            }).showToast();
                            updateDisplay();
                        } catch (error) {
                            Toastify({
                                text: 'Error parsing JSONL file: ' + error.message,
                                duration: 5000,
                                gravity: "top",
                                position: "right",
                                backgroundColor: "linear-gradient(to right, #E91E63, #C2185B)",
                            }).showToast();
                        }
                    } else if (file.name.endsWith('.csv')) {
                        const lines = content.trim().split('\n').filter(line => line.trim());
                        if (lines.length < 2) {
                            Toastify({
                                text: 'CSV file must have at least header and one data row',
                                duration: 3000,
                                gravity: "top",
                                position: "right",
                                backgroundColor: "linear-gradient(to right, #FF9800, #F57C00)",
                            }).showToast();
                            return;
                        }
                        const headers = lines[0].split(',');
                        dataset = lines.slice(1).map(line => {
                            const values = line.split(',');
                            const obj = {};
                            headers.forEach((h, i) => {
                                const header = h.trim();
                                const value = values[i] ? values[i].trim() : '';
                                if (header === 'timestamp') {
                                    obj[header] = parseInt(value);
                                } else if (header === 'value') {
                                    obj[header] = parseFloat(value);
                                } else {
                                    obj[header] = value;
                                }
                            });
                            return obj;
                        });
                        dataset.sort((a, b) => a.timestamp - b.timestamp);
                        Toastify({
                            text: `Loaded ${dataset.length} data points from ${file.name}`,
                            duration: 3000,
                            gravity: "top",
                            position: "right",
                            backgroundColor: "linear-gradient(to right, #4CAF50, #388E3C)",
                        }).showToast();
                        updateDisplay();
                    }
                };
                reader.readAsText(file);
            }
        }

        function handleMetadataUpload(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    let content = e.target.result;
                    content = content.replace(/^\uFEFF/, ''); // Remove BOM
                    try {
                        anomalies = JSON.parse(content);
                        Toastify({
                            text: `Loaded ${anomalies.length} anomaly labels`,
                            duration: 3000,
                            gravity: "top",
                            position: "right",
                            backgroundColor: "linear-gradient(to right, #2196F3, #1976D2)",
                        }).showToast();
                        updateDisplay();
                    } catch (error) {
                        Toastify({
                            text: 'Error parsing JSON file: ' + error.message,
                            duration: 5000,
                            gravity: "top",
                            position: "right",
                            backgroundColor: "linear-gradient(to right, #E91E63, #C2185B)",
                        }).showToast();
                    }
                };
                reader.readAsText(file);
            }
        }

        function setGranularity(newGranularity) {
            granularity = newGranularity;
            document.getElementById('minuteBtn').classList.toggle('active', granularity === 'minute');
            document.getElementById('hourBtn').classList.toggle('active', granularity === 'hour');
            updateDisplay();
        }

        function handleTimeRangeChange(event) {
            timeRangePercent = parseInt(event.target.value);
            document.getElementById('timeRangeValue').textContent = timeRangePercent + '%';
            updateDisplay();
        }

        async function trainModel() {
            if (dataset.length === 0) {
                Toastify({
                    text: 'Please upload dataset first!',
                    duration: 3000,
                    gravity: "top",
                    position: "right",
                    backgroundColor: "linear-gradient(to right, #FF9800, #F57C00)",
                }).showToast();
                return;
            }

            const trainBtn = document.getElementById('trainBtn');
            trainBtn.classList.add('loading');
            trainBtn.disabled = true;

            try {
                const statusResponse = await fetch('http://localhost:8001/status');
                if (!statusResponse.ok) {
                    Toastify({
                        text: 'API server not running. Please start with: python3 api_server.py',
                        duration: 5000,
                        gravity: "top",
                        position: "right",
                        backgroundColor: "linear-gradient(to right, #E91E63, #C2185B)",
                    }).showToast();
                    return;
                }

                Toastify({
                    text: "Training HTM and LSM models...",
                    duration: 3000,
                    gravity: "top",
                    position: "right",
                    backgroundColor: "linear-gradient(to right, #00BCD4, #0097A7)",
                }).showToast();

                const trainResponse = await fetch('http://localhost:8001/train', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ data: dataset })
                });

                if (!trainResponse.ok) {
                    const error = await trainResponse.text();
                    Toastify({
                        text: 'Training failed: ' + error,
                        duration: 5000,
                        gravity: "top",
                        position: "right",
                        backgroundColor: "linear-gradient(to right, #E91E63, #C2185B)",
                    }).showToast();
                    return;
                }

                Toastify({
                    text: "Model trained successfully!",
                    duration: 3000,
                    gravity: "top",
                    position: "right",
                    backgroundColor: "linear-gradient(to right, #4CAF50, #388E3C)",
                }).showToast();

            } catch (error) {
                Toastify({
                    text: 'Error communicating with API server: ' + error.message,
                    duration: 5000,
                    gravity: "top",
                    position: "right",
                    backgroundColor: "linear-gradient(to right, #E91E63, #C2185B)",
                }).showToast();
            } finally {
                trainBtn.classList.remove('loading');
                trainBtn.disabled = false;
            }
        }

        async function detectAnomalies() {
            if (dataset.length === 0) {
                Toastify({
                    text: 'Please upload dataset and train the model first!',
                    duration: 3000,
                    gravity: "top",
                    position: "right",
                    backgroundColor: "linear-gradient(to right, #FF9800, #F57C00)",
                }).showToast();
                return;
            }

            const detectBtn = document.getElementById('detectBtn');
            detectBtn.classList.add('loading');
            detectBtn.disabled = true;

            try {
                const statusResponse = await fetch('http://localhost:8001/status');
                const statusData = await statusResponse.json();
                if (statusData.status !== 'trained') {
                    Toastify({
                        text: 'Model not trained. Please train first.',
                        duration: 4000,
                        gravity: "top",
                        position: "right",
                        backgroundColor: "linear-gradient(to right, #FF9800, #F57C00)",
                    }).showToast();
                    return;
                }

                Toastify({
                    text: "Detecting anomalies...",
                    duration: 3000,
                    gravity: "top",
                    position: "right",
                    backgroundColor: "linear-gradient(to right, #FF9800, #F57C00)",
                }).showToast();

                const detectedAnomalies = [];
                for (const point of dataset) {
                    const detectResponse = await fetch('http://localhost:8001/detect', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ point: point })
                    });

                    if (detectResponse.ok) {
                        const result = await detectResponse.json();
                        if (result.anomaly_flag) {
                            detectedAnomalies.push(point);
                        }
                    }
                }

                anomalies = detectedAnomalies.map(a => ({
                    timestamp: a.timestamp,
                    entity_id: a.entity_id,
                    signal_type: a.signal_type,
                    metric: a.metric,
                    anomaly_type: 'detected'
                }));

                updateDisplay();
                Toastify({
                    text: `Anomaly detection complete! Found ${detectedAnomalies.length} anomalies.`,
                    duration: 5000,
                    gravity: "top",
                    position: "right",
                    backgroundColor: "linear-gradient(to right, #4CAF50, #388E3C)",
                }).showToast();

            } catch (error) {
                Toastify({
                    text: 'Error communicating with API server: ' + error.message,
                    duration: 5000,
                    gravity: "top",
                    position: "right",
                    backgroundColor: "linear-gradient(to right, #E91E63, #C2185B)",
                }).showToast();
            } finally {
                detectBtn.classList.remove('loading');
                detectBtn.disabled = false;
            }
        }

        function clearMetadata() {
            anomalies = [];
            updateDisplay();
            Toastify({
                text: "Metadata cleared!",
                duration: 3000,
                gravity: "top",
                position: "right",
                backgroundColor: "linear-gradient(to right, #9C27B0, #7B1FA2)",
            }).showToast();
        }





        function computeStatistics(data) {
            const entityStats = {};

            data.forEach(d => {
                const key = `${d.entity_id}-${d.metric}`;
                if (!entityStats[key]) {
                    entityStats[key] = { values: [], mean: 0, std: 0 };
                }
                entityStats[key].values.push(d.value);
            });

            Object.keys(entityStats).forEach(key => {
                const values = entityStats[key].values;
                const mean = values.reduce((a, b) => a + b, 0) / values.length;
                const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
                const std = Math.sqrt(variance);
                entityStats[key].mean = mean;
                entityStats[key].std = std;
            });

            return entityStats;
        }

        function detectAnomalies(data, stats) {
            const anomalies = [];
            const threshold = 3; // Z-score threshold

            data.forEach(d => {
                const key = `${d.entity_id}-${d.metric}`;
                const stat = stats[key];
                if (stat && stat.std > 0) {
                    const zScore = Math.abs((d.value - stat.mean) / stat.std);
                    if (zScore > threshold) {
                        anomalies.push(d);
                    }
                }
            });

            return anomalies;
        }

        function updateDisplay() {
            if (dataset.length > 0) {
                drawChart();
                updateStats();
                updateAnomalyList();
                initTimeRangeSlider();
                document.getElementById('bottomPanel').style.display = 'flex';
            }
        }

        function initTimeRangeSlider() {
            if (timeRangeSlider) return; // Already initialized

            const sliderElement = document.getElementById('timeRangeSlider');
            timeRangeSlider = noUiSlider.create(sliderElement, {
                start: [0, 100],
                connect: true,
                range: {
                    'min': 0,
                    'max': 100
                },
                step: 1,
                tooltips: [true, true],
                format: {
                    to: function (value) {
                        return Math.round(value) + '%';
                    },
                    from: function (value) {
                        return Number(value.replace('%', ''));
                    }
                }
            });

            timeRangeSlider.on('update', function (values, handle) {
                minTimeRangePercent = parseInt(values[0]);
                maxTimeRangePercent = parseInt(values[1]);
                document.getElementById('timeRangeValue').textContent = values[0] + ' - ' + values[1];
                updateDisplay();
            });
        }

        function aggregateData(data, granularity) {
            if (granularity === 'minute') return data;

            // Aggregate to hourly
            const hourlyData = [];
            const hourMap = new Map();

            data.forEach(d => {
                const hour = Math.floor(d.timestamp / 3600) * 3600; // Round to hour
                const key = `${d.entity_id}-${d.signal_type}-${d.metric}-${hour}`;
                if (!hourMap.has(key)) {
                    hourMap.set(key, { values: [], timestamp: hour, entity_id: d.entity_id, signal_type: d.signal_type, metric: d.metric });
                }
                hourMap.get(key).values.push(d.value);
            });

            hourMap.forEach(entry => {
                const avgValue = entry.values.reduce((a, b) => a + b, 0) / entry.values.length;
                hourlyData.push({ ...entry, value: avgValue });
            });

            return hourlyData.sort((a, b) => a.timestamp - b.timestamp);
        }

        function filterByTimeRange(data, minPercent, maxPercent) {
            if (minPercent <= 0 && maxPercent >= 100) return data;

            const minTime = Math.min(...data.map(d => d.timestamp));
            const maxTime = Math.max(...data.map(d => d.timestamp));
            const range = maxTime - minTime;
            const newMinTime = minTime + (range * minPercent / 100);
            const newMaxTime = minTime + (range * maxPercent / 100);

            return data.filter(d => d.timestamp >= newMinTime && d.timestamp <= newMaxTime);
        }

        function drawChart() {
            if (dataset.length === 0) return;

            let displayData = aggregateData(dataset, granularity);
            displayData = filterByTimeRange(displayData, minTimeRangePercent, maxTimeRangePercent);

            if (displayData.length === 0) return;

            const ctx = document.getElementById('chart').getContext('2d');

            // Create gradient for fill
            const gradient = ctx.createLinearGradient(0, 0, 0, 400);
            gradient.addColorStop(0, 'rgba(0, 255, 255, 0.3)');
            gradient.addColorStop(1, 'rgba(0, 255, 255, 0.05)');

            // Prepare normal data
            const normalData = displayData.map(d => ({
                x: d.timestamp * 1000, // milliseconds
                y: d.value
            }));

            // Prepare anomaly data
            const anomalyData = displayData
                .filter(d => anomalies.some(a => a.timestamp === d.timestamp && a.entity_id === d.entity_id && a.metric === d.metric))
                .map(d => ({
                    x: d.timestamp * 1000,
                    y: d.value
                }));

            const data = {
                datasets: [{
                    label: 'Time Series',
                    data: normalData,
                    borderColor: '#00FFFF',
                    backgroundColor: gradient,
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 5
                }, {
                    label: 'Anomalies',
                    data: anomalyData,
                    borderColor: '#FF00FF',
                    backgroundColor: '#FF00FF',
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    showLine: false,
                    pointStyle: 'circle',
                    animation: {
                        onComplete: function() {
                            // Pulsing effect
                            const meta = this.getDatasetMeta(1);
                            meta.data.forEach((point, index) => {
                                point._model.radius = 6;
                                this.animatePoint(index, point);
                            });
                        }
                    }
                }]
            };

            const config = {
                type: 'line',
                data: data,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false
                    },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `${context.dataset.label}: ${context.parsed.y.toFixed(2)}`;
                                },
                                title: function(context) {
                                    return new Date(context[0].parsed.x).toLocaleString();
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                displayFormats: {
                                    minute: 'HH:mm',
                                    hour: 'MMM dd HH:mm'
                                }
                            },
                            title: {
                                display: true,
                                text: 'Time'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Value'
                            }
                        }
                    },
                    animation: {
                        duration: 1000,
                        easing: 'easeInOutQuart'
                    }
                }
            };

            if (chart) {
                chart.destroy();
            }
            chart = new Chart(ctx, config);
        }

        function updateStats() {
            let displayData = aggregateData(dataset, granularity);
            displayData = filterByTimeRange(displayData, minTimeRangePercent, maxTimeRangePercent);
            document.getElementById('totalPoints').textContent = dataset.length;
            document.getElementById('anomalyCount').textContent = anomalies.length;
            if (displayData.length > 0) {
                const minTime = Math.min(...displayData.map(d => d.timestamp));
                const maxTime = Math.max(...displayData.map(d => d.timestamp));
                const duration = (maxTime - minTime) / 3600; // hours
                document.getElementById('timeRange').textContent = `${duration.toFixed(1)} hours`;
            }
        }

        function updateAnomalyList() {
            const container = document.getElementById('anomalyItems');
            container.innerHTML = '';
            anomalies.forEach(a => {
                const div = document.createElement('div');
                div.className = 'anomaly-item';
                div.innerHTML = `
                    <strong>Anomaly Type:</strong> ${a.anomaly_type || 'Unknown'}<br>
                    <strong>Entity ID:</strong> ${a.entity_id}<br>
                    <strong>Entity Type:</strong> ${a.entity_type || 'Unknown'}<br>
                    <strong>Signal Type:</strong> ${a.signal_type} - <strong>Metric:</strong> ${a.metric}<br>
                    <strong>Timestamp:</strong> ${new Date(a.timestamp * 1000).toLocaleString()}<br>
                    <strong>Value:</strong> <span style="font-family: 'JetBrains Mono', monospace; color: #00FFFF;">${'value' in a ? a.value.toFixed(2) : 'N/A'}</span>
                `;
                container.appendChild(div);
            });
        }

        // Make canvas responsive
        window.addEventListener('resize', () => {
            const canvas = document.getElementById('chart');
            canvas.width = Math.min(1000, window.innerWidth - 40);
            if (dataset.length > 0) drawChart();
        });
    </script>
