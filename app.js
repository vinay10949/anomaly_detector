let dataset = [];
let referenceAnomalies = [];
let detectedAnomalies = [];
let mismatchAnomalies = { fp: [], fn: [] };
let comparisonStats = emptyComparisonStats();

let granularity = 'raw';
let minTimeRangePercent = 0;
let maxTimeRangePercent = 100;
let chart = null;
let timeRangeSlider = null;
let trainingStatusPollHandle = null;

const layerVisibility = {
    detected: true,
    reference: true,
    overlap: true,
};

const TIMING_MATCH_WINDOW_SEC = 300;
const MAX_LIST_ITEMS = 200;

const datasetDropZone = document.getElementById('datasetDropZone');
const datasetFileInput = document.getElementById('datasetFile');
const metadataDropZone = document.getElementById('metadataDropZone');
const metadataFileInput = document.getElementById('metadataFile');

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

document.getElementById('aggregateBtn').addEventListener('click', () => setGranularity('raw'));
document.getElementById('minuteBtn').addEventListener('click', () => setGranularity('minute'));
document.getElementById('hourBtn').addEventListener('click', () => setGranularity('hour'));
document.getElementById('trainBtn').addEventListener('click', trainModel);
document.getElementById('detectBtn').addEventListener('click', detectAnomaliesBatch);
document.getElementById('clearBtn').addEventListener('click', clearAnomalies);

const toggleDetected = document.getElementById('toggleDetected');
const toggleReference = document.getElementById('toggleReference');
const toggleOverlap = document.getElementById('toggleOverlap');

if (toggleDetected) {
    toggleDetected.addEventListener('change', (e) => {
        layerVisibility.detected = e.target.checked;
        drawChart();
    });
}
if (toggleReference) {
    toggleReference.addEventListener('change', (e) => {
        layerVisibility.reference = e.target.checked;
        drawChart();
    });
}
if (toggleOverlap) {
    toggleOverlap.addEventListener('change', (e) => {
        layerVisibility.overlap = e.target.checked;
        drawChart();
    });
}

function emptyComparisonStats() {
    return {
        tp: 0,
        fp: 0,
        fn: 0,
        precision: 0,
        recall: 0,
        f1: 0,
    };
}

function fullKey(point) {
    return `${point.timestamp}|${point.entity_id}|${point.signal_type}|${point.metric}`;
}

function streamKey(point) {
    return `${point.entity_id}|${point.signal_type}|${point.metric}`;
}

function chartKey(point, granularityMode) {
    const bucketTs = granularityMode === 'hour'
        ? Math.floor(point.timestamp / 3600) * 3600
        : granularityMode === 'minute'
            ? Math.floor(point.timestamp / 60) * 60
            : point.timestamp;
    return `${bucketTs}|${point.entity_id}|${point.signal_type}|${point.metric}`;
}

function parseCsv(content) {
    const lines = content.trim().split('\n').filter(line => line.trim());
    if (lines.length < 2) {
        throw new Error('CSV file must have at least header and one data row');
    }
    const headers = lines[0].split(',').map(h => h.trim());
    return lines.slice(1).map(line => {
        const values = line.split(',');
        const row = {};
        headers.forEach((header, i) => {
            const value = values[i] ? values[i].trim() : '';
            if (header === 'timestamp') {
                row[header] = parseInt(value, 10);
            } else if (header === 'value') {
                row[header] = parseFloat(value);
            } else {
                row[header] = value;
            }
        });
        return row;
    });
}

function dedupeDatasetPoints(points) {
    const map = new Map();
    points.forEach((point) => {
        if (
            point &&
            Number.isFinite(Number(point.timestamp)) &&
            Number.isFinite(Number(point.value)) &&
            point.entity_id != null &&
            point.signal_type != null &&
            point.metric != null
        ) {
            const normalized = {
                timestamp: Number(point.timestamp),
                entity_id: String(point.entity_id),
                signal_type: String(point.signal_type),
                metric: String(point.metric),
                value: Number(point.value),
            };
            map.set(fullKey(normalized), normalized);
        }
    });
    return Array.from(map.values()).sort((a, b) => a.timestamp - b.timestamp);
}

function dedupeReferenceAnomalies(records) {
    const map = new Map();
    records.forEach((record) => {
        if (!record) return;
        if (
            !Number.isFinite(Number(record.timestamp)) ||
            record.entity_id == null ||
            record.signal_type == null ||
            record.metric == null
        ) {
            return;
        }

        const normalized = {
            timestamp: Number(record.timestamp),
            entity_id: String(record.entity_id),
            signal_type: String(record.signal_type),
            metric: String(record.metric),
            entity_type: record.entity_type ? String(record.entity_type) : 'Unknown',
            anomaly_types: new Set(),
        };

        const key = fullKey(normalized);
        if (!map.has(key)) {
            map.set(key, normalized);
        }
        const existing = map.get(key);

        if (Array.isArray(record.anomaly_types)) {
            record.anomaly_types.forEach((t) => existing.anomaly_types.add(String(t)));
        }
        if (record.anomaly_type) {
            existing.anomaly_types.add(String(record.anomaly_type));
        }
    });

    return Array.from(map.values())
        .map((x) => ({ ...x, anomaly_types: Array.from(x.anomaly_types).sort() }))
        .sort((a, b) => a.timestamp - b.timestamp);
}

function formatPct(value) {
    return `${(value * 100).toFixed(2)}%`;
}

function toast(text, backgroundColor = 'linear-gradient(to right, #4CAF50, #388E3C)', duration = 3000) {
    Toastify({
        text,
        duration,
        gravity: 'top',
        position: 'right',
        backgroundColor,
    }).showToast();
}

async function parseResponsePayload(response) {
    const raw = await response.text();
    if (!raw) return { data: null, text: '' };
    try {
        return { data: JSON.parse(raw), text: raw };
    } catch {
        return { data: null, text: raw };
    }
}

function setTrainingProgress(message, color = 'var(--muted)') {
    const el = document.getElementById('trainingProgress');
    if (!el) return;
    el.textContent = message;
    el.style.color = color;
}

function formatEtaSeconds(seconds) {
    if (!Number.isFinite(Number(seconds)) || Number(seconds) < 0) {
        return 'n/a';
    }
    const total = Math.floor(Number(seconds));
    const mins = Math.floor(total / 60);
    const secs = total % 60;
    if (mins <= 0) return `${secs}s`;
    return `${mins}m ${secs}s`;
}

async function refreshTrainingProgress() {
    try {
        const response = await fetch('http://localhost:8001/status');
        if (!response.ok) return;
        const { data } = await parseResponsePayload(response);
        const training = data?.training;
        if (!training?.active) return;

        const processed = Number(training.processed ?? 0);
        const total = Math.max(1, Number(training.total ?? 0));
        const pct = Math.min(100, (processed / total) * 100);
        const rate = Number(training.rate_points_per_sec ?? 0);
        const eta = formatEtaSeconds(training.eta_sec);
        setTrainingProgress(
            `Training: ${processed}/${total} (${pct.toFixed(1)}%) | ${rate.toFixed(1)} pts/s | ETA ${eta}`,
            '#2dd4bf',
        );
    } catch {
        // Ignore intermittent polling failures while training request is in-flight.
    }
}

function startTrainingProgressPolling() {
    if (trainingStatusPollHandle) {
        clearInterval(trainingStatusPollHandle);
    }
    setTrainingProgress('Training: starting...', '#2dd4bf');
    refreshTrainingProgress();
    trainingStatusPollHandle = setInterval(refreshTrainingProgress, 1000);
}

function stopTrainingProgressPolling(message, color = 'var(--muted)') {
    if (trainingStatusPollHandle) {
        clearInterval(trainingStatusPollHandle);
        trainingStatusPollHandle = null;
    }
    setTrainingProgress(message, color);
}

function legacyMetadataToRecords(raw) {
    if (!raw || typeof raw !== 'object') return null;
    if (!raw.anomalies_injected || !raw.start_timestamp || !raw.step_seconds) return null;

    const startTs = Number(raw.start_timestamp);
    const step = Number(raw.step_seconds);
    if (!Number.isFinite(startTs) || !Number.isFinite(step) || step <= 0) return null;

    const entity_id = raw.entity_id != null ? String(raw.entity_id) : 'unknown';
    const signal_type = raw.signal_type != null ? String(raw.signal_type) : 'default';
    const metric = raw.metric != null ? String(raw.metric) : 'value';
    const entity_type = raw.entity_type != null ? String(raw.entity_type) : 'Unknown';

    const injected = raw.anomalies_injected || {};
    const records = [];

    function pushPoint(idx, type) {
        const timestamp = startTs + idx * step;
        records.push({
            timestamp,
            entity_id,
            signal_type,
            metric,
            entity_type,
            anomaly_type: type,
        });
    }

    function pushRange(startIdx, endIdxExclusive, type) {
        const start = Math.max(0, Number(startIdx));
        const end = Math.max(start, Number(endIdxExclusive));
        if (!Number.isFinite(start) || !Number.isFinite(end)) return;
        for (let i = start; i < end; i += 1) pushPoint(i, type);
    }

    if (Array.isArray(injected.point_spikes)) {
        injected.point_spikes.forEach((idx) => {
            if (Number.isFinite(Number(idx))) pushPoint(Number(idx), 'point_spike');
        });
    }

    if (Array.isArray(injected.collective_anomaly) && injected.collective_anomaly.length === 2) {
        pushRange(injected.collective_anomaly[0], injected.collective_anomaly[1], 'collective');
    }
    if (Array.isArray(injected.gradual_drift) && injected.gradual_drift.length === 2) {
        pushRange(injected.gradual_drift[0], injected.gradual_drift[1], 'drift');
    }
    if (Array.isArray(injected.contextual_anomaly) && injected.contextual_anomaly.length === 2) {
        pushRange(injected.contextual_anomaly[0], injected.contextual_anomaly[1], 'contextual');
    }

    // Older generator only stored start index for these, so use the script's defaults.
    if (Number.isFinite(Number(injected.sudden_drop))) {
        const start = Number(injected.sudden_drop);
        pushRange(start, start + 45, 'drop');
    }
    if (Number.isFinite(Number(injected.oscillating_anomaly))) {
        const start = Number(injected.oscillating_anomaly);
        pushRange(start, start + 90, 'oscillating');
    }

    return records;
}

function refreshComparison() {
    const computed = computeComparisonStats(detectedAnomalies, referenceAnomalies);
    comparisonStats = computed.stats;
    mismatchAnomalies = computed.mismatch;
}

function handleDatasetUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            let content = e.target.result;
            content = content.replace(/^\uFEFF/, '');

            let rawPoints = [];
            if (file.name.endsWith('.jsonl')) {
                rawPoints = content
                    .trim()
                    .split('\n')
                    .filter((line) => line.trim())
                    .map((line) => JSON.parse(line));
            } else if (file.name.endsWith('.csv')) {
                rawPoints = parseCsv(content);
            } else {
                throw new Error('Unsupported dataset format. Use .jsonl or .csv');
            }

            const deduped = dedupeDatasetPoints(rawPoints);
            const removed = rawPoints.length - deduped.length;
            dataset = deduped;
            refreshComparison();
            updateDisplay();

            if (removed > 0) {
                toast(
                    `Loaded ${dataset.length} points (${removed} duplicates removed) from ${file.name}`,
                    'linear-gradient(to right, #4CAF50, #388E3C)',
                    4500,
                );
            } else {
                toast(`Loaded ${dataset.length} points from ${file.name}`);
            }
        } catch (error) {
            toast(
                `Error parsing dataset: ${error.message}`,
                'linear-gradient(to right, #E91E63, #C2185B)',
                5000,
            );
        }
    };
    reader.readAsText(file);
}

function handleMetadataUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            let content = e.target.result;
            content = content.replace(/^\uFEFF/, '');
            const raw = JSON.parse(content);
            const records = Array.isArray(raw) ? raw : legacyMetadataToRecords(raw);
            if (!Array.isArray(records)) {
                throw new Error('Metadata should be an array of anomaly objects (or legacy generator metadata)');
            }

            referenceAnomalies = dedupeReferenceAnomalies(records);
            refreshComparison();
            updateDisplay();
            toast(
                `Loaded ${referenceAnomalies.length} reference anomalies`,
                'linear-gradient(to right, #2196F3, #1976D2)',
            );
        } catch (error) {
            toast(
                `Error parsing metadata: ${error.message}`,
                'linear-gradient(to right, #E91E63, #C2185B)',
                5000,
            );
        }
    };
    reader.readAsText(file);
}

function setGranularity(newGranularity) {
    granularity = newGranularity;
    document.getElementById('aggregateBtn').classList.toggle('active', granularity === 'raw');
    document.getElementById('minuteBtn').classList.toggle('active', granularity === 'minute');
    document.getElementById('hourBtn').classList.toggle('active', granularity === 'hour');
    updateDisplay();
}

async function trainModel() {
    if (dataset.length === 0) {
        toast('Please upload dataset first!', 'linear-gradient(to right, #FF9800, #F57C00)');
        return;
    }

    const trainBtn = document.getElementById('trainBtn');
    trainBtn.classList.add('loading');
    trainBtn.disabled = true;

    try {
        const statusResponse = await fetch('http://localhost:8001/status');
        if (!statusResponse.ok) {
            toast(
                'API server not running. Start with: .venv/bin/python api_server.py (HTM requires Python 3.11 env)',
                'linear-gradient(to right, #E91E63, #C2185B)',
                5000,
            );
            return;
        }
        const statusData = await statusResponse.json();
        if (statusData?.runtime?.htm_available === false) {
            toast(
                `HTM not available in server runtime (${statusData.runtime.python_version}). Start server with .venv/bin/python.`,
                'linear-gradient(to right, #E91E63, #C2185B)',
                7000,
            );
            return;
        }

        toast('Training detector...', 'linear-gradient(to right, #00BCD4, #0097A7)');
        startTrainingProgressPolling();

        const trainResponse = await fetch('http://localhost:8001/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                data: dataset,
                reference_metadata: referenceAnomalies.length > 0 ? referenceAnomalies : undefined,
            }),
        });

        if (!trainResponse.ok) {
            const errorText = await trainResponse.text();
            stopTrainingProgressPolling(`Training failed: ${errorText}`, '#f43f5e');
            toast(`Training failed: ${errorText}`, 'linear-gradient(to right, #E91E63, #C2185B)', 5000);
            return;
        }

        const payload = await trainResponse.json();
        if (Number.isFinite(Number(payload.excluded_reference)) && payload.excluded_reference > 0) {
            toast(
                `Training excluded ${payload.excluded_reference} reference-anomaly points`,
                'linear-gradient(to right, #2196F3, #1976D2)',
                4500,
            );
        }
        toast(
            `Model trained: ${payload.trained_streams} streams, ${payload.points_used} points used`,
            'linear-gradient(to right, #4CAF50, #388E3C)',
            4500,
        );
        stopTrainingProgressPolling(
            `Training complete: ${payload.points_used} points across ${payload.trained_streams} stream(s)`,
            '#2dd4bf',
        );
    } catch (error) {
        stopTrainingProgressPolling(`Training failed: ${error.message}`, '#f43f5e');
        toast(
            `Error communicating with API server: ${error.message}`,
            'linear-gradient(to right, #E91E63, #C2185B)',
            5000,
        );
    } finally {
        trainBtn.classList.remove('loading');
        trainBtn.disabled = false;
    }
}

async function detectAnomaliesBatch() {
    if (dataset.length === 0) {
        toast('Please upload dataset and train the model first!', 'linear-gradient(to right, #FF9800, #F57C00)');
        return;
    }

    const detectBtn = document.getElementById('detectBtn');
    detectBtn.classList.add('loading');
    detectBtn.disabled = true;

    try {
        const statusResponse = await fetch('http://localhost:8001/status');
        if (!statusResponse.ok) {
            toast('Cannot reach API server.', 'linear-gradient(to right, #E91E63, #C2185B)', 5000);
            return;
        }

        const statusData = await statusResponse.json();
        if (statusData.status !== 'trained') {
            toast('Model not trained. Please train first.', 'linear-gradient(to right, #FF9800, #F57C00)', 4000);
            return;
        }

        toast('Detecting anomalies (batch)...', 'linear-gradient(to right, #FF9800, #F57C00)');

        const detectResponse = await fetch('http://localhost:8001/detect_batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                points: dataset,
                return_scores: true,
                learn: false,
                reset_sequence: true,
                batch_warmup_points: 8,
            }),
        });

        if (!detectResponse.ok) {
            const errorText = await detectResponse.text();
            toast(`Detection failed: ${errorText}`, 'linear-gradient(to right, #E91E63, #C2185B)', 5000);
            return;
        }

        const payload = await detectResponse.json();
        const results = Array.isArray(payload.results) ? payload.results : [];

        const valueMap = new Map(dataset.map((point) => [fullKey(point), point.value]));
        detectedAnomalies = results
            .filter((result) => result.anomaly_flag)
            .map((result) => {
                const key = `${result.timestamp}|${result.entity_id}|${result.signal_type}|${result.metric}`;
                return {
                    timestamp: result.timestamp,
                    entity_id: result.entity_id,
                    signal_type: result.signal_type,
                    metric: result.metric,
                    anomaly_type: 'detected',
                    score: Number(result.score ?? 0),
                    score_likelihood: Number(result.score_likelihood ?? 0),
                    confidence: Number(result.confidence ?? 0),
                    value: valueMap.get(key),
                };
            });

        refreshComparison();
        updateDisplay();

        const detectedCount = payload.summary?.detected_count ?? detectedAnomalies.length;
        toast(
            `Detection complete: ${detectedCount} anomalies found`,
            'linear-gradient(to right, #4CAF50, #388E3C)',
            5000,
        );
    } catch (error) {
        toast(
            `Error communicating with API server: ${error.message}`,
            'linear-gradient(to right, #E91E63, #C2185B)',
            5000,
        );
    } finally {
        detectBtn.classList.remove('loading');
        detectBtn.disabled = false;
    }
}

function clearAnomalies() {
    detectedAnomalies = [];
    referenceAnomalies = [];
    mismatchAnomalies = { fp: [], fn: [] };
    comparisonStats = emptyComparisonStats();
    updateDisplay();
    toast('Cleared detected and reference anomalies', 'linear-gradient(to right, #9C27B0, #7B1FA2)');
}

function updateDisplay() {
    if (dataset.length > 0) {
        drawChart();
        updateStats();
        updateAnomalyLists();
        initTimeRangeSlider();
        document.getElementById('bottomPanel').style.display = 'grid';
        return;
    }

    if (chart) {
        chart.destroy();
        chart = null;
    }
    document.getElementById('bottomPanel').style.display = 'none';
}

function initTimeRangeSlider() {
    if (timeRangeSlider) return;

    const sliderElement = document.getElementById('timeRangeSlider');
    timeRangeSlider = noUiSlider.create(sliderElement, {
        start: [0, 100],
        connect: true,
        range: { min: 0, max: 100 },
        step: 1,
        tooltips: [true, true],
        format: {
            to: (value) => `${Math.round(value)}%`,
            from: (value) => Number(value.replace('%', '')),
        },
    });

    timeRangeSlider.on('update', function(values) {
        minTimeRangePercent = parseInt(values[0], 10);
        maxTimeRangePercent = parseInt(values[1], 10);
        document.getElementById('timeRangeValue').textContent = `${values[0]} - ${values[1]}`;
        drawChart();
        updateStats();
    });
}

function aggregateData(data, granularityMode) {
    if (granularityMode === 'raw') return data;

    const bucketSeconds = granularityMode === 'hour' ? 3600 : 60;
    const bucketMap = new Map();

    data.forEach((point) => {
        const bucketTs = Math.floor(point.timestamp / bucketSeconds) * bucketSeconds;
        const key = `${point.entity_id}|${point.signal_type}|${point.metric}|${bucketTs}`;
        const entry = bucketMap.get(key);
        if (!entry) {
            bucketMap.set(key, {
                timestamp: bucketTs,
                entity_id: point.entity_id,
                signal_type: point.signal_type,
                metric: point.metric,
                sum: point.value,
                count: 1,
            });
            return;
        }
        entry.sum += point.value;
        entry.count += 1;
    });

    const result = [];
    bucketMap.forEach((entry) => {
        result.push({
            timestamp: entry.timestamp,
            entity_id: entry.entity_id,
            signal_type: entry.signal_type,
            metric: entry.metric,
            value: entry.sum / entry.count,
        });
    });

    return result.sort((a, b) => a.timestamp - b.timestamp);
}

function filterByTimeRange(data, minPercent, maxPercent) {
    if (!data.length || (minPercent <= 0 && maxPercent >= 100)) return data;

    const minTs = Math.min(...data.map((d) => d.timestamp));
    const maxTs = Math.max(...data.map((d) => d.timestamp));
    const range = maxTs - minTs;
    const startTs = minTs + (range * minPercent / 100);
    const endTs = minTs + (range * maxPercent / 100);
    return data.filter((d) => d.timestamp >= startTs && d.timestamp <= endTs);
}

function buildAnomalyLookup(anomalies, granularityMode) {
    const set = new Set();
    anomalies.forEach((a) => set.add(chartKey(a, granularityMode)));
    return set;
}

function drawChart() {
    const existingChart = Chart.getChart("chart");
    if (existingChart) {
        existingChart.destroy();
    }
    chart = null; // Ensure global reference is cleared
    
    if (!dataset.length) return;

    let displayData = aggregateData(dataset, granularity);
    displayData = filterByTimeRange(displayData, minTimeRangePercent, maxTimeRangePercent);
    if (!displayData.length) return;

    const ctx = document.getElementById('chart').getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(0, 255, 255, 0.3)');
    gradient.addColorStop(1, 'rgba(0, 255, 255, 0.05)');

    const detectedSet = buildAnomalyLookup(detectedAnomalies, granularity);
    const referenceSet = buildAnomalyLookup(referenceAnomalies, granularity);

    const keyedDisplay = displayData.map((point) => ({
        ...point,
        chart_key: chartKey(point, granularity),
    }));

    const detectedMarkers = keyedDisplay.filter((point) => detectedSet.has(point.chart_key));
    const referenceMarkers = keyedDisplay.filter((point) => referenceSet.has(point.chart_key));
    const overlapMarkers = keyedDisplay.filter(
        (point) => detectedSet.has(point.chart_key) && referenceSet.has(point.chart_key),
    );

    const overlapSet = new Set(overlapMarkers.map((point) => point.chart_key));
    const detectedOnly = detectedMarkers.filter((point) => !overlapSet.has(point.chart_key));
    const referenceOnly = referenceMarkers.filter((point) => !overlapSet.has(point.chart_key));

    const datasets = [
        {
            label: 'Time Series',
            data: keyedDisplay.map((point) => ({ x: point.timestamp * 1000, y: point.value })),
            borderColor: '#00FFFF',
            backgroundColor: gradient,
            fill: true,
            tension: 0.1,
            pointRadius: 0,
            pointHoverRadius: 4,
        },
    ];

    if (layerVisibility.detected) {
        datasets.push({
            label: 'Detected Anomalies',
            data: detectedOnly.map((point) => ({ x: point.timestamp * 1000, y: point.value })),
            borderColor: '#FF00FF',
            backgroundColor: '#FF00FF',
            pointRadius: 5,
            pointHoverRadius: 7,
            showLine: false,
        });
    }

    if (layerVisibility.reference) {
        datasets.push({
            label: 'Reference Anomalies',
            data: referenceOnly.map((point) => ({ x: point.timestamp * 1000, y: point.value })),
            borderColor: '#FFC107',
            backgroundColor: '#FFC107',
            pointRadius: 5,
            pointHoverRadius: 7,
            showLine: false,
        });
    }

    if (layerVisibility.overlap) {
        datasets.push({
            label: 'Overlap (TP)',
            data: overlapMarkers.map((point) => ({ x: point.timestamp * 1000, y: point.value })),
            borderColor: '#FFFFFF',
            backgroundColor: 'rgba(255,255,255,0)',
            pointBorderColor: '#FFFFFF',
            pointBorderWidth: 2,
            pointRadius: 7,
            pointHoverRadius: 9,
            showLine: false,
        });
    }

    const config = {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'nearest',
                intersect: false,
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#ffffff',
                    },
                },
                tooltip: {
                    callbacks: {
                        label(context) {
                            return `${context.dataset.label}: ${context.parsed.y.toFixed(2)}`;
                        },
                        title(context) {
                            return new Date(context[0].parsed.x).toLocaleString();
                        },
                    },
                },
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        displayFormats: {
                            minute: 'HH:mm',
                            hour: 'MMM dd HH:mm',
                        },
                    },
                    title: {
                        display: true,
                        text: 'Time',
                        color: '#ffffff',
                    },
                    ticks: {
                        color: '#dddddd',
                    },
                    grid: {
                        color: 'rgba(255,255,255,0.1)',
                    },
                },
                y: {
                    title: {
                        display: true,
                        text: 'Value',
                        color: '#ffffff',
                    },
                    ticks: {
                        color: '#dddddd',
                    },
                    grid: {
                        color: 'rgba(255,255,255,0.1)',
                    },
                },
            },
            animation: {
                duration: 700,
                easing: 'easeInOutQuart',
            },
        },
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
    document.getElementById('anomalyCount').textContent = detectedAnomalies.length;
    document.getElementById('detectedCount').textContent = detectedAnomalies.length;
    document.getElementById('referenceCount').textContent = referenceAnomalies.length;
    document.getElementById('tpCount').textContent = comparisonStats.tp;
    document.getElementById('fpCount').textContent = comparisonStats.fp;
    document.getElementById('fnCount').textContent = comparisonStats.fn;
    document.getElementById('precision').textContent = formatPct(comparisonStats.precision);
    document.getElementById('recall').textContent = formatPct(comparisonStats.recall);
    document.getElementById('f1').textContent = formatPct(comparisonStats.f1);

    if (displayData.length) {
        const minTs = Math.min(...displayData.map((d) => d.timestamp));
        const maxTs = Math.max(...displayData.map((d) => d.timestamp));
        const durationHours = (maxTs - minTs) / 3600;
        document.getElementById('timeRange').textContent = `${durationHours.toFixed(1)} hours`;
    } else {
        document.getElementById('timeRange').textContent = 'N/A';
    }
}

function renderAnomalyItems(containerId, title, items, mapper) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';

    const header = document.createElement('div');
    header.className = 'anomaly-item';
    header.innerHTML = `<strong>${title} (${items.length})</strong>`;
    container.appendChild(header);

    const sliced = items.slice(0, MAX_LIST_ITEMS);
    sliced.forEach((item) => {
        const div = document.createElement('div');
        div.className = 'anomaly-item';
        div.innerHTML = mapper(item);
        container.appendChild(div);
    });

    if (items.length > MAX_LIST_ITEMS) {
        const more = document.createElement('div');
        more.className = 'anomaly-item';
        more.innerHTML = `<em>Showing first ${MAX_LIST_ITEMS} of ${items.length} items</em>`;
        container.appendChild(more);
    }
}

function updateAnomalyLists() {
    renderAnomalyItems(
        'detectedAnomalyItems',
        'Detected',
        detectedAnomalies,
        (a) => `
            <strong>Entity:</strong> ${a.entity_id}<br>
            <strong>Signal:</strong> ${a.signal_type} / ${a.metric}<br>
            <strong>Timestamp:</strong> ${new Date(a.timestamp * 1000).toLocaleString()}<br>
            <strong>Score:</strong> ${Number(a.score).toFixed(3)} (L=${Number(a.score_likelihood).toFixed(3)})
        `,
    );

    renderAnomalyItems(
        'referenceAnomalyItems',
        'Reference',
        referenceAnomalies,
        (a) => `
            <strong>Entity:</strong> ${a.entity_id}<br>
            <strong>Signal:</strong> ${a.signal_type} / ${a.metric}<br>
            <strong>Timestamp:</strong> ${new Date(a.timestamp * 1000).toLocaleString()}<br>
            <strong>Types:</strong> ${(a.anomaly_types || []).join(', ') || 'Unknown'}
        `,
    );

    const mismatches = [
        ...mismatchAnomalies.fp.map((x) => ({ ...x, mismatch_type: 'FP' })),
        ...mismatchAnomalies.fn.map((x) => ({ ...x, mismatch_type: 'FN' })),
    ];

    renderAnomalyItems(
        'mismatchAnomalyItems',
        'Mismatches',
        mismatches,
        (a) => `
            <strong>${a.mismatch_type}</strong><br>
            <strong>Entity:</strong> ${a.entity_id}<br>
            <strong>Signal:</strong> ${a.signal_type} / ${a.metric}<br>
            <strong>Timestamp:</strong> ${new Date(a.timestamp * 1000).toLocaleString()}
        `,
    );
}

function isTimingReference(record) {
    const types = record.anomaly_types || [];
    return types.some((type) => String(type).startsWith('timing_'));
}

function lowerBoundByTimestamp(entries, target) {
    let left = 0;
    let right = entries.length;
    while (left < right) {
        const mid = Math.floor((left + right) / 2);
        if (entries[mid].timestamp < target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

function findTimingMatch(entryList, timestamp, matchedRefKeys) {
    if (!entryList || !entryList.length) return null;

    const minTs = timestamp - TIMING_MATCH_WINDOW_SEC;
    const maxTs = timestamp + TIMING_MATCH_WINDOW_SEC;
    const start = lowerBoundByTimestamp(entryList, minTs);

    let best = null;
    let bestDelta = Number.POSITIVE_INFINITY;
    for (let i = start; i < entryList.length; i += 1) {
        const entry = entryList[i];
        if (entry.timestamp > maxTs) break;
        if (matchedRefKeys.has(entry.key)) continue;
        const delta = Math.abs(entry.timestamp - timestamp);
        if (delta <= TIMING_MATCH_WINDOW_SEC && delta < bestDelta) {
            best = entry;
            bestDelta = delta;
        }
    }
    return best;
}

function computeComparisonStats(detected, reference) {
    if (!detected.length && !reference.length) {
        return { stats: emptyComparisonStats(), mismatch: { fp: [], fn: [] } };
    }

    const refByKey = new Map();
    const timingRefByStream = new Map();

    reference.forEach((ref) => {
        const key = fullKey(ref);
        refByKey.set(key, ref);

        if (isTimingReference(ref)) {
            const stream = streamKey(ref);
            if (!timingRefByStream.has(stream)) {
                timingRefByStream.set(stream, []);
            }
            timingRefByStream.get(stream).push({ key, timestamp: ref.timestamp });
        }
    });

    timingRefByStream.forEach((entries) => {
        entries.sort((a, b) => a.timestamp - b.timestamp);
    });

    const matchedRefKeys = new Set();
    const fp = [];

    detected.forEach((det) => {
        const exactKey = fullKey(det);
        if (refByKey.has(exactKey) && !matchedRefKeys.has(exactKey)) {
            matchedRefKeys.add(exactKey);
            return;
        }

        const stream = streamKey(det);
        const timingEntries = timingRefByStream.get(stream);
        const timingMatch = findTimingMatch(timingEntries, det.timestamp, matchedRefKeys);
        if (timingMatch) {
            matchedRefKeys.add(timingMatch.key);
            return;
        }

        fp.push(det);
    });

    const fn = [];
    reference.forEach((ref) => {
        const key = fullKey(ref);
        if (!matchedRefKeys.has(key)) {
            fn.push(ref);
        }
    });

    const tp = matchedRefKeys.size;
    const precision = tp + fp.length > 0 ? tp / (tp + fp.length) : 0;
    const recall = tp + fn.length > 0 ? tp / (tp + fn.length) : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

    return {
        stats: {
            tp,
            fp: fp.length,
            fn: fn.length,
            precision,
            recall,
            f1,
        },
        mismatch: { fp, fn },
    };
}

window.addEventListener('resize', () => {
    if (dataset.length > 0) {
        drawChart();
    }
});

// ---------------------------------------------------------
// Model Persistence
// ---------------------------------------------------------

document.getElementById('saveModelBtn').addEventListener('click', async () => {
    const btn = document.getElementById('saveModelBtn');
    btn.textContent = 'Saving...';
    btn.disabled = true;
    try {
        const status = await fetch('http://localhost:8001/status');
        const statusData = status.ok ? await status.json() : null;
        const canFull = Boolean(statusData?.runtime?.htm_available);
        const endpoint = canFull ? 'http://localhost:8001/model/save_full' : 'http://localhost:8001/model/save';

        const response = await fetch(endpoint, { method: 'POST' });
        const { data, text } = await parseResponsePayload(response);
        if (response.ok) {
            toast(
                canFull ? 'Model saved (full HTM state)' : 'Model saved (portable, no HTM state)',
                '#22c55e'
            );
        } else {
            throw new Error(data?.message || text || 'Save failed');
        }
    } catch (e) {
        toast(`Save Error: ${e.message}`, '#f43f5e');
    } finally {
        btn.textContent = 'Save Model';
        btn.disabled = false;
    }
});

document.getElementById('loadModelBtn').addEventListener('click', async () => {
    const btn = document.getElementById('loadModelBtn');
    btn.textContent = 'Loading...';
    btn.disabled = true;
    try {
        const status = await fetch('http://localhost:8001/status');
        const statusData = status.ok ? await status.json() : null;
        const canFull = Boolean(statusData?.runtime?.htm_available);
        const endpoint = canFull ? 'http://localhost:8001/model/load_full' : 'http://localhost:8001/model/load';

        const response = await fetch(endpoint, { method: 'POST' });
        const { data, text } = await parseResponsePayload(response);
        if (response.ok) {
            const source = canFull ? 'full HTM state' : 'portable state';
            toast(`Model loaded (${source})`, '#22c55e');
            if (data?.details) {
                console.log("Loaded Model Config:", data.details);
            }
        } else {
            throw new Error(data?.message || text || 'Load failed');
        }
    } catch (e) {
        toast(`Load Error: ${e.message}`, '#f43f5e');
    } finally {
        btn.textContent = 'Load Model';
        btn.disabled = false;
    }
});

// ---------------------------------------------------------
// Simulation Logic
// ---------------------------------------------------------

let simulationChart = null;
let simulationInterval = null;
const SIM_WINDOW_SIZE = 100;

document.getElementById('startSimBtn').addEventListener('click', startSimulation);
document.getElementById('stopSimBtn').addEventListener('click', stopSimulation);

function initSimulationChart() {
    if (simulationChart) return;
    
    const ctx = document.getElementById('simulationChart').getContext('2d');
    simulationChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Simulated Stream',
                    data: [],
                    borderColor: '#2dd4bf', // accent color
                    backgroundColor: 'rgba(45, 212, 191, 0.1)',
                    borderWidth: 2,
                    tension: 0.2,
                    pointRadius: 2,
                    order: 3
                },
                {
                    label: 'Detected Anomalies',
                    data: [],
                    borderColor: '#f43f5e', // detected color (red)
                    backgroundColor: '#f43f5e',
                    pointStyle: 'circle',
                    pointRadius: 6,
                    showLine: false,
                    order: 1
                },
                {
                    label: 'Injected Truth',
                    data: [],
                    borderColor: '#ffbe2e', // reference color
                    backgroundColor: '#ffbe2e',
                    pointStyle: 'triangle',
                    pointRadius: 6,
                    showLine: false,
                    order: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: true, labels: { color: '#e9efff' } }
            },
            scales: {
                x: {
                    type: 'time',
                    time: { unit: 'minute' },
                    ticks: { color: '#a8b3d6' },
                    grid: { color: 'rgba(255,255,255,0.06)' }
                },
                y: {
                    ticks: { color: '#a8b3d6' },
                    grid: { color: 'rgba(255,255,255,0.06)' }
                }
            }
        }
    });
}

function updateSimulationChart(point, result) {
    if (!simulationChart) initSimulationChart();
    
    const timestamp = new Date(point.timestamp * 1000);
    
    // 0. Stream Value
    simulationChart.data.datasets[0].data.push({ x: timestamp, y: point.value });
    
    // 1. Detected Anomaly
    if (result.anomaly_flag) {
        simulationChart.data.datasets[1].data.push({ x: timestamp, y: point.value });
    } else {
        simulationChart.data.datasets[1].data.push({ x: timestamp, y: null });
    }

    // 2. Injected Truth (from generator label)
    if (point.label === 1) {
        simulationChart.data.datasets[2].data.push({ x: timestamp, y: point.value });
    } else {
        simulationChart.data.datasets[2].data.push({ x: timestamp, y: null });
    }
    
    // Maintain window size
    if (simulationChart.data.datasets[0].data.length > SIM_WINDOW_SIZE) {
        simulationChart.data.datasets[0].data.shift();
        simulationChart.data.datasets[1].data.shift();
        simulationChart.data.datasets[2].data.shift();
    }
    
    simulationChart.update();
}

async function startSimulation() {
    const btn = document.getElementById('startSimBtn');
    btn.classList.add('loading');
    
    try {
        // Start backend simulation
        const response = await fetch('http://localhost:8001/simulation/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                metric: 'sendmsg', // could be dynamic
                entity_id: 'web_server_01'
            })
        });
        
        if (!response.ok) throw new Error('Failed to start');
        
        // Update UI
        document.getElementById('startSimBtn').style.display = 'none';
        document.getElementById('stopSimBtn').style.display = 'inline-block';
        document.getElementById('simStatus').textContent = 'Running (Network Event / sendmsg)';
        document.getElementById('simStatus').style.color = '#2dd4bf';
        
        initSimulationChart();
        
        // Start loop
        simulationInterval = setInterval(fetchNextSimulationPoint, 1000); // 1 sec interval
        
    } catch (error) {
        toast(`Simulation Error: ${error.message}`, 'linear-gradient(to right, #f43f5e, #e11d48)');
    } finally {
        btn.classList.remove('loading');
    }
}

async function stopSimulation() {
    if (simulationInterval) clearInterval(simulationInterval);
    
    try {
        await fetch('http://localhost:8001/simulation/stop', { method: 'POST' });
        
        document.getElementById('startSimBtn').style.display = 'inline-block';
        document.getElementById('stopSimBtn').style.display = 'none';
        document.getElementById('simStatus').textContent = 'Inactive';
        document.getElementById('simStatus').style.color = '#a8b3d6';
        
    } catch (error) {
        console.error("Stop failed", error);
    }
}

async function fetchNextSimulationPoint() {
    try {
        // 1. Generate Point
        const genResponse = await fetch('http://localhost:8001/simulation/generate');
        if (!genResponse.ok) {
            stopSimulation(); 
            return;
        }
        
        const genData = await genResponse.json();
        const point = genData.point;
        
        // 2. Detect Anomaly (Standard API call)
        const learningEnabled = document.getElementById('learningToggle').checked;
        const detectResponse = await fetch('http://localhost:8001/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                point: point,
                learn: learningEnabled
            })
        });
        
        let result = { anomaly_flag: false }; // Default
        if (detectResponse.ok) {
            // Note: /detect response structure is flat, not nested in 'result'
            result = await detectResponse.json(); 
        } else {
            console.error("Detection failed");
        }
        
        // 3. Update Chart
        updateSimulationChart(point, result);
        
    } catch (error) {
        console.error("Loop error", error);
    }
}
