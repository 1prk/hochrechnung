// Counter Verification App

let map;
let countersData = [];
let metadata = {};
let severityConfig = {};
let selectedCounter = null;
let changes = {};

// Default severity colors (overridden by server config)
const DEFAULT_SEVERITY_COLORS = {
    critical: '#dc3545',
    ambiguous: '#6f42c1',
    no_volume: '#0d6efd',
    warning: '#ffc107',
    campaign_bias: '#fd7e14',
    carryover: '#6c757d',
    ok: '#6c757d',
    verified: '#28a745'
};

// Get color for severity level
function getSeverityColor(severity) {
    if (severityConfig[severity]) {
        return severityConfig[severity].color;
    }
    return DEFAULT_SEVERITY_COLORS[severity] || '#6c757d';
}

// Get label for severity level
function getSeverityLabel(severity) {
    if (severityConfig[severity]) {
        return severityConfig[severity].label;
    }
    // Fallback: capitalize and replace underscores
    return severity.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// Initialize app
async function init() {
    try {
        // Load verification data
        const response = await fetch('/api/verification-data');
        const data = await response.json();

        metadata = data.metadata;
        severityConfig = data.severity_config || {};
        countersData = data.counters;

        // Update header stats
        document.getElementById('year').textContent = metadata.year;
        document.getElementById('stat-total').textContent = metadata.n_counters;
        document.getElementById('stat-flagged').textContent = metadata.n_flagged;
        document.getElementById('stat-ratio').textContent = metadata.median_ratio.toFixed(2);

        // Render severity breakdown
        renderSeverityBreakdown();

        // Initialize map
        initMap();

        // Render counter list
        renderCounterList();

    } catch (error) {
        console.error('Failed to load data:', error);
        showToast('Failed to load verification data', 'error');
    }
}

// Initialize MapLibre map
function initMap() {
    // Start with just OSM base layer - volumes layer added later if available
    map = new maplibregl.Map({
        container: 'map',
        style: {
            version: 8,
            sources: {
                'osm': {
                    type: 'raster',
                    tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
                    tileSize: 256,
                    attribution: '© OpenStreetMap contributors'
                }
            },
            layers: [
                {
                    id: 'osm',
                    type: 'raster',
                    source: 'osm'
                }
            ]
        },
        center: [8.6821, 50.1109], // Hessen center
        zoom: 8
    });

    map.addControl(new maplibregl.NavigationControl());

    map.on('load', () => {
        // Add counters as circle layer (always works)
        addCountersToMap();

        // Try to add volumes layer (may fail if MBTiles unavailable)
        tryAddVolumesLayer();

        // Add click handler for counters
        map.on('click', 'counters', onCounterClick);

        // Fit bounds to flagged counters if any
        const flagged = countersData.filter(c => c.is_outlier);
        if (flagged.length > 0) {
            const bounds = new maplibregl.LngLatBounds();
            flagged.forEach(c => bounds.extend([c.longitude, c.latitude]));
            map.fitBounds(bounds, { padding: 50 });
        }
    });
}

// Add counters to map
function addCountersToMap() {
    const geojson = {
        type: 'FeatureCollection',
        features: countersData.map(c => ({
            type: 'Feature',
            geometry: {
                type: 'Point',
                coordinates: [c.longitude, c.latitude]
            },
            properties: {
                counter_id: c.counter_id,
                name: c.name,
                dtv: c.dtv,
                count: c.count,
                ratio: c.ratio,
                flag_severity: c.flag_severity,
                is_outlier: c.is_outlier,
                verification_status: c.verification_status
            }
        }))
    };

    map.addSource('counters', {
        type: 'geojson',
        data: geojson
    });

    // Add circles with color based on severity
    map.addLayer({
        id: 'counters',
        type: 'circle',
        source: 'counters',
        paint: {
            'circle-radius': [
                'interpolate',
                ['linear'],
                ['zoom'],
                8, 6,
                16, 14
            ],
            'circle-color': [
                'case',
                ['==', ['get', 'flag_severity'], 'critical'], getSeverityColor('critical'),
                ['==', ['get', 'flag_severity'], 'ambiguous'], getSeverityColor('ambiguous'),
                ['==', ['get', 'flag_severity'], 'no_volume'], getSeverityColor('no_volume'),
                ['==', ['get', 'flag_severity'], 'warning'], getSeverityColor('warning'),
                ['==', ['get', 'flag_severity'], 'campaign_bias'], getSeverityColor('campaign_bias'),
                ['==', ['get', 'flag_severity'], 'verified'], getSeverityColor('verified'),
                ['==', ['get', 'verification_status'], 'verified'], getSeverityColor('verified'),
                getSeverityColor('ok')
            ],
            'circle-stroke-color': '#ffffff',
            'circle-stroke-width': 2,
            'circle-opacity': 0.9
        }
    });

    // Add tooltips
    const popup = new maplibregl.Popup({
        closeButton: false,
        closeOnClick: false
    });

    map.on('mouseenter', 'counters', (e) => {
        map.getCanvas().style.cursor = 'pointer';

        const props = e.features[0].properties;
        const html = `
            <div style="font-size: 0.75rem;">
                <strong>${props.counter_id}</strong> - ${props.name}<br>
                DTV: ${props.dtv}<br>
                Volume: ${props.count}<br>
                Ratio: ${props.ratio ? props.ratio.toFixed(2) : 'N/A'}
            </div>
        `;

        popup.setLngLat(e.lngLat).setHTML(html).addTo(map);
    });

    map.on('mouseleave', 'counters', () => {
        map.getCanvas().style.cursor = '';
        popup.remove();
    });
}

// Try to add volumes layer (gracefully handles missing MBTiles)
function tryAddVolumesLayer() {
    // Check if MBTiles is available by fetching metadata
    const tilesUrl = window.location.origin + '/tiles/{z}/{x}/{y}.pbf';

    map.addSource('volumes', {
        type: 'vector',
        tiles: [tilesUrl],
        minzoom: 10,
        maxzoom: 16
    });

    map.addLayer({
        id: 'volumes-line',
        type: 'line',
        source: 'volumes',
        'source-layer': 'volumes',
        paint: {
            'line-color': '#3388ff',
            // Line width based on volume (absolute scale, not relative)
            // Uses step function for clear visual distinction
            'line-width': [
                'step',
                ['get', 'count'],
                1,       // default: count < 10 → width 1
                10, 2,   // count >= 10 → width 2
                50, 3,   // count >= 50 → width 3
                100, 4,  // count >= 100 → width 4
                250, 5,  // count >= 250 → width 5
                500, 6,  // count >= 500 → width 6
                1000, 8  // count >= 1000 → width 8
            ],
            'line-opacity': 0.8
        }
    });

    // Add click handler for volumes
    map.on('click', 'volumes-line', onEdgeClick);

    // Add hover effects for volumes
    map.on('mouseenter', 'volumes-line', () => {
        map.getCanvas().style.cursor = 'pointer';
    });
    map.on('mouseleave', 'volumes-line', () => {
        map.getCanvas().style.cursor = '';
    });

    // Edge tooltips with directional info
    const edgePopup = new maplibregl.Popup({
        closeButton: false,
        closeOnClick: false
    });

    // Store current arrow layer for cleanup
    let currentArrowLayer = null;

    map.on('mouseenter', 'volumes-line', (e) => {
        const props = e.features[0].properties;
        const geometry = e.features[0].geometry;

        // Build directional counts display if available
        let directionalHtml = '';
        if (props.count_forward !== undefined || props.count_backward !== undefined) {
            const fwd = props.count_forward ?? 'N/A';
            const bwd = props.count_backward ?? 'N/A';
            directionalHtml = `
                <div style="margin-top: 4px; border-top: 1px solid #ddd; padding-top: 4px;">
                    <span style="color: #28a745;">→ Forward: ${fwd}</span><br>
                    <span style="color: #dc3545;">← Backward: ${bwd}</span>
                </div>
            `;
        }

        const html = `
            <div style="font-size: 0.75rem;">
                <strong>OSM Way ${props.base_id}</strong><br>
                Volume: ${props.count}<br>
                Infrastructure: ${props.bicycle_infrastructure || 'N/A'}
                ${directionalHtml}
            </div>
        `;
        edgePopup.setLngLat(e.lngLat).setHTML(html).addTo(map);

        // Add arrow layer to show edge direction if directional counts exist
        if ((props.count_forward !== undefined || props.count_backward !== undefined) &&
            geometry.type === 'LineString' && geometry.coordinates.length >= 2) {
            addDirectionArrows(geometry.coordinates, props);
        }
    });

    map.on('mouseleave', 'volumes-line', () => {
        edgePopup.remove();
        removeDirectionArrows();
    });

    // Suppress tile loading errors (expected when MBTiles unavailable)
    map.on('error', (e) => {
        if (e.sourceId === 'volumes') {
            // Silently ignore volume tile errors
            return;
        }
        console.error('Map error:', e);
    });
}



// Render severity breakdown in header
function renderSeverityBreakdown() {
    const container = document.getElementById('severity-breakdown');
    if (!container || !metadata.n_by_severity) return;

    const severityOrder = ['critical', 'ambiguous', 'no_volume', 'warning', 'campaign_bias', 'carryover', 'ok', 'verified'];
    const items = [];

    for (const severity of severityOrder) {
        const count = metadata.n_by_severity[severity];
        if (count && count > 0) {
            const color = getSeverityColor(severity);
            const label = getSeverityLabel(severity);
            items.push(`<span class="severity-item" style="color: ${color};" title="${label}">${label}: ${count}</span>`);
        }
    }

    container.innerHTML = items.join(' | ');
}

// Render counter list
function renderCounterList() {
    const listEl = document.getElementById('counter-list');

    if (countersData.length === 0) {
        listEl.innerHTML = `
            <div class="empty-state">
                <div>No counters to verify</div>
            </div>
        `;
        return;
    }

    listEl.innerHTML = '';

    countersData.forEach(counter => {
        const item = document.createElement('div');
        item.className = `counter-item ${counter.flag_severity}`;
        if (counter.verification_status === 'verified') {
            item.classList.add('verified');
        }
        if (counter.is_discarded) {
            item.classList.add('discarded');
        }

        // Determine badge based on flag_severity
        const severity = counter.flag_severity || 'ok';
        const showBadge = severity !== 'ok';
        const badgeColor = getSeverityColor(severity);
        const badgeText = getSeverityLabel(severity);

        // Count candidates for ambiguous display
        const candidateCount = counter.candidate_edges ? counter.candidate_edges.length : 0;
        const candidateInfo = candidateCount > 1 ? ` (${candidateCount} candidates)` : '';

        item.innerHTML = `
            <div class="counter-header">
                <span class="counter-id">${counter.counter_id}</span>
                ${showBadge ? `<span class="counter-badge" style="background-color: ${badgeColor};">${badgeText}${severity === 'ambiguous' ? candidateInfo : ''}</span>` : ''}
            </div>
            <div class="counter-name">${counter.name}</div>
            <div class="counter-stats">
                <div class="stat-row">
                    <label>DTV:</label>
                    <value>${counter.dtv || 'N/A'}</value>
                </div>
                <div class="stat-row">
                    <label>Volume:</label>
                    <value>${counter.count !== null ? counter.count : 'N/A'}</value>
                </div>
                <div class="stat-row">
                    <label>Ratio:</label>
                    <value>${counter.ratio ? counter.ratio.toFixed(2) : 'N/A'}</value>
                </div>
                <div class="stat-row">
                    <label>OSM ID:</label>
                    <value>${counter.base_id || 'N/A'}</value>
                </div>
            </div>
        `;

        item.addEventListener('click', () => selectCounter(counter));
        listEl.appendChild(item);
    });
}

// Select counter
function selectCounter(counter) {
    selectedCounter = counter;

    // Update UI
    document.querySelectorAll('.counter-item').forEach(el => el.classList.remove('active'));
    event.currentTarget?.classList.add('active');

    // Show editor
    const editorEl = document.getElementById('counter-editor');
    editorEl.style.display = 'block';

    document.getElementById('edit-counter-id').textContent = counter.counter_id;
    document.getElementById('edit-base-id').value = changes[counter.counter_id]?.base_id || counter.base_id || '';
    document.getElementById('edit-count').value = changes[counter.counter_id]?.count || counter.count || '';
    document.getElementById('edit-metadata').value = changes[counter.counter_id]?.metadata || counter.verification_metadata || '';

    // Set discard checkbox state
    const isDiscarded = changes[counter.counter_id]?.is_discarded ?? counter.is_discarded ?? false;
    document.getElementById('edit-discard').checked = isDiscarded;

    // Show severity info
    const severityInfoEl = document.getElementById('severity-info');
    if (severityInfoEl) {
        const severity = counter.flag_severity || 'ok';
        const color = getSeverityColor(severity);
        const label = getSeverityLabel(severity);
        severityInfoEl.innerHTML = `<span style="color: ${color}; font-weight: bold;">${label}</span>`;
    }

    // Render candidate edges if available
    renderCandidateEdges(counter);

    // Load and render counter images
    loadAndRenderImages(counter);

    // Fly to counter on map
    if (map && counter.longitude && counter.latitude) {
        map.flyTo({
            center: [counter.longitude, counter.latitude],
            zoom: 15,
            duration: 1000
        });
    }
}

// Render candidate edges dropdown
function renderCandidateEdges(counter) {
    const container = document.getElementById('candidate-edges');
    if (!container) return;

    const candidates = counter.candidate_edges || [];

    if (candidates.length === 0) {
        container.innerHTML = '<div class="no-candidates">No candidate edges found</div>';
        return;
    }

    let html = '<div class="candidates-label">Candidate Edges (click to select):</div>';
    html += '<div class="candidates-list">';

    candidates.forEach((candidate, index) => {
        const isSelected = candidate.base_id === counter.base_id;
        const distance = candidate.distance ? candidate.distance.toFixed(1) : '?';
        const count = candidate.count !== null && candidate.count !== undefined ? candidate.count : 'N/A';
        const infra = candidate.bicycle_infrastructure || 'unknown';

        html += `
            <div class="candidate-item ${isSelected ? 'selected' : ''}"
                 onclick="selectCandidate(${candidate.base_id}, ${candidate.count || 'null'})"
                 title="Click to select this edge">
                <span class="candidate-rank">#${index + 1}</span>
                <span class="candidate-info">
                    <strong>OSM ${candidate.base_id}</strong> -
                    ${distance}m away,
                    Vol: ${count},
                    ${infra}
                </span>
            </div>
        `;
    });

    html += '</div>';
    container.innerHTML = html;
}

// Select a candidate edge
function selectCandidate(baseId, count) {
    document.getElementById('edit-base-id').value = baseId;
    if (count !== null) {
        document.getElementById('edit-count').value = count;
    }
    showToast(`Selected edge ${baseId}`, 'success');

    // Update visual selection in candidates list
    document.querySelectorAll('.candidate-item').forEach(el => el.classList.remove('selected'));
    event.currentTarget?.classList.add('selected');
}

// Handle counter click on map
function onCounterClick(e) {
    const props = e.features[0].properties;
    const counter = countersData.find(c => c.counter_id === props.counter_id);
    if (counter) {
        selectCounter(counter);

        // Scroll to counter in list
        const listItems = document.querySelectorAll('.counter-item');
        const index = countersData.indexOf(counter);
        if (listItems[index]) {
            listItems[index].scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }
}

// Handle edge click on map
function onEdgeClick(e) {
    if (!selectedCounter) {
        showToast('Select a counter first', 'error');
        return;
    }

    const props = e.features[0].properties;

    // Fill in base_id and count from clicked edge
    document.getElementById('edit-base-id').value = props.base_id;
    document.getElementById('edit-count').value = props.count;

    showToast(`Selected edge ${props.base_id}`, 'success');
}

// Save counter changes
async function saveCounter() {
    if (!selectedCounter) return;

    const base_id = parseInt(document.getElementById('edit-base-id').value);
    const count = document.getElementById('edit-count').value ? parseInt(document.getElementById('edit-count').value) : null;
    const metadata = document.getElementById('edit-metadata').value;
    const is_discarded = document.getElementById('edit-discard').checked;

    if (!base_id) {
        showToast('OSM Way ID is required', 'error');
        return;
    }

    // Store changes
    changes[selectedCounter.counter_id] = {
        counter_id: selectedCounter.counter_id,
        base_id: base_id,
        count: count,
        metadata: metadata,
        is_discarded: is_discarded
    };

    // Update counter data
    const counter = countersData.find(c => c.counter_id === selectedCounter.counter_id);
    if (counter) {
        counter.base_id = base_id;
        if (count) counter.count = count;
        counter.verification_metadata = metadata;
        counter.verification_status = 'verified';
        counter.flag_severity = 'verified';
        counter.is_discarded = is_discarded;
    }

    // Save to backend
    try {
        const response = await fetch('/api/save-corrections', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ changes: Object.values(changes) })
        });

        if (!response.ok) throw new Error('Save failed');

        showToast('Counter saved successfully', 'success');

        // Re-render list and map
        renderCounterList();
        updateMapSource();

        // Clear editor
        cancelEdit();

    } catch (error) {
        console.error('Save failed:', error);
        showToast('Failed to save changes', 'error');
    }
}

// Cancel edit
function cancelEdit() {
    selectedCounter = null;
    document.getElementById('counter-editor').style.display = 'none';
    hideImagePanel();
    document.querySelectorAll('.counter-item').forEach(el => el.classList.remove('active'));
}

// Update map source
function updateMapSource() {
    if (!map.getSource('counters')) return;

    const geojson = {
        type: 'FeatureCollection',
        features: countersData.map(c => ({
            type: 'Feature',
            geometry: {
                type: 'Point',
                coordinates: [c.longitude, c.latitude]
            },
            properties: {
                counter_id: c.counter_id,
                name: c.name,
                dtv: c.dtv,
                count: c.count,
                ratio: c.ratio,
                flag_severity: c.flag_severity,
                is_outlier: c.is_outlier,
                verification_status: c.verification_status
            }
        }))
    };

    map.getSource('counters').setData(geojson);
}

// Show toast notification
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;

    document.body.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 3000);
}

// Direction arrow helpers for edge visualization
function addDirectionArrows(coordinates, props) {
    // Remove any existing arrows first
    removeDirectionArrows();

    if (!coordinates || coordinates.length < 2) return;

    // Place arrows at fixed intervals along the edge (like o-->-->-->-->--o)
    const arrowFeatures = [];

    // Arrow size scales with zoom level
    const zoom = map.getZoom();
    const baseSize = 0.00006;
    const size = baseSize * Math.pow(2, 16 - zoom);

    // Calculate total line length and place arrows every ~30 meters
    const totalLength = getLineLength(coordinates);
    const intervalM = 30; // meters between arrows
    const numArrows = Math.max(1, Math.floor(totalLength / intervalM));

    // Distribute arrows evenly, avoiding endpoints (15% to 85% of line)
    for (let i = 0; i < numArrows; i++) {
        const fraction = 0.15 + (0.7 * (i + 0.5) / numArrows);
        const point = getPointAlongLine(coordinates, fraction);
        const lineBearing = getBearingAtPoint(coordinates, fraction);

        if (point && lineBearing !== null) {
            const arrowCoords = createArrowPolygon(point, lineBearing, size);
            arrowFeatures.push({
                type: 'Feature',
                geometry: { type: 'Polygon', coordinates: [arrowCoords] },
                properties: { direction: 'forward' }
            });
        }
    }

    if (arrowFeatures.length === 0) return;

    // Add arrow source and layer
    map.addSource('direction-arrows', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: arrowFeatures }
    });

    map.addLayer({
        id: 'direction-arrows-layer',
        type: 'fill',
        source: 'direction-arrows',
        paint: {
            'fill-color': '#28a745',
            'fill-opacity': 0.9
        }
    });

    // Add outline for visibility
    map.addLayer({
        id: 'direction-arrows-outline',
        type: 'line',
        source: 'direction-arrows',
        paint: {
            'line-color': '#ffffff',
            'line-width': 1,
            'line-opacity': 0.8
        }
    });
}

// Create arrow polygon coordinates pointing in given bearing
function createArrowPolygon(center, bearingDeg, size) {
    // Arrow pointing in bearing direction
    // Triangle with tip pointing forward
    const tipBearing = bearingDeg;
    const leftBearing = (bearingDeg + 140) % 360;
    const rightBearing = (bearingDeg - 140 + 360) % 360;

    const tip = offsetPoint(center, tipBearing, size);
    const left = offsetPoint(center, leftBearing, size );
    const right = offsetPoint(center, rightBearing, size );

    return [tip, left, right, tip]; // Closed polygon
}

// Offset a point by distance in a bearing direction
function offsetPoint(point, bearingDeg, distance) {
    const bearingRad = bearingDeg * Math.PI / 180;
    // Approximate: 1 degree ~ 111km at equator, less at higher latitudes
    const latFactor = Math.cos(point[1] * Math.PI / 180);
    return [
        point[0] + distance * Math.sin(bearingRad) / latFactor,
        point[1] + distance * Math.cos(bearingRad)
    ];
}

function removeDirectionArrows() {
    if (map.getLayer('direction-arrows-outline')) {
        map.removeLayer('direction-arrows-outline');
    }
    if (map.getLayer('direction-arrows-layer')) {
        map.removeLayer('direction-arrows-layer');
    }
    if (map.getSource('direction-arrows')) {
        map.removeSource('direction-arrows');
    }
}

// Get total line length in meters
function getLineLength(coordinates) {
    if (!coordinates || coordinates.length < 2) return 0;
    let totalLength = 0;
    for (let i = 0; i < coordinates.length - 1; i++) {
        totalLength += distance(coordinates[i], coordinates[i + 1]);
    }
    return totalLength;
}

// Get point at fraction along line
function getPointAlongLine(coordinates, fraction) {
    if (!coordinates || coordinates.length < 2) return null;

    // Calculate total length
    let totalLength = 0;
    const segmentLengths = [];
    for (let i = 0; i < coordinates.length - 1; i++) {
        const len = distance(coordinates[i], coordinates[i + 1]);
        segmentLengths.push(len);
        totalLength += len;
    }

    const targetLength = totalLength * fraction;
    let accumulatedLength = 0;

    for (let i = 0; i < segmentLengths.length; i++) {
        if (accumulatedLength + segmentLengths[i] >= targetLength) {
            const remainingFraction = (targetLength - accumulatedLength) / segmentLengths[i];
            const p1 = coordinates[i];
            const p2 = coordinates[i + 1];
            return [
                p1[0] + (p2[0] - p1[0]) * remainingFraction,
                p1[1] + (p2[1] - p1[1]) * remainingFraction
            ];
        }
        accumulatedLength += segmentLengths[i];
    }

    return coordinates[coordinates.length - 1];
}

// Get bearing at fraction along line
function getBearingAtPoint(coordinates, fraction) {
    if (!coordinates || coordinates.length < 2) return null;

    // Find the segment containing the point
    let totalLength = 0;
    const segmentLengths = [];
    for (let i = 0; i < coordinates.length - 1; i++) {
        const len = distance(coordinates[i], coordinates[i + 1]);
        segmentLengths.push(len);
        totalLength += len;
    }

    const targetLength = totalLength * fraction;
    let accumulatedLength = 0;

    for (let i = 0; i < segmentLengths.length; i++) {
        if (accumulatedLength + segmentLengths[i] >= targetLength) {
            return bearing(coordinates[i], coordinates[i + 1]);
        }
        accumulatedLength += segmentLengths[i];
    }

    return bearing(coordinates[coordinates.length - 2], coordinates[coordinates.length - 1]);
}

// Haversine distance between two points (in meters)
function distance(p1, p2) {
    const R = 6371000; // Earth radius in meters
    const lat1 = p1[1] * Math.PI / 180;
    const lat2 = p2[1] * Math.PI / 180;
    const dLat = (p2[1] - p1[1]) * Math.PI / 180;
    const dLon = (p2[0] - p1[0]) * Math.PI / 180;

    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
              Math.cos(lat1) * Math.cos(lat2) *
              Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

    return R * c;
}

// Calculate bearing from p1 to p2 (in degrees)
function bearing(p1, p2) {
    const lon1 = p1[0] * Math.PI / 180;
    const lon2 = p2[0] * Math.PI / 180;
    const lat1 = p1[1] * Math.PI / 180;
    const lat2 = p2[1] * Math.PI / 180;

    const dLon = lon2 - lon1;
    const x = Math.sin(dLon) * Math.cos(lat2);
    const y = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);

    let bearing = Math.atan2(x, y) * 180 / Math.PI;
    return (bearing + 360) % 360;
}

// Load and render counter location images in map overlay
async function loadAndRenderImages(counter) {
    const container = document.getElementById('counter-images');
    if (!container) return;

    const imageId = counter.original_id || counter.counter_id;

    if (!counter.image_count || counter.image_count === 0) {
        container.style.display = 'none';
        return;
    }

    container.style.display = 'block';
    container.innerHTML = '<div class="images-panel-header"><span class="images-panel-title">Laden...</span></div>';

    try {
        const response = await fetch(`/api/images/${imageId}`);
        if (!response.ok) {
            container.style.display = 'none';
            return;
        }

        const images = await response.json();
        if (images.length === 0) {
            container.style.display = 'none';
            return;
        }

        let html = '<div class="images-panel-header">';
        html += `<span class="images-panel-title">Standortbilder (${images.length})</span>`;
        html += '<button class="images-panel-close" onclick="hideImagePanel()" title="Schließen">&times;</button>';
        html += '</div>';
        html += '<div class="images-gallery">';
        images.forEach(img => {
            const url = `/api/images/${imageId}/${img.id}`;
            html += `<img class="image-thumb" src="${url}" alt="${img.filename || ''}" onclick="openImageModal('${url}', '${(img.filename || '').replace(/'/g, "\\'")}')" loading="lazy">`;
        });
        html += '</div>';
        container.innerHTML = html;

    } catch (error) {
        console.error('Failed to load images:', error);
        container.style.display = 'none';
    }
}

// Hide image panel
function hideImagePanel() {
    const container = document.getElementById('counter-images');
    if (container) container.style.display = 'none';
}

// Open full-size image in modal overlay
function openImageModal(url, filename) {
    const modal = document.createElement('div');
    modal.className = 'image-modal';
    modal.innerHTML = `
        <img src="${url}" alt="${filename}">
        ${filename ? `<div class="image-modal-filename">${filename}</div>` : ''}
    `;
    modal.addEventListener('click', () => modal.remove());
    document.body.appendChild(modal);
}

// Initialize on load
document.addEventListener('DOMContentLoaded', init);
