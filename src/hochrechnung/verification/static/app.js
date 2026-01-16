// Counter Verification App

let map;
let countersData = [];
let metadata = {};
let selectedCounter = null;
let changes = {};

// Initialize app
async function init() {
    try {
        // Load verification data
        const response = await fetch('/api/verification-data');
        const data = await response.json();

        metadata = data.metadata;
        countersData = data.counters;

        // Update header stats
        document.getElementById('year').textContent = metadata.year;
        document.getElementById('stat-total').textContent = metadata.n_counters;
        document.getElementById('stat-flagged').textContent = metadata.n_flagged;
        document.getElementById('stat-ratio').textContent = metadata.median_ratio.toFixed(2);

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
                },
                'volumes': {
                    type: 'vector',
                    tiles: [window.location.origin + '/tiles/{z}/{x}/{y}.pbf'],
                    minzoom: 10,
                    maxzoom: 16
                }
            },
            layers: [
                {
                    id: 'osm',
                    type: 'raster',
                    source: 'osm'
                },
                {
                    id: 'volumes-line',
                    type: 'line',
                    source: 'volumes',
                    'source-layer': 'volumes',
                    paint: {
                        'line-color': '#3388ff',
                        'line-width': 2,
                        'line-opacity': 0.7
                    }
                }
            ]
        },
        center: [8.6821, 50.1109], // Hessen center
        zoom: 8
    });

    map.addControl(new maplibregl.NavigationControl());

    map.on('load', () => {
        // Add counters as circle layer
        addCountersToMap();

        // Add click handlers
        map.on('click', 'counters', onCounterClick);
        map.on('click', 'volumes-line', onEdgeClick);

        // Add hover effects
        map.on('mouseenter', 'volumes-line', () => {
            map.getCanvas().style.cursor = 'pointer';
        });
        map.on('mouseleave', 'volumes-line', () => {
            map.getCanvas().style.cursor = '';
        });

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
                8, 4,
                16, 12
            ],
            'circle-color': [
                'match',
                ['get', 'flag_severity'],
                'critical', '#dc3545',
                'warning', '#ffc107',
                ['==', ['get', 'verification_status'], 'verified'], '#28a745',
                '#6c757d'
            ],
            'circle-stroke-color': '#ffffff',
            'circle-stroke-width': 2,
            'circle-opacity': 0.8
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

    // Edge tooltips
    const edgePopup = new maplibregl.Popup({
        closeButton: false,
        closeOnClick: false
    });

    map.on('mouseenter', 'volumes-line', (e) => {
        const props = e.features[0].properties;
        const html = `
            <div style="font-size: 0.75rem;">
                <strong>OSM Way ${props.base_id}</strong><br>
                Volume: ${props.count}<br>
                Infrastructure: ${props.bicycle_infrastructure || 'N/A'}
            </div>
        `;

        edgePopup.setLngLat(e.lngLat).setHTML(html).addTo(map);
    });

    map.on('mouseleave', 'volumes-line', () => {
        edgePopup.remove();
    });
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

        const badgeClass = counter.flag_severity === 'critical' ? 'badge-critical' :
                          counter.flag_severity === 'warning' ? 'badge-warning' :
                          counter.verification_status === 'verified' ? 'badge-verified' : '';

        const badgeText = counter.verification_status === 'verified' ? 'Verified' :
                         counter.flag_severity === 'critical' ? 'Critical' :
                         counter.flag_severity === 'warning' ? 'Warning' : '';

        item.innerHTML = `
            <div class="counter-header">
                <span class="counter-id">${counter.counter_id}</span>
                ${badgeText ? `<span class="counter-badge ${badgeClass}">${badgeText}</span>` : ''}
            </div>
            <div class="counter-name">${counter.name}</div>
            <div class="counter-stats">
                <div class="stat-row">
                    <label>DTV:</label>
                    <value>${counter.dtv || 'N/A'}</value>
                </div>
                <div class="stat-row">
                    <label>Volume:</label>
                    <value>${counter.count || 'N/A'}</value>
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

    // Fly to counter on map
    if (map && counter.longitude && counter.latitude) {
        map.flyTo({
            center: [counter.longitude, counter.latitude],
            zoom: 15,
            duration: 1000
        });
    }
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

    if (!base_id) {
        showToast('OSM Way ID is required', 'error');
        return;
    }

    // Store changes
    changes[selectedCounter.counter_id] = {
        counter_id: selectedCounter.counter_id,
        base_id: base_id,
        count: count,
        metadata: metadata
    };

    // Update counter data
    const counter = countersData.find(c => c.counter_id === selectedCounter.counter_id);
    if (counter) {
        counter.base_id = base_id;
        if (count) counter.count = count;
        counter.verification_metadata = metadata;
        counter.verification_status = 'verified';
        counter.flag_severity = 'verified';
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

// Initialize on load
document.addEventListener('DOMContentLoaded', init);
