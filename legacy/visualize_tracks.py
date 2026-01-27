import json
import folium
from folium import plugins
import numpy as np

def load_data(filepath):
    """Load JSONL data file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def convert_to_latlon(x, y, origin_lat=40.0, origin_lon=-75.0):
    """
    Convert local coordinates (meters) to lat/lon.
    Approximate conversion for visualization purposes.
    """
    # Rough conversion: 1 degree latitude ≈ 111,000 meters
    # 1 degree longitude ≈ 111,000 * cos(latitude) meters
    lat = origin_lat + (y / 111000.0)
    lon = origin_lon + (x / (111000.0 * np.cos(np.radians(origin_lat))))
    return lat, lon

def create_track_map(data_file, output_html='track_map.html'):
    """Create an interactive map showing tracks and plots."""
    
    # Load data
    print(f"Loading data from {data_file}...")
    frames = load_data(data_file)
    
    # Organize measurements by track_id
    tracks = {}  # track_id -> list of measurements
    clutter = []  # track_id == -1
    
    for frame in frames:
        timestamp = frame['timestamp']
        for measurement in frame['measurements']:
            track_id = measurement['track_id']
            
            # Add timestamp to measurement
            measurement['frame_timestamp'] = timestamp
            
            if track_id == -1:
                clutter.append(measurement)
            else:
                if track_id not in tracks:
                    tracks[track_id] = []
                tracks[track_id].append(measurement)
    
    print(f"Found {len(tracks)} tracks and {len(clutter)} clutter measurements")
    
    # Create map centered on the data
    # Calculate center from first few measurements
    all_measurements = []
    for measurements in list(tracks.values())[:5]:
        all_measurements.extend(measurements[:5])
    
    if all_measurements:
        avg_x = np.mean([m['x'] for m in all_measurements])
        avg_y = np.mean([m['y'] for m in all_measurements])
        center_lat, center_lon = convert_to_latlon(avg_x, avg_y)
    else:
        center_lat, center_lon = 40.0, -75.0
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles='OpenStreetMap'
    )
    
    # Color palette for tracks
    colors = [
        'red', 'blue', 'green', 'purple', 'orange', 'darkred',
        'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
        'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray',
        'black', 'lightgray'
    ]
    
    # Create feature groups for each track
    for track_id, measurements_list in sorted(tracks.items()):
        color = colors[track_id % len(colors)]
        
        # Create feature group for this track
        track_group = folium.FeatureGroup(name=f'Track {track_id}')
        
        # Sort measurements by timestamp
        measurements = sorted(measurements_list, key=lambda m: m['frame_timestamp'])
        
        # Create path line
        path_coords = []
        for meas in measurements:
            lat, lon = convert_to_latlon(meas['x'], meas['y'])
            path_coords.append([lat, lon])
        
        # Add polyline for track path
        if len(path_coords) > 1:
            folium.PolyLine(
                path_coords,
                color=color,
                weight=2,
                opacity=0.7,
                popup=f'Track {track_id}'
            ).add_to(track_group)
        
        # Add markers for each measurement
        for i, meas in enumerate(measurements):
            lat, lon = convert_to_latlon(meas['x'], meas['y'])
            
            # Create popup text
            popup_text = f"""
            <b>Track ID:</b> {track_id}<br>
            <b>Sensor:</b> {meas['sensor_id']}<br>
            <b>Time:</b> {meas['frame_timestamp']:.1f}s<br>
            <b>Position:</b> ({meas['x']:.0f}, {meas['y']:.0f}, {meas['z']:.0f})m<br>
            <b>Velocity:</b> ({meas['vx']:.1f}, {meas['vy']:.1f})m/s<br>
            """
            
            # Add beacon info if available
            if 'callsign' in meas:
                popup_text += f"<b>Callsign:</b> {meas['callsign']}<br>"
            if 'identity_code' in meas:
                popup_text += f"<b>Code:</b> {meas['identity_code']}<br>"
            if 'amplitude' in meas:
                popup_text += f"<b>Amplitude:</b> {meas['amplitude']:.1f}<br>"
            
            # Use different icon for first/last points
            if i == 0:
                icon = folium.Icon(color=color, icon='play', prefix='fa')
            elif i == len(measurements) - 1:
                icon = folium.Icon(color=color, icon='stop', prefix='fa')
            else:
                icon = folium.Icon(color=color, icon='circle', prefix='fa')
            
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_text, max_width=300),
                icon=icon,
                tooltip=f'Track {track_id} @ {meas["frame_timestamp"]:.1f}s'
            ).add_to(track_group)
        
        track_group.add_to(m)
    
    # Add clutter as a separate layer
    if clutter:
        clutter_group = folium.FeatureGroup(name='Clutter (False Alarms)', show=False)
        
        for c in clutter[:100]:  # Limit clutter points for performance
            lat, lon = convert_to_latlon(c['x'], c['y'])
            
            popup_text = f"""
            <b>Clutter</b><br>
            <b>Sensor:</b> {c['sensor_id']}<br>
            <b>Time:</b> {c['frame_timestamp']:.1f}s<br>
            <b>Position:</b> ({c['x']:.0f}, {c['y']:.0f}, {c['z']:.0f})m<br>
            """
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                popup=folium.Popup(popup_text, max_width=300),
                color='black',
                fill=True,
                fillColor='black',
                fillOpacity=0.3,
                tooltip=f'Clutter @ {c["frame_timestamp"]:.1f}s'
            ).add_to(clutter_group)
        
        clutter_group.add_to(m)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add fullscreen button
    plugins.Fullscreen().add_to(m)
    
    # Save map
    m.save(output_html)
    print(f"Map saved to {output_html}")
    print(f"\nVisualization includes:")
    print(f"  - {len(tracks)} tracks (each with a unique color)")
    print(f"  - Track paths shown as colored lines")
    print(f"  - Start points marked with play icon")
    print(f"  - End points marked with stop icon")
    print(f"  - Click on any marker for detailed information")
    print(f"  - Use layer control to show/hide individual tracks")
    
    return output_html

if __name__ == '__main__':
    # Create visualization
    output_file = create_track_map('data/sim_realistic_003.jsonl')
    print(f"\nOpen {output_file} in your web browser to view the interactive map!")
