import { useEffect } from "react";
import { MapContainer, TileLayer, useMap, LayersControl } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { DrawControl } from "./DrawControl";
import { MaskOverlay } from "./MaskOverlay";
import type { LatLng, BoundingBox } from "@/types/aoi";

// Fix for default marker icons in Leaflet with Vite
import icon from "leaflet/dist/images/marker-icon.png";
import iconShadow from "leaflet/dist/images/marker-shadow.png";

const DefaultIcon = L.icon({
  iconUrl: icon,
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
});
L.Marker.prototype.options.icon = DefaultIcon;

interface MapControllerProps {
  flyTo: { lat: number; lng: number; zoom: number } | null;
}

function MapController({ flyTo }: MapControllerProps) {
  const map = useMap();

  useEffect(() => {
    if (flyTo) {
      map.flyTo([flyTo.lat, flyTo.lng], flyTo.zoom, {
        duration: 1.5,
      });
    }
  }, [map, flyTo]);

  return null;
}

interface AOIMapProps {
  onShapeCreated: (coordinates: LatLng[], type: "Polygon" | "Rectangle") => void;
  onShapeDeleted: () => void;
  flyTo: { lat: number; lng: number; zoom: number } | null;
  maskImage?: string | null;
  maskBbox?: BoundingBox | null;
  maskOpacity?: number;
  showMask?: boolean;
}

export function AOIMap({ 
  onShapeCreated, 
  onShapeDeleted, 
  flyTo,
  maskImage,
  maskBbox,
  maskOpacity = 0.7,
  showMask = true,
}: AOIMapProps) {
  return (
    <MapContainer
      center={[20, 0]}
      zoom={2}
      className="h-full w-full"
      style={{ background: "hsl(var(--muted))" }}
    >
      <LayersControl position="topright">
        <LayersControl.BaseLayer checked name="OpenStreetMap">
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
        </LayersControl.BaseLayer>
        <LayersControl.BaseLayer name="Satellite (Esri)">
          <TileLayer
            attribution='&copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP'
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          />
        </LayersControl.BaseLayer>
        <LayersControl.BaseLayer name="Satellite (Google)">
          <TileLayer
            attribution='&copy; Google'
            url="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
          />
        </LayersControl.BaseLayer>
        <LayersControl.BaseLayer name="Terrain">
          <TileLayer
            attribution='&copy; OpenTopoMap'
            url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
          />
        </LayersControl.BaseLayer>
      </LayersControl>
      <DrawControl onShapeCreated={onShapeCreated} onShapeDeleted={onShapeDeleted} />
      <MapController flyTo={flyTo} />
      <MaskOverlay 
        maskImage={maskImage || null} 
        bbox={maskBbox || null} 
        opacity={maskOpacity}
        visible={showMask}
      />
    </MapContainer>
  );
}
