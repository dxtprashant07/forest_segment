import type { LatLng, BoundingBox, GeoJSONPolygon, AOIData } from "@/types/aoi";

/**
 * Calculate the area of a polygon in square kilometers
 * Uses the Shoelace formula with spherical Earth approximation
 */
export function calculateAreaKm2(coordinates: LatLng[]): number {
  if (coordinates.length < 3) return 0;

  const earthRadiusKm = 6371;

  // Convert to radians
  const toRad = (deg: number) => (deg * Math.PI) / 180;

  // Calculate using spherical excess formula (simplified)
  let area = 0;
  const n = coordinates.length;

  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const lat1 = toRad(coordinates[i].lat);
    const lat2 = toRad(coordinates[j].lat);
    const lng1 = toRad(coordinates[i].lng);
    const lng2 = toRad(coordinates[j].lng);

    area += (lng2 - lng1) * (2 + Math.sin(lat1) + Math.sin(lat2));
  }

  area = Math.abs((area * earthRadiusKm * earthRadiusKm) / 2);

  return Math.round(area * 100) / 100;
}

/**
 * Calculate bounding box from coordinates
 */
export function calculateBoundingBox(coordinates: LatLng[]): BoundingBox {
  if (coordinates.length === 0) {
    return { minLat: 0, maxLat: 0, minLng: 0, maxLng: 0 };
  }

  const lats = coordinates.map((c) => c.lat);
  const lngs = coordinates.map((c) => c.lng);

  return {
    minLat: Math.min(...lats),
    maxLat: Math.max(...lats),
    minLng: Math.min(...lngs),
    maxLng: Math.max(...lngs),
  };
}

/**
 * Convert AOI data to GeoJSON format
 */
export function toGeoJSON(aoiData: AOIData): GeoJSONPolygon {
  // GeoJSON uses [lng, lat] order and closes the polygon
  const coords = aoiData.coordinates.map((c) => [c.lng, c.lat]);
  // Close the polygon by adding the first point at the end
  if (coords.length > 0) {
    coords.push([...coords[0]]);
  }

  return {
    type: "Feature",
    geometry: {
      type: "Polygon",
      coordinates: [coords],
    },
    properties: {
      areaKm2: aoiData.areaKm2,
      createdAt: new Date().toISOString(),
    },
  };
}

/**
 * Download GeoJSON as a file
 */
export function downloadGeoJSON(aoiData: AOIData): void {
  const geojson = toGeoJSON(aoiData);
  const blob = new Blob([JSON.stringify(geojson, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `aoi-${Date.now()}.geojson`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Format coordinates for display
 */
export function formatCoordinate(value: number, decimals: number = 6): string {
  return value.toFixed(decimals);
}
