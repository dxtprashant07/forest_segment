// Types for Area of Interest (AOI) selection

export interface LatLng {
  lat: number;
  lng: number;
}

export interface BoundingBox {
  minLat: number;
  maxLat: number;
  minLng: number;
  maxLng: number;
}

export interface AOIData {
  type: "Polygon" | "Rectangle";
  coordinates: LatLng[];
  boundingBox: BoundingBox;
  areaKm2: number;
}

export interface GeoJSONPolygon {
  type: "Feature";
  geometry: {
    type: "Polygon";
    coordinates: number[][][];
  };
  properties: {
    areaKm2: number;
    createdAt: string;
  };
}

export interface SearchResult {
  display_name: string;
  lat: string;
  lon: string;
  boundingbox: string[];
}
