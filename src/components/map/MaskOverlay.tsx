import { useEffect, useRef } from "react";
import { useMap } from "react-leaflet";
import L from "leaflet";
import type { BoundingBox } from "@/types/aoi";

interface MaskOverlayProps {
  maskImage: string | null;
  bbox: BoundingBox | null;
  opacity?: number;
  visible?: boolean;
}

export function MaskOverlay({ maskImage, bbox, opacity = 0.7, visible = true }: MaskOverlayProps) {
  const map = useMap();
  const overlayRef = useRef<L.ImageOverlay | null>(null);

  useEffect(() => {
    // Remove existing overlay
    if (overlayRef.current) {
      map.removeLayer(overlayRef.current);
      overlayRef.current = null;
    }

    // Don't add if not visible or no data
    if (!visible || !maskImage || !bbox) return;

    // Create bounds from bounding box
    const bounds = L.latLngBounds(
      [bbox.minLat, bbox.minLng], // Southwest corner
      [bbox.maxLat, bbox.maxLng]  // Northeast corner
    );

    // Create image overlay
    const overlay = L.imageOverlay(maskImage, bounds, {
      opacity: opacity,
      interactive: false,
      className: 'prediction-mask',
    });

    overlay.addTo(map);
    overlayRef.current = overlay;

    return () => {
      if (overlayRef.current) {
        map.removeLayer(overlayRef.current);
        overlayRef.current = null;
      }
    };
  }, [map, maskImage, bbox, opacity, visible]);

  // Update opacity when changed
  useEffect(() => {
    if (overlayRef.current) {
      overlayRef.current.setOpacity(opacity);
    }
  }, [opacity]);

  return null;
}
