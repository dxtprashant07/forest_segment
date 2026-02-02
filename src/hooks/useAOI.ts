import { useState, useCallback } from "react";
import type { AOIData, LatLng } from "@/types/aoi";
import { calculateAreaKm2, calculateBoundingBox } from "@/utils/geoUtils";

export function useAOI() {
  const [aoiData, setAOIData] = useState<AOIData | null>(null);
  const [isConfirmed, setIsConfirmed] = useState(false);

  const updateAOI = useCallback((coordinates: LatLng[], type: "Polygon" | "Rectangle") => {
    if (coordinates.length < 3) {
      setAOIData(null);
      setIsConfirmed(false);
      return;
    }

    const boundingBox = calculateBoundingBox(coordinates);
    const areaKm2 = calculateAreaKm2(coordinates);

    setAOIData({
      type,
      coordinates,
      boundingBox,
      areaKm2,
    });
    setIsConfirmed(false);
  }, []);

  const clearAOI = useCallback(() => {
    setAOIData(null);
    setIsConfirmed(false);
  }, []);

  const confirmAOI = useCallback(() => {
    setIsConfirmed(true);
  }, []);

  return {
    aoiData,
    isConfirmed,
    updateAOI,
    clearAOI,
    confirmAOI,
  };
}
