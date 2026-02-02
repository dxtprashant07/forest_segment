import { useEffect, useRef } from "react";
import { useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet-draw";
import "leaflet-draw/dist/leaflet.draw.css";
import type { LatLng } from "@/types/aoi";

// Fix leaflet-draw icon paths for Vite bundler
// @ts-ignore
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "/images/spritesheet-2x.png",
  iconUrl: "/images/spritesheet.png",
  shadowUrl: "",
});

interface DrawControlProps {
  onShapeCreated: (coordinates: LatLng[], type: "Polygon" | "Rectangle") => void;
  onShapeDeleted: () => void;
}

export function DrawControl({ onShapeCreated, onShapeDeleted }: DrawControlProps) {
  const map = useMap();
  const drawnItemsRef = useRef<L.FeatureGroup | null>(null);
  const drawControlRef = useRef<L.Control.Draw | null>(null);
  const isInitializedRef = useRef(false);

  useEffect(() => {
    if (!map || isInitializedRef.current) return;
    isInitializedRef.current = true;

    // Create feature group for drawn items
    const drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);
    drawnItemsRef.current = drawnItems;

    // Configure draw control with proper polygon settings
    const drawControl = new L.Control.Draw({
      position: "topleft",
      draw: {
        polyline: false,
        circle: false,
        circlemarker: false,
        marker: false,
        polygon: {
          allowIntersection: false,
          // leaflet-draw's area tooltip can throw "type is not defined" in some builds.
          // Disabling it avoids the runtime crash while drawing.
          showArea: false,
          drawError: {
            color: "#e74c3c",
            message: "<strong>Cannot intersect!</strong>",
          },
          shapeOptions: {
            color: "#3b82f6",
            fillColor: "#3b82f6",
            fillOpacity: 0.2,
            weight: 2,
          },
        },
        rectangle: {
          // Same issue as polygon area tooltip.
          showArea: false,
          shapeOptions: {
            color: "#3b82f6",
            fillColor: "#3b82f6",
            fillOpacity: 0.2,
            weight: 2,
          },
        },
      },
      edit: {
        featureGroup: drawnItems,
        remove: true,
      },
    });

    map.addControl(drawControl);
    drawControlRef.current = drawControl;

    // Handle shape creation
    const handleCreated = (e: L.LeafletEvent) => {
      const event = e as L.DrawEvents.Created;

      // Clear previous shapes - only allow one at a time
      drawnItems.clearLayers();
      drawnItems.addLayer(event.layer);

      // Extract coordinates
      const layer = event.layer;
      let coordinates: LatLng[] = [];
      let shapeType: "Polygon" | "Rectangle" = "Polygon";

      if (event.layerType === "rectangle") {
        const bounds = (layer as L.Rectangle).getBounds();
        coordinates = [
          { lat: bounds.getNorthWest().lat, lng: bounds.getNorthWest().lng },
          { lat: bounds.getNorthEast().lat, lng: bounds.getNorthEast().lng },
          { lat: bounds.getSouthEast().lat, lng: bounds.getSouthEast().lng },
          { lat: bounds.getSouthWest().lat, lng: bounds.getSouthWest().lng },
        ];
        // Close the ring (GeoJSON-style)
        coordinates.push(coordinates[0]);
        shapeType = "Rectangle";
      } else if (layer instanceof L.Polygon) {
        const latLngs = layer.getLatLngs()[0] as L.LatLng[];
        coordinates = latLngs.map((ll) => ({ lat: ll.lat, lng: ll.lng }));
        if (coordinates.length > 0) coordinates.push(coordinates[0]);
        shapeType = "Polygon";
      }

      onShapeCreated(coordinates, shapeType);
    };

    // Handle shape editing
    const handleEdited = (e: L.LeafletEvent) => {
      const event = e as L.DrawEvents.Edited;
      event.layers.eachLayer((layer) => {
        if (layer instanceof L.Rectangle) {
          const bounds = layer.getBounds();
          const coordinates = [
            { lat: bounds.getNorthWest().lat, lng: bounds.getNorthWest().lng },
            { lat: bounds.getNorthEast().lat, lng: bounds.getNorthEast().lng },
            { lat: bounds.getSouthEast().lat, lng: bounds.getSouthEast().lng },
            { lat: bounds.getSouthWest().lat, lng: bounds.getSouthWest().lng },
          ];
          coordinates.push(coordinates[0]);
          onShapeCreated(coordinates, "Rectangle");
        } else if (layer instanceof L.Polygon) {
          const latLngs = layer.getLatLngs()[0] as L.LatLng[];
          const coordinates = latLngs.map((ll) => ({ lat: ll.lat, lng: ll.lng }));
          if (coordinates.length > 0) coordinates.push(coordinates[0]);
          onShapeCreated(coordinates, "Polygon");
        }
      });
    };

    // Handle shape deletion
    const handleDeleted = () => {
      onShapeDeleted();
    };

    map.on(L.Draw.Event.CREATED, handleCreated);
    map.on(L.Draw.Event.EDITED, handleEdited);
    map.on(L.Draw.Event.DELETED, handleDeleted);

    return () => {
      isInitializedRef.current = false;
      map.off(L.Draw.Event.CREATED, handleCreated);
      map.off(L.Draw.Event.EDITED, handleEdited);
      map.off(L.Draw.Event.DELETED, handleDeleted);

      if (drawControlRef.current) {
        map.removeControl(drawControlRef.current);
      }
      if (drawnItemsRef.current) {
        map.removeLayer(drawnItemsRef.current);
      }
    };
  }, [map, onShapeCreated, onShapeDeleted]);

  return null;
}
