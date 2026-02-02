import { MapPin, Square, Ruler, Download, Check, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import type { AOIData } from "@/types/aoi";
import { formatCoordinate, downloadGeoJSON } from "@/utils/geoUtils";

interface CoordinatePanelProps {
  aoiData: AOIData | null;
  isConfirmed: boolean;
  onConfirm: () => void;
  onClear: () => void;
  onSaveToBackend: () => void;
  isSaving: boolean;
}

export function CoordinatePanel({
  aoiData,
  isConfirmed,
  onConfirm,
  onClear,
  onSaveToBackend,
  isSaving,
}: CoordinatePanelProps) {
  if (!aoiData) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <MapPin className="h-5 w-5" />
            Area Selection
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            <Square className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p className="text-sm">Draw a rectangle or polygon on the map to select an area of interest.</p>
            <p className="text-xs mt-2">Use the drawing tools in the top-left corner of the map.</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          <MapPin className="h-5 w-5" />
          Selected Area
        </CardTitle>
      </CardHeader>
      
      <CardContent className="flex-1 flex flex-col gap-4">
        {/* Area Info */}
        <div className="space-y-3">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Type:</span>
            <span className="font-medium">{aoiData.type}</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground flex items-center gap-1">
              <Ruler className="h-3 w-3" /> Area:
            </span>
            <span className="font-medium">{aoiData.areaKm2.toLocaleString()} km²</span>
          </div>
        </div>

        <Separator />

        {/* Bounding Box */}
        <div>
          <h4 className="text-sm font-medium mb-2">Bounding Box</h4>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-muted rounded p-2">
              <span className="text-muted-foreground block">Min Lat</span>
              <span className="font-mono">{formatCoordinate(aoiData.boundingBox.minLat)}</span>
            </div>
            <div className="bg-muted rounded p-2">
              <span className="text-muted-foreground block">Max Lat</span>
              <span className="font-mono">{formatCoordinate(aoiData.boundingBox.maxLat)}</span>
            </div>
            <div className="bg-muted rounded p-2">
              <span className="text-muted-foreground block">Min Lng</span>
              <span className="font-mono">{formatCoordinate(aoiData.boundingBox.minLng)}</span>
            </div>
            <div className="bg-muted rounded p-2">
              <span className="text-muted-foreground block">Max Lng</span>
              <span className="font-mono">{formatCoordinate(aoiData.boundingBox.maxLng)}</span>
            </div>
          </div>
        </div>

        <Separator />

        {/* Coordinates */}
        <div className="flex-1 min-h-0">
          <h4 className="text-sm font-medium mb-2">Polygon Coordinates ({aoiData.coordinates.length} points)</h4>
          <ScrollArea className="h-32 border border-border rounded">
            <div className="p-2 space-y-1">
              {aoiData.coordinates.map((coord, index) => (
                <div key={index} className="text-xs font-mono bg-muted/50 rounded px-2 py-1 flex justify-between">
                  <span className="text-muted-foreground">#{index + 1}</span>
                  <span>{formatCoordinate(coord.lat)}, {formatCoordinate(coord.lng)}</span>
                </div>
              ))}
            </div>
          </ScrollArea>
        </div>

        <Separator />

        {/* Actions */}
        <div className="space-y-2">
          {!isConfirmed ? (
            <Button onClick={onConfirm} className="w-full" size="lg">
              <Check className="h-4 w-4 mr-2" />
              Confirm Selection
            </Button>
          ) : (
            <Button onClick={onSaveToBackend} className="w-full" size="lg" disabled={isSaving}>
              {isSaving ? "Saving..." : "Save to Backend"}
            </Button>
          )}
          
          <div className="flex gap-2">
            <Button
              variant="outline"
              onClick={() => downloadGeoJSON(aoiData)}
              className="flex-1"
            >
              <Download className="h-4 w-4 mr-2" />
              Export GeoJSON
            </Button>
            <Button variant="destructive" onClick={onClear} size="icon">
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
