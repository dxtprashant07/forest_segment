import { useState, useCallback } from "react";
import { Map } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { AOIMap } from "@/components/map/AOIMap";
import { CoordinatePanel } from "@/components/ui/CoordinatePanel";
import { PredictionPanel } from "@/components/ui/PredictionPanel";
import { ModelType } from "@/hooks/useSatelliteProcessing";
import { LocationSearch } from "@/components/ui/LocationSearch";
import { BandSelector } from "@/components/ui/BandSelector";
import { ImageVisualization } from "@/components/ui/ImageVisualization";
import { useAOI } from "@/hooks/useAOI";
import { useSatelliteProcessing } from "@/hooks/useSatelliteProcessing";
import { saveArea } from "@/services/areaService";
import type { LatLng } from "@/types/aoi";

const Index = () => {
  const { toast } = useToast();
  const { aoiData, isConfirmed, updateAOI, clearAOI, confirmAOI } = useAOI();
  const [flyTo, setFlyTo] = useState<{ lat: number; lng: number; zoom: number } | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [selectedBands, setSelectedBands] = useState<string[]>(["B02", "B03", "B04", "B08", "NDVI"]);
  const [maskOpacity, setMaskOpacity] = useState(0.7);
  const [showMask, setShowMask] = useState(true);
  const [threshold, setThreshold] = useState(0.5);

  const {
    step,
    error,
    satelliteImage,
    predictionResult,
    isProcessing,
    fetchImagery,
    predict,
    reset: resetProcessing,
    predictionMode,
    setPredictionMode,
    isMLAvailable,
  } = useSatelliteProcessing();

  const handleShapeCreated = useCallback(
    (coordinates: LatLng[], type: "Polygon" | "Rectangle") => {
      updateAOI(coordinates, type);
      resetProcessing();
      toast({
        title: "Area Selected",
        description: `${type} with ${coordinates.length} points created.`,
      });
    },
    [updateAOI, resetProcessing, toast]
  );

  const handleShapeDeleted = useCallback(() => {
    clearAOI();
    resetProcessing();
    toast({
      title: "Area Cleared",
      description: "Selection has been removed.",
    });
  }, [clearAOI, resetProcessing, toast]);

  const handleLocationSelect = useCallback((lat: number, lng: number, zoom: number) => {
    setFlyTo({ lat, lng, zoom });
  }, []);

  const handleConfirm = useCallback(() => {
    confirmAOI();
    toast({
      title: "Selection Confirmed",
      description: "You can now fetch satellite imagery or export as GeoJSON.",
    });
  }, [confirmAOI, toast]);

  const handleClear = useCallback(() => {
    clearAOI();
    resetProcessing();
  }, [clearAOI, resetProcessing]);

  const handleSaveToBackend = useCallback(async () => {
    if (!aoiData) return;

    setIsSaving(true);
    try {
      const payload = {
        type: "Polygon" as const,
        coordinates: aoiData.coordinates.map((c) => [c.lat, c.lng] as [number, number]),
        boundingBox: aoiData.boundingBox,
      };
      await saveArea(payload);
      toast({
        title: "Saved Successfully",
        description: "Area has been saved to the database.",
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error occurred";
      toast({
        title: "Save Failed",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setIsSaving(false);
    }
  }, [aoiData, toast]);

  const handleFetchImagery = useCallback(async () => {
    if (!aoiData) return;

    toast({
      title: "Fetching Imagery",
      description: `Requesting satellite data (${selectedBands.length} bands) at 30m resolution...`,
    });

    const image = await fetchImagery(aoiData.boundingBox);
    
    if (image) {
      toast({
        title: "Imagery Retrieved",
        description: `Satellite image with bands [${selectedBands.join(", ")}] fetched successfully.`,
      });
    }
  }, [aoiData, fetchImagery, toast, selectedBands]);

  const handleRunPrediction = useCallback(async (modelType: ModelType) => {
    if (!aoiData) return;

    const modeLabel = predictionMode === 'ndvi' ? 'NDVI calculation' : 'ML prediction';

    toast({
      title: "Running Analysis",
      description: `Running ${modeLabel} for forest segmentation (threshold: ${threshold})...`,
    });

    // Pass threshold to predict
    const result = await predict(
      aoiData.boundingBox,
      modelType,
      predictionMode === 'ndvi',
      threshold
    );

    if (result) {
      const resultType = result.demo_mode ? 'NDVI calculation' : 'ML prediction';
      toast({
        title: "Analysis Complete",
        description: `${resultType} results are ready. Toggle 'Show on Map' to view the overlay.`,
      });
    }
  }, [aoiData, predict, toast, predictionMode, threshold]);

  const handleResetProcessing = useCallback(() => {
    resetProcessing();
    setShowMask(true);
    toast({
      title: "Analysis Reset",
      description: "Ready for new analysis.",
    });
  }, [resetProcessing, toast]);

  return (
    <div className="h-screen w-screen flex flex-col bg-background">
      {/* Header */}
      <header className="h-16 border-b border-border px-4 flex items-center justify-between shrink-0 bg-card">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-lg bg-primary flex items-center justify-center">
            <Map className="h-5 w-5 text-primary-foreground" />
          </div>
          <div>
            <h1 className="font-semibold text-foreground">Forest Analysis Tool</h1>
            <p className="text-xs text-muted-foreground">Satellite imagery & ML predictions</p>
          </div>
        </div>
        <div className="w-80">
          <LocationSearch onLocationSelect={handleLocationSelect} />
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex min-h-0">
        {/* Map Container */}
        <div className="flex-1 relative">
          <AOIMap
            onShapeCreated={handleShapeCreated}
            onShapeDeleted={handleShapeDeleted}
            flyTo={flyTo}
            maskImage={predictionResult?.mask}
            maskBbox={aoiData?.boundingBox}
            maskOpacity={maskOpacity}
            showMask={showMask}
          />
        </div>

        {/* Side Panel */}
        <aside className="w-96 border-l border-border p-4 bg-card overflow-y-auto space-y-4">
          <CoordinatePanel
            aoiData={aoiData}
            isConfirmed={isConfirmed}
            onConfirm={handleConfirm}
            onClear={handleClear}
            onSaveToBackend={handleSaveToBackend}
            isSaving={isSaving}
          />
          
          <BandSelector
            selectedBands={selectedBands}
            onBandsChange={setSelectedBands}
          />
          
          <PredictionPanel
            hasAOI={!!aoiData}
            isConfirmed={isConfirmed}
            step={step}
            error={error}
            satelliteImage={satelliteImage}
            predictionResult={predictionResult}
            onFetchImagery={handleFetchImagery}
            onRunPrediction={handleRunPrediction}
            onReset={handleResetProcessing}
            maskOpacity={maskOpacity}
            onMaskOpacityChange={setMaskOpacity}
            showMask={showMask}
            onShowMaskChange={setShowMask}
            predictionMode={predictionMode}
            onPredictionModeChange={setPredictionMode}
            isMLAvailable={isMLAvailable}
            threshold={threshold}
            onThresholdChange={setThreshold}
          />
          
          <ImageVisualization
            satelliteImage={satelliteImage}
            selectedBands={selectedBands}
            bbox={aoiData?.boundingBox}
          />
        </aside>
      </main>
    </div>
  );
};

export default Index;
