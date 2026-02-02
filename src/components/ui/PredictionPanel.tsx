import { Satellite, Brain, AlertCircle, CheckCircle2, Loader2, TreeDeciduous, Eye, EyeOff, Leaf, Cpu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import type { ProcessingStep, PredictionMode } from "@/hooks/useSatelliteProcessing";

export type ModelType = 'segmentation';

interface PredictionPanelProps {
  hasAOI: boolean;
  isConfirmed: boolean;
  step: ProcessingStep;
  error: string | null;
  satelliteImage: string | null;
  predictionResult: {
    mask?: string;
    forest_percentage?: number;
    loss_areas?: Array<{
      coordinates: number[][];
      area_km2: number;
    }>;
    classes?: string[];
    confidence?: number;
    demo_mode?: boolean;
  } | null;
  onFetchImagery: () => void;
  onRunPrediction: (modelType: ModelType) => void;
  onReset: () => void;
  maskOpacity: number;
  onMaskOpacityChange: (opacity: number) => void;
  showMask: boolean;
  onShowMaskChange: (show: boolean) => void;
  predictionMode: PredictionMode;
  onPredictionModeChange: (mode: PredictionMode) => void;
  isMLAvailable: boolean;
  threshold: number;
  onThresholdChange: (threshold: number) => void;
}

export function PredictionPanel({
  hasAOI,
  isConfirmed,
  step,
  error,
  satelliteImage,
  predictionResult,
  onFetchImagery,
  onRunPrediction,
  onReset,
  maskOpacity,
  onMaskOpacityChange,
  showMask,
  onShowMaskChange,
  predictionMode,
  onPredictionModeChange,
  isMLAvailable,
  threshold,
  onThresholdChange,
}: PredictionPanelProps) {
  const modelType: ModelType = 'segmentation';

  const getProgress = () => {
    switch (step) {
      case 'fetching_imagery': return 33;
      case 'running_prediction': return 66;
      case 'complete': return 100;
      default: return 0;
    }
  };

  const getStatusText = () => {
    switch (step) {
      case 'fetching_imagery': return 'Fetching satellite imagery...';
      case 'running_prediction': return 'Running ML prediction...';
      case 'complete': return 'Analysis complete!';
      case 'error': return 'Error occurred';
      default: return 'Ready to analyze';
    }
  };

  if (!hasAOI) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2">
            <TreeDeciduous className="h-5 w-5" />
            Forest Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-4 text-muted-foreground">
            <p className="text-sm">Draw an area on the map to start forest analysis.</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          <TreeDeciduous className="h-5 w-5" />
          Forest Analysis
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Status */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Status:</span>
            <span className="font-medium flex items-center gap-1">
              {step === 'fetching_imagery' || step === 'running_prediction' ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : step === 'complete' ? (
                <CheckCircle2 className="h-3 w-3 text-green-500" />
              ) : step === 'error' ? (
                <AlertCircle className="h-3 w-3 text-destructive" />
              ) : null}
              {getStatusText()}
            </span>
          </div>
          {(step === 'fetching_imagery' || step === 'running_prediction' || step === 'complete') && (
            <Progress value={getProgress()} className="h-2" />
          )}
        </div>

        {/* Error/Info Alert */}
        {error && (
          <Alert variant={error.includes('not configured') ? 'default' : 'destructive'}>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="text-xs">
              {error.includes('not configured') 
                ? 'Sentinel Hub API not configured. Add SENTINEL_CLIENT_ID and SENTINEL_CLIENT_SECRET secrets to enable satellite imagery.'
                : error}
            </AlertDescription>
          </Alert>
        )}

        {/* Analysis Mode Toggle */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Analysis Mode</label>
          <div className="flex items-center justify-between p-2 bg-muted rounded-lg">
            <div className="flex items-center gap-2">
              <Leaf className={`h-4 w-4 ${predictionMode === 'ndvi' ? 'text-green-600' : 'text-muted-foreground'}`} />
              <span className={`text-sm ${predictionMode === 'ndvi' ? 'font-medium' : 'text-muted-foreground'}`}>NDVI</span>
            </div>
            <Switch
              checked={predictionMode === 'ml'}
              onCheckedChange={(checked) => onPredictionModeChange(checked ? 'ml' : 'ndvi')}
            />
            <div className="flex items-center gap-2">
              <span className={`text-sm ${predictionMode === 'ml' ? 'font-medium' : 'text-muted-foreground'}`}>ML</span>
              <Cpu className={`h-4 w-4 ${predictionMode === 'ml' ? 'text-primary' : 'text-muted-foreground'}`} />
            </div>
          </div>
          <p className="text-xs text-muted-foreground">
            {predictionMode === 'ndvi' 
              ? 'NDVI-based vegetation index calculation (fast, always available)'
              : 'External ML model prediction (falls back to NDVI if unavailable)'}
          </p>
        </div>

        {/* Model Info */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Analysis Type</label>
          <div className="flex items-center gap-2 p-2 bg-muted rounded-lg">
            <TreeDeciduous className="h-4 w-4 text-green-600" />
            <span className="text-sm font-medium">Forest Segmentation</span>
          </div>
        </div>

        {/* Satellite Image Preview */}
        {satelliteImage && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">Satellite Image</label>
              <span className="text-xs text-muted-foreground bg-muted px-2 py-0.5 rounded">
                512 × 512 px
              </span>
            </div>
            <div className="border rounded overflow-hidden">
              <img 
                src={satelliteImage} 
                alt="Satellite imagery" 
                className="w-full h-32 object-cover"
              />
            </div>
            <p className="text-xs text-muted-foreground">
              Resolution: ~30m/pixel (Landsat/Sentinel-2 resampled)
            </p>
          </div>
        )}

        {predictionResult && (
          <div className="space-y-3 p-3 bg-muted rounded-lg">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium">Results</h4>
              <Badge variant={predictionResult.demo_mode ? 'secondary' : 'default'}>
                {predictionResult.demo_mode ? (
                  <><Leaf className="h-3 w-3 mr-1" /> NDVI Calculation</>
                ) : (
                  <><Cpu className="h-3 w-3 mr-1" /> ML Prediction</>
                )}
              </Badge>
            </div>
            {predictionResult.forest_percentage !== undefined && (
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Forest Coverage:</span>
                <span className="font-medium text-green-600">
                  {predictionResult.forest_percentage.toFixed(1)}%
                </span>
              </div>
            )}
            {predictionResult.confidence !== undefined && (
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Confidence:</span>
                <span className="font-medium">
                  {(predictionResult.confidence * 100).toFixed(1)}%
                </span>
              </div>
            )}
            {predictionResult.classes && predictionResult.classes.length > 0 && (
              <div className="text-sm">
                <span className="text-muted-foreground">Classes:</span>
                <div className="flex flex-wrap gap-1 mt-1">
                  {predictionResult.classes.map((cls, i) => (
                    <span key={i} className="text-xs bg-primary/10 text-primary px-2 py-0.5 rounded">
                      {cls}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {predictionResult.loss_areas && predictionResult.loss_areas.length > 0 && (
              <div className="text-sm">
                <span className="text-muted-foreground">Loss Areas Detected:</span>
                <span className="font-medium text-destructive ml-2">
                  {predictionResult.loss_areas.length}
                </span>
              </div>
            )}
            {/* Mask preview (if available) */}
            {predictionResult.mask && (
              <div className="border rounded overflow-hidden">
                <img 
                  src={predictionResult.mask} 
                  alt="Prediction mask" 
                  className="w-full h-32 object-cover"
                  loading="lazy"
                />
              </div>
            )}

            {/* Mask Overlay Controls (always shown once we have results) */}
            <div className="space-y-3 pt-2 border-t border-border/50">
              <div className="flex items-center justify-between">
                <Label htmlFor="show-mask" className="text-sm">Show on Map</Label>
                <div className="flex items-center gap-2">
                  <Switch
                    id="show-mask"
                    checked={showMask}
                    onCheckedChange={onShowMaskChange}
                    disabled={!predictionResult.mask}
                  />
                  {showMask ? (
                    <Eye className="h-4 w-4 text-primary" />
                  ) : (
                    <EyeOff className="h-4 w-4 text-muted-foreground" />
                  )}
                </div>
              </div>

              {!predictionResult.mask && (
                <p className="text-xs text-muted-foreground">
                  No overlay was returned for this run.
                </p>
              )}
              {showMask && predictionResult.mask && (
                <div className="space-y-3">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label className="text-sm">Opacity</Label>
                      <span className="text-xs text-muted-foreground">
                        {Math.round(maskOpacity * 100)}%
                      </span>
                    </div>
                    <Slider
                      value={[maskOpacity]}
                      onValueChange={(v) => onMaskOpacityChange(v[0])}
                      min={0.1}
                      max={1}
                      step={0.1}
                      className="w-full"
                    />
                  </div>

                  {/* Threshold Slider */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label className="text-sm">Threshold</Label>
                      <span className="text-xs text-muted-foreground">
                        {threshold.toFixed(2)}
                      </span>
                    </div>
                    <Slider
                      value={[threshold]}
                      onValueChange={(v) => onThresholdChange(v[0])}
                      min={0}
                      max={1}
                      step={0.01}
                      className="w-full"
                    />
                    <p className="text-xs text-muted-foreground">
                      Probability ≥ {threshold.toFixed(2)} → Forest (green)
                    </p>
                  </div>

                  {/* Color Legend - Binary Mask */}
                  <div className="space-y-1.5">
                    <Label className="text-sm">Legend</Label>
                    <div className="grid grid-cols-2 gap-1.5 text-xs">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-sm border border-border" style={{ backgroundColor: '#22c55e' }} />
                        <span className="text-muted-foreground">Forest</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-sm border border-border" style={{ backgroundColor: '#3b82f6' }} />
                        <span className="text-muted-foreground">Non-Forest</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="space-y-2">
          {!satelliteImage && (
            <Button 
              onClick={onFetchImagery} 
              className="w-full"
              disabled={!isConfirmed || step === 'fetching_imagery'}
            >
              <Satellite className="h-4 w-4 mr-2" />
              {step === 'fetching_imagery' ? 'Fetching...' : 'Fetch Satellite Imagery'}
            </Button>
          )}

          {satelliteImage && step !== 'complete' && (
            <Button 
              onClick={() => onRunPrediction(modelType)} 
              className="w-full"
              disabled={step === 'running_prediction'}
            >
              <Brain className="h-4 w-4 mr-2" />
              {step === 'running_prediction' ? 'Running...' : `Run ${modelType === 'segmentation' ? 'Segmentation' : 'Loss Detection'}`}
            </Button>
          )}

          {(step === 'complete' || step === 'error' || satelliteImage) && (
            <Button variant="outline" onClick={onReset} className="w-full">
              Reset Analysis
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
