import { useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Eye, Palette, Leaf, Droplets, Flame, Mountain, Loader2, RefreshCw } from "lucide-react";
import { fetchSatelliteImagery, type VisualizationType } from "@/services/satelliteService";
import type { BoundingBox } from "@/types/aoi";
import { useToast } from "@/hooks/use-toast";

interface VisualizationConfig {
  id: VisualizationType;
  name: string;
  description: string;
  icon: React.ReactNode;
  bands: string[];
  colorScale: string[];
  legend: { label: string; color: string }[];
}

const VISUALIZATION_CONFIGS: VisualizationConfig[] = [
  {
    id: "true_color",
    name: "True Color",
    description: "Natural RGB (B4, B3, B2)",
    icon: <Eye className="h-4 w-4" />,
    bands: ["B04", "B03", "B02"],
    colorScale: [],
    legend: [],
  },
  {
    id: "ndvi",
    name: "NDVI",
    description: "Vegetation index: (NIR-Red)/(NIR+Red)",
    icon: <Leaf className="h-4 w-4" />,
    bands: ["B04", "B08"],
    colorScale: ["#0d1a66", "#805633", "#bfa673", "#cccc4d", "#8cbf40", "#40a626", "#1a731a"],
    legend: [
      { label: "Water", color: "#0d1a66" },
      { label: "Bare", color: "#805633" },
      { label: "Sparse", color: "#bfa673" },
      { label: "Low Veg", color: "#cccc4d" },
      { label: "Moderate", color: "#8cbf40" },
      { label: "Good Veg", color: "#40a626" },
      { label: "Dense", color: "#1a731a" },
    ],
  },
  {
    id: "ndwi",
    name: "NDWI",
    description: "Water index: (Green-NIR)/(Green+NIR)",
    icon: <Droplets className="h-4 w-4" />,
    bands: ["B03", "B08"],
    colorScale: ["#8c5933", "#b38c59", "#999988", "#80b3d9", "#4d8cd9", "#1a4db3"],
    legend: [
      { label: "Very Dry", color: "#8c5933" },
      { label: "Dry", color: "#b38c59" },
      { label: "Moist", color: "#999988" },
      { label: "Wet", color: "#80b3d9" },
      { label: "Wetland", color: "#4d8cd9" },
      { label: "Water", color: "#1a4db3" },
    ],
  },
  {
    id: "false_color",
    name: "False Color",
    description: "NIR-Red-Green composite (B8, B4, B3)",
    icon: <Palette className="h-4 w-4" />,
    bands: ["B08", "B04", "B03"],
    colorScale: [],
    legend: [],
  },
  {
    id: "swir",
    name: "SWIR",
    description: "Short-wave IR (B12, B8A, B4)",
    icon: <Mountain className="h-4 w-4" />,
    bands: ["B12", "B8A", "B04"],
    colorScale: [],
    legend: [],
  },
  {
    id: "nbr",
    name: "NBR",
    description: "Burn ratio: (NIR-SWIR)/(NIR+SWIR)",
    icon: <Flame className="h-4 w-4" />,
    bands: ["B08", "B12"],
    colorScale: ["#4d0d0d", "#992608", "#cc591a", "#e6b366", "#b3cc66", "#66b340", "#269926"],
    legend: [
      { label: "Severe Burn", color: "#4d0d0d" },
      { label: "Mod Burn", color: "#992608" },
      { label: "Low Burn", color: "#cc591a" },
      { label: "Unburned", color: "#e6b366" },
      { label: "Low Veg", color: "#b3cc66" },
      { label: "Mod Veg", color: "#66b340" },
      { label: "Healthy", color: "#269926" },
    ],
  },
];

interface ImageVisualizationProps {
  satelliteImage: string | null;
  selectedBands: string[];
  bbox?: BoundingBox | null;
  className?: string;
}

interface VisualizationCache {
  [key: string]: string;
}

export function ImageVisualization({ satelliteImage, selectedBands, bbox, className }: ImageVisualizationProps) {
  const { toast } = useToast();
  const [activeVisualization, setActiveVisualization] = useState<VisualizationType>("true_color");
  const [visualizationCache, setVisualizationCache] = useState<VisualizationCache>({});
  const [loadingViz, setLoadingViz] = useState<VisualizationType | null>(null);

  // Store true_color in cache when satelliteImage changes
  if (satelliteImage && !visualizationCache.true_color) {
    setVisualizationCache(prev => ({ ...prev, true_color: satelliteImage }));
  }

  const fetchVisualization = useCallback(async (vizType: VisualizationType) => {
    if (!bbox) {
      toast({
        title: "No Area Selected",
        description: "Please draw an area on the map first.",
        variant: "destructive",
      });
      return;
    }

    // Return cached if available
    if (visualizationCache[vizType]) {
      setActiveVisualization(vizType);
      return;
    }

    setLoadingViz(vizType);

    try {
      const result = await fetchSatelliteImagery(bbox, undefined, undefined, vizType, 30);
      
      if (result.error) {
        throw new Error(result.message || result.error);
      }

      if (result.image) {
        setVisualizationCache(prev => ({ ...prev, [vizType]: result.image! }));
        setActiveVisualization(vizType);
        toast({
          title: `${vizType.toUpperCase()} Calculated`,
          description: "Real index values computed from satellite bands.",
        });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to fetch visualization";
      toast({
        title: "Visualization Failed",
        description: message,
        variant: "destructive",
      });
    } finally {
      setLoadingViz(null);
    }
  }, [bbox, visualizationCache, toast]);

  const handleVisualizationChange = (vizType: string) => {
    const viz = vizType as VisualizationType;
    
    // If cached, just switch
    if (visualizationCache[viz]) {
      setActiveVisualization(viz);
      return;
    }

    // Otherwise fetch it
    fetchVisualization(viz);
  };

  const handleRefresh = (vizType: VisualizationType) => {
    // Clear cache for this viz and refetch
    setVisualizationCache(prev => {
      const { [vizType]: _, ...rest } = prev;
      return rest;
    });
    fetchVisualization(vizType);
  };

  if (!satelliteImage) {
    return null;
  }

  const activeConfig = VISUALIZATION_CONFIGS.find(c => c.id === activeVisualization);
  const currentImage = visualizationCache[activeVisualization] || satelliteImage;

  return (
    <Card className={className}>
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          <Palette className="h-5 w-5" />
          Band Visualization
          <Badge variant="outline" className="ml-auto text-xs">
            Real Calculation
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Visualization Tabs */}
        <Tabs value={activeVisualization} onValueChange={handleVisualizationChange}>
          <TabsList className="grid w-full grid-cols-3 h-auto">
            {VISUALIZATION_CONFIGS.slice(0, 6).map((config) => (
              <TabsTrigger
                key={config.id}
                value={config.id}
                disabled={loadingViz !== null}
                className="text-xs py-2 px-1 flex flex-col gap-1"
              >
                {loadingViz === config.id ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  config.icon
                )}
                <span className="truncate">{config.name}</span>
                {visualizationCache[config.id] && (
                  <span className="w-1.5 h-1.5 rounded-full bg-primary" />
                )}
              </TabsTrigger>
            ))}
          </TabsList>

          {VISUALIZATION_CONFIGS.map((config) => (
            <TabsContent key={config.id} value={config.id} className="mt-3">
              <div className="space-y-3">
                {/* Image Preview */}
                <div className="relative border rounded-lg overflow-hidden bg-muted">
                  <img
                    src={visualizationCache[config.id] || satelliteImage}
                    alt={`${config.name} visualization`}
                    className="w-full h-48 object-cover"
                  />
                  {loadingViz === config.id && (
                    <div className="absolute inset-0 bg-background/80 flex items-center justify-center">
                      <div className="flex flex-col items-center gap-2">
                        <Loader2 className="h-8 w-8 animate-spin text-primary" />
                        <span className="text-sm text-muted-foreground">Calculating {config.name}...</span>
                      </div>
                    </div>
                  )}
                  {visualizationCache[config.id] && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="absolute top-2 right-2 h-7 w-7 bg-background/80"
                      onClick={() => handleRefresh(config.id)}
                    >
                      <RefreshCw className="h-3.5 w-3.5" />
                    </Button>
                  )}
                </div>

                {/* Visualization Info */}
                <div className="space-y-2">
                  <p className="text-xs text-muted-foreground">{config.description}</p>
                  
                  {/* Bands Used */}
                  <div className="flex flex-wrap gap-1">
                    {config.bands.map((band) => (
                      <Badge key={band} variant="secondary" className="text-xs">
                        {band}
                      </Badge>
                    ))}
                  </div>

                  {/* Color Legend for Index Visualizations */}
                  {config.legend.length > 0 && (
                    <div className="space-y-1.5 p-3 bg-muted/30 rounded-lg">
                      <span className="text-xs font-medium">Color Legend:</span>
                      <div className="flex h-4 rounded overflow-hidden">
                        {config.colorScale.map((color, i) => (
                          <div
                            key={i}
                            className="flex-1"
                            style={{ backgroundColor: color }}
                          />
                        ))}
                      </div>
                      <div className="flex justify-between text-[10px] text-muted-foreground">
                        <span>{config.legend[0]?.label}</span>
                        <span>{config.legend[config.legend.length - 1]?.label}</span>
                      </div>
                    </div>
                  )}

                  {/* Fetch Button if not cached */}
                  {!visualizationCache[config.id] && config.id !== 'true_color' && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="w-full"
                      onClick={() => fetchVisualization(config.id)}
                      disabled={loadingViz !== null || !bbox}
                    >
                      {loadingViz === config.id ? (
                        <>
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                          Calculating...
                        </>
                      ) : (
                        <>
                          Calculate {config.name}
                        </>
                      )}
                    </Button>
                  )}
                </div>
              </div>
            </TabsContent>
          ))}
        </Tabs>

        {/* Resolution Info */}
        <div className="pt-2 border-t border-border flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Resolution</span>
          <Badge variant="outline" className="text-xs">
            ~30m/pixel
          </Badge>
        </div>
      </CardContent>
    </Card>
  );
}
