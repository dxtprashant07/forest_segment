import { useState } from "react";
import { Checkbox } from "@/components/ui/checkbox";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Layers, Info } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export interface BandConfig {
  id: string;
  name: string;
  description: string;
  wavelength?: string;
  category: "visible" | "infrared" | "index";
}

const AVAILABLE_BANDS: BandConfig[] = [
  // Visible bands
  { id: "B02", name: "B2 (Blue)", description: "Blue band", wavelength: "490nm", category: "visible" },
  { id: "B03", name: "B3 (Green)", description: "Green band", wavelength: "560nm", category: "visible" },
  { id: "B04", name: "B4 (Red)", description: "Red band", wavelength: "665nm", category: "visible" },
  
  // Red Edge & NIR bands
  { id: "B05", name: "B5 (Red Edge 1)", description: "Vegetation red edge", wavelength: "705nm", category: "infrared" },
  { id: "B06", name: "B6 (Red Edge 2)", description: "Vegetation red edge", wavelength: "740nm", category: "infrared" },
  { id: "B07", name: "B7 (Red Edge 3)", description: "Vegetation red edge", wavelength: "783nm", category: "infrared" },
  { id: "B08", name: "B8 (NIR)", description: "Near Infrared", wavelength: "842nm", category: "infrared" },
  { id: "B8A", name: "B8A (NIR Narrow)", description: "Narrow NIR", wavelength: "865nm", category: "infrared" },
  
  // SWIR bands
  { id: "B11", name: "B11 (SWIR 1)", description: "Short-wave Infrared", wavelength: "1610nm", category: "infrared" },
  { id: "B12", name: "B12 (SWIR 2)", description: "Short-wave Infrared", wavelength: "2190nm", category: "infrared" },
  
  // Spectral indices
  { id: "NDVI", name: "NDVI", description: "Normalized Difference Vegetation Index - Forest health indicator", category: "index" },
  { id: "NDWI", name: "NDWI", description: "Normalized Difference Water Index - Moisture content", category: "index" },
  { id: "EVI", name: "EVI", description: "Enhanced Vegetation Index - Biomass estimation", category: "index" },
  { id: "SAVI", name: "SAVI", description: "Soil Adjusted Vegetation Index - Sparse vegetation", category: "index" },
  { id: "NBR", name: "NBR", description: "Normalized Burn Ratio - Fire damage detection", category: "index" },
];

interface BandSelectorProps {
  selectedBands: string[];
  onBandsChange: (bands: string[]) => void;
}

export function BandSelector({ selectedBands, onBandsChange }: BandSelectorProps) {
  const toggleBand = (bandId: string) => {
    if (selectedBands.includes(bandId)) {
      onBandsChange(selectedBands.filter((b) => b !== bandId));
    } else {
      onBandsChange([...selectedBands, bandId]);
    }
  };

  const visibleBands = AVAILABLE_BANDS.filter((b) => b.category === "visible");
  const infraredBands = AVAILABLE_BANDS.filter((b) => b.category === "infrared");
  const indexBands = AVAILABLE_BANDS.filter((b) => b.category === "index");

  const selectPreset = (preset: "rgb" | "infrared" | "vegetation" | "all") => {
    switch (preset) {
      case "rgb":
        onBandsChange(["B02", "B03", "B04"]);
        break;
      case "infrared":
        onBandsChange(["B08", "B11", "B12"]);
        break;
      case "vegetation":
        onBandsChange(["B04", "B08", "NDVI", "EVI"]);
        break;
      case "all":
        onBandsChange(AVAILABLE_BANDS.map((b) => b.id));
        break;
    }
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          <Layers className="h-5 w-5" />
          Band Selection
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Quick Presets */}
        <div className="flex flex-wrap gap-2">
          <Badge
            variant="outline"
            className="cursor-pointer hover:bg-primary hover:text-primary-foreground transition-colors"
            onClick={() => selectPreset("rgb")}
          >
            True Color (RGB)
          </Badge>
          <Badge
            variant="outline"
            className="cursor-pointer hover:bg-primary hover:text-primary-foreground transition-colors"
            onClick={() => selectPreset("infrared")}
          >
            Infrared
          </Badge>
          <Badge
            variant="outline"
            className="cursor-pointer hover:bg-primary hover:text-primary-foreground transition-colors"
            onClick={() => selectPreset("vegetation")}
          >
            Vegetation
          </Badge>
          <Badge
            variant="outline"
            className="cursor-pointer hover:bg-primary hover:text-primary-foreground transition-colors"
            onClick={() => selectPreset("all")}
          >
            All Bands
          </Badge>
        </div>

        {/* Visible Bands */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-muted-foreground">Visible Bands</h4>
          <div className="grid grid-cols-2 gap-2">
            {visibleBands.map((band) => (
              <BandCheckbox
                key={band.id}
                band={band}
                checked={selectedBands.includes(band.id)}
                onToggle={() => toggleBand(band.id)}
              />
            ))}
          </div>
        </div>

        {/* Infrared Bands */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-muted-foreground">Infrared Bands</h4>
          <div className="grid grid-cols-2 gap-2">
            {infraredBands.map((band) => (
              <BandCheckbox
                key={band.id}
                band={band}
                checked={selectedBands.includes(band.id)}
                onToggle={() => toggleBand(band.id)}
              />
            ))}
          </div>
        </div>

        {/* Spectral Indices */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-muted-foreground">Spectral Indices</h4>
          <div className="grid grid-cols-2 gap-2">
            {indexBands.map((band) => (
              <BandCheckbox
                key={band.id}
                band={band}
                checked={selectedBands.includes(band.id)}
                onToggle={() => toggleBand(band.id)}
              />
            ))}
          </div>
        </div>

        {/* Selected Count */}
        <div className="pt-2 border-t border-border">
          <p className="text-xs text-muted-foreground">
            {selectedBands.length} band{selectedBands.length !== 1 ? "s" : ""} selected
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

function BandCheckbox({
  band,
  checked,
  onToggle,
}: {
  band: BandConfig;
  checked: boolean;
  onToggle: () => void;
}) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div
            className={`flex items-center gap-2 p-2 rounded border cursor-pointer transition-colors ${
              checked
                ? "border-primary bg-primary/10"
                : "border-border hover:border-primary/50"
            }`}
            onClick={onToggle}
          >
            <Checkbox checked={checked} className="pointer-events-none" />
            <span className="text-xs font-medium truncate">{band.name}</span>
            <Info className="h-3 w-3 text-muted-foreground ml-auto shrink-0" />
          </div>
        </TooltipTrigger>
        <TooltipContent side="right" className="max-w-[200px]">
          <p className="font-medium">{band.name}</p>
          <p className="text-xs text-muted-foreground">{band.description}</p>
          {band.wavelength && (
            <p className="text-xs text-muted-foreground">λ: {band.wavelength}</p>
          )}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
