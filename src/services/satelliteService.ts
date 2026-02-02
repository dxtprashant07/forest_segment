import { supabase } from "@/integrations/supabase/client";
import type { BoundingBox } from "@/types/aoi";

export type VisualizationType = "true_color" | "ndvi" | "ndwi" | "false_color" | "swir" | "nbr";
export type ModelType = 'segmentation' | 'loss_detection' | 'change_detection' | 'classification';

interface FetchSatelliteResponse {
  success?: boolean;
  image?: string;
  bbox?: BoundingBox;
  visualization?: VisualizationType;
  resolution?: number;
  dateRange?: {
    start: string;
    end: string;
  };
  error?: string;
  message?: string;
}

interface PredictionResponse {
  success?: boolean;
  prediction?: {
    mask?: string;
    forest_percentage?: number;
    confidence?: number;
    classes?: string[];
    loss_areas?: Array<{
      coordinates: number[][];
      area_km2: number;
    }>;
    demo_mode?: boolean;
  };
  bbox?: BoundingBox;
  modelType?: string;
  ml_available?: boolean;
  error?: string;
  message?: string;
}

export async function fetchSatelliteImagery(
  bbox: BoundingBox,
  startDate?: string,
  endDate?: string,
  visualization: VisualizationType = 'true_color',
  resolution: number = 30
): Promise<FetchSatelliteResponse> {
  const { data, error } = await supabase.functions.invoke<FetchSatelliteResponse>('fetch-satellite', {
    body: {
      bbox,
      startDate,
      endDate,
      visualization,
      resolution,
    },
  });

  if (error) {
    console.error('Error fetching satellite imagery:', error);
    // Check if the error contains the API not configured message
    const errorContext = error?.context;
    if (errorContext?.body) {
      try {
        const body = JSON.parse(errorContext.body);
        if (body.message) {
          return { error: body.error, message: body.message };
        }
      } catch {
        // Ignore JSON parse errors
      }
    }
    throw new Error(error.message || 'Failed to fetch satellite imagery');
  }

  return data as FetchSatelliteResponse;
}

export async function runPrediction(
  bbox: BoundingBox,
  modelType: ModelType = 'segmentation',
  forceNdvi: boolean = false,
  threshold: number = 128
): Promise<PredictionResponse> {
  const { data, error } = await supabase.functions.invoke<PredictionResponse>('ml-predict', {
    body: {
      bbox,
      modelType,
      forceNdvi,
      threshold,
    },
  });

  if (error) {
    console.error('Error running prediction:', error);
    throw new Error(error.message || 'Failed to run prediction');
  }

  return data as PredictionResponse;
}
