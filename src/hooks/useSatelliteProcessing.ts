import { useState, useCallback } from "react";
import { fetchSatelliteImagery, runPrediction, ModelType } from "@/services/satelliteService";
import type { BoundingBox } from "@/types/aoi";

export type ProcessingStep = 'idle' | 'fetching_imagery' | 'running_prediction' | 'complete' | 'error';
export type { ModelType };

export type PredictionMode = 'ndvi' | 'ml';

interface SatelliteState {
  step: ProcessingStep;
  satelliteImage: string | null;
  predictionResult: {
    mask?: string;
    forest_percentage?: number;
    confidence?: number;
    classes?: string[];
    loss_areas?: Array<{
      coordinates: number[][];
      area_km2: number;
    }>;
    demo_mode?: boolean;
  } | null;
  error: string | null;
  predictionMode: PredictionMode;
  isMLAvailable: boolean;
}

export function useSatelliteProcessing() {
  const [state, setState] = useState<SatelliteState>({
    step: 'idle',
    satelliteImage: null,
    predictionResult: null,
    error: null,
    predictionMode: 'ndvi',
    isMLAvailable: false,
  });

  const setPredictionMode = useCallback((mode: PredictionMode) => {
    setState(prev => ({ ...prev, predictionMode: mode }));
  }, []);

  const reset = useCallback(() => {
    setState(prev => ({
      step: 'idle',
      satelliteImage: null,
      predictionResult: null,
      error: null,
      predictionMode: prev.predictionMode,
      isMLAvailable: prev.isMLAvailable,
    }));
  }, []);

  const fetchImagery = useCallback(async (bbox: BoundingBox) => {
    setState(prev => ({ ...prev, step: 'fetching_imagery', error: null }));

    try {
      const response = await fetchSatelliteImagery(bbox);
      
      if (response.error) {
        setState(prev => ({
          ...prev,
          step: 'error',
          error: response.message || response.error,
        }));
        return null;
      }

      setState(prev => ({
        ...prev,
        step: 'idle',
        satelliteImage: response.image || null,
      }));

      return response.image;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch imagery';
      setState(prev => ({
        ...prev,
        step: 'error',
        error: errorMessage,
      }));
      return null;
    }
  }, []);

  const predict = useCallback(async (
    bbox: BoundingBox,
    modelType: ModelType = 'segmentation',
    forceNdvi: boolean = false,
    threshold: number = 128
  ) => {
    setState(prev => ({ ...prev, step: 'running_prediction', error: null }));

    try {
      const response = await runPrediction(bbox, modelType, forceNdvi, threshold);
      
      if (response.error) {
        setState(prev => ({
          ...prev,
          step: 'error',
          error: response.message || response.error,
        }));
        return null;
      }

      setState(prev => ({
        ...prev,
        step: 'complete',
        predictionResult: response.prediction || null,
        isMLAvailable: response.ml_available ?? prev.isMLAvailable,
      }));

      return response.prediction;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to run prediction';
      setState(prev => ({
        ...prev,
        step: 'error',
        error: errorMessage,
      }));
      return null;
    }
  }, []);

  const processAOI = useCallback(async (
    bbox: BoundingBox,
    modelType: ModelType = 'segmentation'
  ) => {
    // Step 1: Fetch satellite imagery
    const image = await fetchImagery(bbox);
    if (!image) return null;

    // Step 2: Run prediction
    const prediction = await predict(bbox, modelType);
    return prediction;
  }, [fetchImagery, predict]);

  return {
    ...state,
    isProcessing: state.step === 'fetching_imagery' || state.step === 'running_prediction',
    fetchImagery,
    predict,
    processAOI,
    reset,
    setPredictionMode,
  };
}
